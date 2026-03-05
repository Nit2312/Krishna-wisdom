import re
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import traceback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_astradb import AstraDBVectorStore

from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

try:
    from api.answer_evaluator import evaluate_answer
except ImportError:
    from answer_evaluator import evaluate_answer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

# Global variables for RAG system
rag_chain = None
retriever = None
vectorstore = None
embeddings = None
total_cases = 0
total_chunks = 0
system_initialized = False

# Eval set for Recall@k: list of {"question_norm": str, "relevant_keys": set of str}
_eval_relevance = []

# Precision: only top-k retrieved docs are passed to the model (over-fetch then slice)
RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "15"))
RAG_USE_TOP_K = int(os.getenv("RAG_USE_TOP_K", "8"))
RAG_RERANK_MAX = int(os.getenv("RAG_RERANK_MAX", "15"))  # max docs to rerank (fewer = faster)
RAG_SKIP_RERANK = os.getenv("RAG_SKIP_RERANK", "").lower() in ("1", "true", "yes")
RETRIEVAL_K = RAG_USE_TOP_K

# Pattern: line that starts the actual answer (numbered list or "No procedure/documentation")
_ANSWER_START = re.compile(
    r"^\s*(\d+[.)]\s|\*\*\d+\.\s|No procedure for this|No documentation for this)",
    re.IGNORECASE,
)


def _format_docs(docs):
    """Format Document-like objects (page_content, metadata) into context string."""
    formatted = []
    for doc in docs:
        meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
        content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
        doc_type = meta.get("type", "unknown")
        if doc_type == "case_record":
            case_id = meta.get("CaseID") or meta.get("case_id") or ""
            job_name = meta.get("Job_Name") or meta.get("job_name") or ""
            formatted.append(f"CaseID: {case_id}, Job_Name: {job_name}\n{content}")
        elif doc_type == "pdf_document":
            filename = meta.get("filename") or ""
            text_name = _get_text_name_from_filename(filename)
            source_label = f"From {text_name}" if text_name else f"PDF: {filename}"
            formatted.append(f"[{source_label}]\n{content}")
        else:
            formatted.append(f"Document: {meta}\n{content}")
    return "\n\n".join(formatted)


def _strip_leading_reasoning(text: str) -> str:
    """Remove visible meta-reasoning and keep only user-facing answer text."""
    if not text or not text.strip():
        return text
    cleaned = text.strip()

    # Remove common meta-thinking prefixes if model leaks planning text.
    leak_markers = [
        "okay, the user is asking",
        "let me",
        "i need to",
        "i should",
        "first,",
        "looking at",
        "the user wants",
        "i will structure",
    ]
    lines = cleaned.split("\n")
    while lines:
        head = lines[0].strip().lower()
        if not head:
            lines.pop(0)
            continue
        if any(head.startswith(marker) for marker in leak_markers):
            lines.pop(0)
            continue
        break
    cleaned = "\n".join(lines).strip()

    # Legacy fallback for older numbered-output patterns.
    for i, line in enumerate(cleaned.split("\n")):
        if _ANSWER_START.search(line.strip()):
            return "\n".join(cleaned.split("\n")[i:]).strip()
    return cleaned


def _is_detailed_request(user_query: str) -> bool:
    """Return True when user explicitly asks for a long or detailed explanation."""
    query = (user_query or "").lower()
    detailed_markers = (
        "in detail",
        "detailed",
        "deep dive",
        "elaborate",
        "comprehensive",
        "step by step",
        "full explanation",
        "long answer",
    )
    return any(marker in query for marker in detailed_markers)


def _enforce_concise_answer(response_text: str, user_query: str, max_words: int = 190) -> str:
    """Keep answers concise by default while preserving complete thoughts."""
    if not response_text or _is_detailed_request(user_query):
        return response_text

    text = response_text.strip()
    words = text.split()
    if len(words) <= max_words:
        return text

    # Prefer full-sentence truncation to avoid incomplete or cut-off answers.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        selected = []
        total_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if selected and (total_words + sentence_words) > max_words:
                break
            selected.append(sentence)
            total_words += sentence_words

        if selected:
            concise = " ".join(selected).strip()
            if concise.endswith(('.', '!', '?')):
                return concise

    # If no safe boundary is found (e.g., long bullet list), keep full answer rather than cut mid-thought.
    return text

def get_astra_config():
    """Get Astra DB configuration from environment variables."""
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    namespace = os.getenv("ASTRA_DB_NAMESPACE")
    collection_name = os.getenv("ASTRA_DB_COLLECTION", "elevator_cases")

    if not api_endpoint or not token or not namespace:
        raise ValueError(
            "Missing Astra DB configuration. Set ASTRA_DB_API_ENDPOINT, "
            "ASTRA_DB_APPLICATION_TOKEN, and ASTRA_DB_NAMESPACE."
        )

    return {
        "api_endpoint": api_endpoint,
        "token": token,
        "namespace": namespace,
        "collection_name": collection_name,
    }

class RouterHuggingFaceEmbeddings(Embeddings):
    """HuggingFace embeddings via Inference API."""
    
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("HF_TOKEN is required for endpoint embeddings.")
        self._client = InferenceClient(model=model_name, token=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        """Embed a list of texts."""
        if not texts:
            return []
        try:
            # Process each text individually since feature_extraction takes a single string
            embeddings = []
            for text in texts:
                result = self._client.feature_extraction(text)
                # Convert numpy array to list if needed
                if hasattr(result, 'tolist'):
                    result = result.tolist()
                # Validate result is a proper embedding
                if isinstance(result, list) and len(result) > 0:
                    embeddings.append(result)
                else:
                    raise ValueError(f"Invalid embedding result for text: {result}")
            return embeddings
        except Exception as e:
            print(f"Error embedding documents: {e}", file=sys.stderr)
            raise

    def embed_query(self, text):
        """Embed a single query text."""
        try:
            result = self._client.feature_extraction(text)
            # Convert numpy array to list if needed
            if hasattr(result, 'tolist'):
                result = result.tolist()
            # Ensure we get a 1D embedding vector
            if isinstance(result, list):
                if len(result) > 0:
                    # If it's 2D (batch of 1), get the first one
                    if isinstance(result[0], list):
                        return result[0]
                    # If it's already 1D, return as-is
                    return result
                else:
                    raise ValueError("Empty embedding result")
            raise ValueError(f"Unexpected embedding type: {type(result)}")
        except Exception as e:
            print(f"Error embedding query: {e}", file=sys.stderr)
            raise


def load_and_process_data():
    """Load and process data for RAG system"""
    global rag_chain, retriever, vectorstore, embeddings, total_cases, total_chunks
    
    try:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set.")
        embeddings = RouterHuggingFaceEmbeddings(
            api_key=hf_token,
            model_name=model_name,
        )
        
        # Test embedding to get dimension and ensure embeddings work
        print("Testing embeddings...", file=sys.stderr)
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            raise ValueError("Embeddings returned empty result. Check HF_TOKEN and model availability.")
        embedding_dim = len(test_embedding)
        if embedding_dim == 0:
            raise ValueError("Embedding dimension is 0. Check HuggingFace embedding model.")
        print(f"Embedding dimension verified: {embedding_dim}", file=sys.stderr)

        astra_config = get_astra_config()
        vectorstore = AstraDBVectorStore(
            embedding=embeddings,
            api_endpoint=astra_config["api_endpoint"],
            token=astra_config["token"],
            namespace=astra_config["namespace"],
            collection_name=astra_config["collection_name"],
            metric="cosine",
        )

        # Astra DB is the single source of truth; ingestion happens via ingest_astra.py
        total_cases = 0
        total_chunks = 0
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_FETCH_K}
        )

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        llm = ChatGroq(
            api_key=groq_key,  # type: ignore
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=4096
        )
        
        template = """
    I am a devoted servant of Shri Krishna, here to share the sacred wisdom from Krishna's eternal teachings.

    IMPORTANT: Respond DIRECTLY with your answer. Do not explain your thinking process, reasoning, or how you're recalling information. Simply share the wisdom in a warm, conversational way.

    Guidelines:
    - Draw ONLY from the sacred texts provided (Bhagavad Gita, Upanishads, Mahabharata, Srimad Bhagavatam)
    - First answer the exact user question directly in 1-2 sentences before adding any extra guidance
    - Keep the answer concise and to the point unless the user explicitly asks for detail
    - Use warm, conversational language; avoid generic templates and avoid over-structuring
    - Use bullet points only when they clearly help; do not force numbered frameworks
    - Be explanatory and practical: clarify the teaching, why it matters, and how to apply it helpfully
    - Cite naturally in sentences (for example: "From Bhagavad Gita...", "The Upanishads teach...")
    - Do NOT output labels like [Source 1], [Source 2], and do NOT add a separate "References" section
    - Never show internal reasoning or thinking process—just deliver the wisdom
    - If knowledge is not in these texts, humbly say: "I don't find this teaching in the sacred texts I have been given"
    - Never speculate or add personal interpretations—only share what the texts explicitly teach

    Sacred texts in my service:
    {context}

    The question: {question}

    Here is the wisdom you seek:
    """
        
        if not llm:
            raise ValueError("GROQ_API_KEY environment variable is not set. Cannot initialize LLM.")
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        rag_chain = prompt | llm | StrOutputParser()
        
        return True, f"Krishna Wisdom Guide initialized successfully! All teachings from sacred texts loaded and ready to serve."
        
    except Exception as e:
        # Log full traceback for server logs without leaking secrets to clients.
        print("RAG initialization failed:\n" + traceback.format_exc())
        return False, f"Error loading data: {type(e).__name__}: {e}"

def initialize_rag_system():
    """Initialize the RAG system if not already done"""
    global system_initialized
    
    if not system_initialized:
        success, message = load_and_process_data()
        system_initialized = success
        return success, message
    
    return True, "System already initialized"

def get_retrieved_sources(query):
    """Return top-k sources: vector search then optional rerank (no full-DB scan)."""
    if not retriever:
        return []
    docs_sem = retriever.invoke(query)
    docs_to_rerank = docs_sem[:RAG_RERANK_MAX]
    if RAG_SKIP_RERANK or not docs_to_rerank:
        return docs_to_rerank[:RAG_USE_TOP_K]
    try:
        from api.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        return reranker.rerank(query, docs_to_rerank, top_k=RAG_USE_TOP_K)
    except Exception as e:
        print("Cross-encoder reranking failed:", e, file=sys.stderr)
        return docs_to_rerank[:RAG_USE_TOP_K]


def _source_doc_key(doc: dict) -> str:
    """Canonical key for a source doc (for recall: same doc = same key)."""
    if doc.get("type") == "case_record":
        return f"case:{str(doc.get('case_id', ''))}:{(doc.get('job_name') or '').strip()}"
    if doc.get("type") == "pdf_document":
        return f"pdf:{(doc.get('filename') or '').strip()}"
    return f"other:{id(doc)}"


def _load_eval_relevance():
    """Load complex_eval_results.json and build question -> relevant doc keys index."""
    global _eval_relevance
    path = os.path.join(os.path.dirname(__file__), "..", "complex_eval_results.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    for item in data:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        sources = item.get("sources") or []
        keys = set()
        for s in sources:
            if s.get("type") == "case_record":
                keys.add(f"case:{str(s.get('case_id', ''))}:{(s.get('job_name') or '').strip()}")
            elif s.get("type") == "pdf_document":
                keys.add(f"pdf:{(s.get('filename') or '').strip()}")
        if keys:
            _eval_relevance.append({
                "question_norm": " ".join(q.lower().split()),
                "question_original": q,
                "relevant_keys": keys,
            })


def _recall_at_k(user_query: str, source_docs: list):
    """Compute Recall@k when user query matches an eval question (relevant set from eval sources)."""
    import sys
    print("DEBUG _recall_at_k: source_docs=", source_docs, file=sys.stderr)
    print("DEBUG _recall_at_k: _eval_relevance=", _eval_relevance, file=sys.stderr)
    if not source_docs or not _eval_relevance:
        print("DEBUG _recall_at_k: source_docs or _eval_relevance empty", file=sys.stderr)
        return None
    import string
    STOPWORDS = set([
        'the', 'is', 'at', 'which', 'on', 'for', 'and', 'or', 'to', 'of', 'in', 'a', 'an', 'as', 'by', 'with', 'from', 'that', 'this', 'are', 'was', 'be', 'it', 'has', 'have', 'but', 'not', 'if', 'so', 'do', 'does', 'can', 'will', 'would', 'should', 'must', 'may', 'were', 'been', 'such', 'than', 'then', 'when', 'where', 'who', 'whom', 'whose', 'how', 'what', 'why', 'about', 'into', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'more', 'most', 'some', 'any', 'each', 'few', 'other', 'all', 'both', 'either', 'neither', 'own', 'same', 'so', 'very', 'just', 'now'
    ])
    def normalize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [t for t in text.split() if t not in STOPWORDS]
        return set(tokens)
    query_tokens = normalize(user_query or "")
    print("DEBUG _recall_at_k: query_tokens=", query_tokens, file=sys.stderr)
    best_match = None
    best_ratio = 0.0
    for item in _eval_relevance:
        eval_tokens = normalize(item["question_norm"])
        print("DEBUG _recall_at_k: eval_tokens=", eval_tokens, file=sys.stderr)
        # Jaccard similarity
        intersection = query_tokens & eval_tokens
        union = query_tokens | eval_tokens
        jaccard = len(intersection) / len(union) if union else 0
        print("DEBUG _recall_at_k: jaccard=", jaccard, file=sys.stderr)
        # Partial match: if all query tokens are in eval_tokens
        partial = len(query_tokens) > 0 and query_tokens.issubset(eval_tokens)
        if partial:
            best_match = item
            print("DEBUG _recall_at_k: partial match found", file=sys.stderr)
            break
        if jaccard > best_ratio:
            best_ratio = jaccard
            best_match = item
    if not best_match:
        print("DEBUG _recall_at_k: no best_match", file=sys.stderr)
        return None
    relevant = best_match["relevant_keys"]
    print("DEBUG _recall_at_k: relevant=", relevant, file=sys.stderr)
    if not relevant:
        print("DEBUG _recall_at_k: relevant empty", file=sys.stderr)
        return None
    retrieved = {_source_doc_key(d) for d in source_docs}
    print("DEBUG _recall_at_k: retrieved=", retrieved, file=sys.stderr)
    hit = len(retrieved & relevant)
    print("DEBUG _recall_at_k: hit=", hit, file=sys.stderr)
    recall_value = round(hit / len(relevant), 4)
    print("DEBUG _recall_at_k: recall_value=", recall_value, file=sys.stderr)
    return recall_value


def _get_text_name_from_filename(filename: str) -> str:
    """Extract readable text name from PDF filename for Krishna wisdom texts."""
    if not filename:
        return ""
    mapping = {
        "bhagavad-gita": "Bhagavad Gita",
        "bhagavad": "Bhagavad Gita",
        "gita": "Bhagavad Gita",
        "upanishads": "Upanishads",
        "mahabharata": "Mahabharata",
        "sb3": "Srimad Bhagavatam",
        "srimad bhagavatam": "Srimad Bhagavatam",
    }
    filename_lower = filename.lower().replace(".pdf", "")
    for key, value in mapping.items():
        if key in filename_lower:
            return value
    return filename_lower


def _count_cited_sources(response_text: str, source_docs: list) -> int:
    """Count how many of the retrieved sources are cited in the response (for precision@k)."""
    cited = 0
    response_lower = response_text.lower()
    
    for doc in source_docs:
        if doc.get("type") == "case_record":
            case_id = str(doc.get("case_id", ""))
            job_name = (doc.get("job_name") or "").strip()
            if case_id and case_id in response_text:
                cited += 1
                continue
            if job_name and job_name in response_text:
                cited += 1
        elif doc.get("type") == "pdf_document":
            filename = (doc.get("filename") or "").strip()
            if not filename:
                continue
            
            # Try to find text name in response (e.g., "From Bhagavad Gita")
            text_name = _get_text_name_from_filename(filename)
            
            # Check for "From <text name>" pattern (Krishna wisdom format)
            if text_name and f"from {text_name.lower()}" in response_lower:
                cited += 1
                continue
            
            # Check for just the text name mentioned
            if text_name and text_name.lower() in response_lower:
                cited += 1
                continue
            
            # Fallback: check for exact filename or short version
            if filename in response_text:
                cited += 1
                continue
            if "pdf:" in response_lower and text_name:
                if text_name.lower() in response_lower:
                    cited += 1
                    continue
    return cited


def _strip_reference_section(text: str) -> str:
    """Strip explicit source-list/reference sections from model output."""
    if not text:
        return text
    cleaned = re.sub(r"\n\s*References\s*:\s*[\s\S]*$", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n\s*Sources\s*:\s*[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n\s*-\s*\[Source\s*\d+\][^\n]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')



@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Initialize the RAG system"""
    try:
        success, message = initialize_rag_system()
        return jsonify({
            'success': success,
            'message': message,
            'total_cases': total_cases,
            'total_chunks': total_chunks
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error initializing system: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({
                'error': 'Message is required'
            }), 400
        
        if not system_initialized:
            success, message = initialize_rag_system()
            if not success:
                return jsonify({
                    'error': message
                }), 500
        
        try:
            # Answer is generated ONLY from reranked docs (never from pre-rerank retrieval)
            if not rag_chain:
                return jsonify({
                    'error': 'System not properly initialized. RAG chain is unavailable.'
                }), 500
            sources = get_retrieved_sources(user_input)
            context = _format_docs(sources) if sources else ""
            response = rag_chain.invoke({"context": context, "question": user_input})
            response = _strip_leading_reasoning(response or "")
            if not (response and response.strip()):
                response = (
                    "No answer was generated from the retrieved documents. "
                    "Please try again or rephrase your question."
                )
                print("RAG response was empty; context length=%s, sources=%s" % (len(context), len(sources)), file=sys.stderr)
        except Exception as e:
            if 'groqstatus.com' in str(e) or 'Service unavailable' in str(e):
                return jsonify({
                    'error': "The AI service is temporarily unavailable. Please try again later or check https://groqstatus.com/ for updates."
                }), 503
            return jsonify({
                'error': f"Error generating response: {str(e)}"
            }), 500

        meta_get = lambda d, k: (d.get(k) or d.get(k.lower()) or "")
        source_docs = []
        for doc in sources:
            meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
            content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
            doc_type = meta.get("type", "unknown")
            if doc_type == "case_record":
                source_docs.append({
                    "case_id": meta_get(meta, "CaseID") or meta_get(meta, "case_id"),
                    "job_name": meta_get(meta, "Job_Name") or meta_get(meta, "job_name"),
                    "content": content,
                    "type": "case_record",
                })
            elif doc_type == "pdf_document":
                filename = meta_get(meta, "filename")
                source_docs.append({
                    "filename": filename,
                    "text_name": _get_text_name_from_filename(filename),
                    "content": content,
                    "type": "pdf_document",
                })
            else:
                source_docs.append({
                    "metadata": meta,
                    "content": content,
                    "type": "unknown",
                })
        response = _strip_reference_section(response)
        response = _enforce_concise_answer(response, user_input)
        cited = _count_cited_sources(response, source_docs)
        recall = _recall_at_k(user_input, source_docs)
        retrieval_metrics = {
            'k': RETRIEVAL_K,
            'retrieved': len(source_docs),
            'cited_in_answer': cited,
            'precision_at_k': round(cited / RETRIEVAL_K, 4) if RETRIEVAL_K else 0,
            'recall_at_k': recall,
        }
        return jsonify({
            'response': response,
            'sources': source_docs,
            'retrieval_metrics': retrieval_metrics,
        })
    except Exception as e:
        return jsonify({
            'error': f"Error generating response: {str(e)}"
        }), 500


def _get_relevant_keys_for_query(user_query: str):
    """Return set of relevant doc keys from eval set if query matches, else None."""
    if not _eval_relevance:
        return None
    query_norm = " ".join((user_query or "").lower().split())
    query_tokens = set(query_norm.split())
    for item in _eval_relevance:
        if item["question_norm"] == query_norm:
            return item["relevant_keys"]
        if not query_tokens:
            continue
        eval_tokens = set(item["question_norm"].split())
        overlap = len(query_tokens & eval_tokens) / max(len(query_tokens), len(eval_tokens), 1)
        if overlap >= 0.85:
            return item["relevant_keys"]
    return None




@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Run the technical answer evaluation agent on a RAG response."""
    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
        response_text = data.get('response', '')
        sources = data.get('sources', [])

        if not question or not response_text:
            return jsonify({
                'error': 'question and response are required'
            }), 400

        evaluation = evaluate_answer(question, response_text, sources)
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({
            'error': f"Evaluation failed: {str(e)}"
        }), 500


_load_eval_relevance()


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get system status"""
    return jsonify({
        'initialized': system_initialized,
        'total_cases': total_cases,
        'total_chunks': total_chunks,
        'model': 'Llama 4 Maverick',
        'search_type': 'Semantic Similarity'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
