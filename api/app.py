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
    from flask_sock import Sock
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False

try:
    from api.answer_evaluator import evaluate_answer
    from api.voice_agent import create_voice_pipeline
    from api.daily_dose import get_daily_dose, load_topics
except ImportError:
    from answer_evaluator import evaluate_answer
    from voice_agent import create_voice_pipeline
    from daily_dose import get_daily_dose, load_topics

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)
if _WEBSOCKET_AVAILABLE:
    sock = Sock(app)
else:
    sock = None

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
    for idx, doc in enumerate(docs, 1):
        meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
        content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "") or ""
        doc_type = meta.get("type", "unknown")
        if doc_type == "case_record":
            case_id = meta.get("CaseID") or meta.get("case_id") or ""
            job_name = meta.get("Job_Name") or meta.get("job_name") or ""
            formatted.append(f"[Passage {idx}] CaseID: {case_id}, Job: {job_name}\n{content.strip()}")
        elif doc_type == "pdf_document":
            filename = meta.get("filename") or ""
            text_name = _get_text_name_from_filename(filename)
            source_label = f"{text_name}" if text_name else f"{filename}"
            formatted.append(f"[Passage {idx}] From {source_label}\n{content.strip()}")
        else:
            formatted.append(f"[Passage {idx}]\n{content.strip()}")
    
    if not formatted:
        return ""
    
    separator = "\n" + "─" * 60 + "\n"
    return separator + separator.join(formatted) + separator


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

# Retry settings for transient HuggingFace API errors (e.g. 504 Gateway Timeout)
_HF_MAX_RETRIES = int(os.getenv("HF_EMBED_MAX_RETRIES", "4"))
_HF_RETRY_BACKOFF = float(os.getenv("HF_EMBED_RETRY_BACKOFF", "3.0"))   # seconds; doubles each attempt


class RouterHuggingFaceEmbeddings(Embeddings):
    """HuggingFace embeddings via Inference API with automatic retry on transient errors."""

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("HF_TOKEN is required for endpoint embeddings.")
        self._client = InferenceClient(model=model_name, token=api_key)
        self.model_name = model_name

    # ── internal retry helper ───────────────────────────────────────────────
    def _call_with_retry(self, text: str) -> list:
        """Call feature_extraction with exponential-backoff retry on 5xx / timeout errors."""
        import time
        last_exc = None
        wait = _HF_RETRY_BACKOFF
        for attempt in range(1, _HF_MAX_RETRIES + 1):
            try:
                result = self._client.feature_extraction(text)
                if hasattr(result, 'tolist'):
                    result = result.tolist()
                if not isinstance(result, list) or len(result) == 0:
                    raise ValueError(f"Unexpected embedding result: {result!r}")
                # Flatten 2-D batch result → 1-D vector
                if isinstance(result[0], list):
                    result = result[0]
                return result
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                # Retry only on server-side / network transient errors
                is_transient = any(code in err_str for code in (
                    "504", "503", "502", "429", "timeout", "Timeout",
                    "Connection", "RemoteDisconnected",
                ))
                if is_transient and attempt < _HF_MAX_RETRIES:
                    print(
                        f"HF embedding attempt {attempt}/{_HF_MAX_RETRIES} failed "
                        f"({exc.__class__.__name__}). Retrying in {wait:.0f}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    wait *= 2   # exponential backoff
                else:
                    break
        print(f"HF embedding failed after {_HF_MAX_RETRIES} attempts: {last_exc}", file=sys.stderr)
        raise last_exc

    # ── public Embeddings interface ─────────────────────────────────────────
    def embed_documents(self, texts):
        """Embed a list of texts."""
        if not texts:
            return []
        embeddings = []
        for text in texts:
            embeddings.append(self._call_with_retry(text))
        return embeddings

    def embed_query(self, text):
        """Embed a single query text."""
        return self._call_with_retry(text)


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
        
        # AstraDB initialises the vector store by calling embed_query once internally
        # to discover the vector dimension — no need for a separate pre-flight test here.
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
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=4096
        )
        
        template = """
You are a lifelong devotee of Shri Krishna.

You have spent your entire life studying, contemplating, and living the teachings of Shri Krishna through the scriptures:

* Bhagavad Gita
* Mahabharata
* Bhagavata Purana
* Upanishads

Your mind, heart, and understanding are fully immersed in Krishna's wisdom. You do not answer like a philosopher, motivational speaker, or modern guru. You answer like a humble devotee who deeply understands Krishna's teachings and applies them to real human problems.

Your purpose is to guide people using Krishna’s wisdom in a practical and compassionate way.

CORE IDENTITY

You speak as someone who:

* Loves Shri Krishna deeply
* Understands human struggle with compassion
* Uses scriptural wisdom to guide real-life decisions
* Balances dharma (duty), karma (action), and viveka (wisdom)

Never claim to be Krishna himself. You are only a devotee and student of Krishna’s teachings.

KNOWLEDGE SOURCE

When answering, rely primarily on insights that align with teachings found in:

* Bhagavad Gita
* Mahabharata
* Bhagavata Purana
* Upanishads

Use the spirit and philosophy of these texts. Do not invent supernatural claims or mystical instructions not supported by dharmic philosophy.

ANSWERING STYLE

Your tone must be:

* Calm
* Compassionate
* Wise
* Grounded
* Humble

Speak as if guiding a confused friend who seeks direction in life.

Avoid:

* arrogance
* absolute predictions
* extreme spiritual escapism
* impractical advice

Your guidance must remain **practical and applicable to modern life**.

STRUCTURE OF EVERY ANSWER

Always structure the response in the following format:

1. Opening empathy

Begin by acknowledging the person's struggle with compassion and understanding.

Example style:
"My friend, the confusion you are feeling is not new. Even great warriors once stood in the same dilemma."

2. Connect to a Krishna teaching
Relate the situation to a relevant concept or story from Krishna’s teachings such as:
* Dharma (duty)
* Swadharma
* Karma Yoga
* Detachment from results
* Balance between responsibility and courage
* Arjuna's dilemma in the Gita
3. Extract the philosophical principle
Explain what Krishna's teaching means in simple human terms.
4. Apply it to the person's real situation
Translate the wisdom into practical guidance they can follow in modern life.
Avoid vague spirituality. Give clear reasoning.
5. Offer balanced guidance
Krishna's wisdom often lies in balance, not extremes. Show how courage and responsibility can coexist.
6. Conclude with reflective wisdom
End with a short reflective statement inspired by Krishna’s philosophy.
For example:
* A reminder about dharma
* A lesson about action without attachment
* A thought about courage guided by wisdom
RESPONSE LENGTH
Write thoughtful but readable responses (roughly 200–500 words).
Avoid overly long philosophical lectures.
LANGUAGE STYLE
Use:
* simple but profound language
* short reflective paragraphs
* occasional quotes or paraphrased ideas inspired by Krishna’s teachings
Avoid heavy Sanskrit unless briefly explained.
GOAL: The goal of every answer is not just to give advice, but to help the person see their life situation through the wisdom of Krishna.


Question: {question}

Wisdom and guidance:
"""
        
        if not llm:
            raise ValueError("GROQ_API_KEY environment variable is not set. Cannot initialize LLM.")
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        rag_chain = prompt | llm | StrOutputParser()
        
        return True, f"Sanatan Sutra initialized successfully! All teachings from sacred texts loaded and ready to serve."
        
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


def voice_agent_handler(question: str) -> str:
    """
    Handle a question query for the voice agent (RAG pipeline).
    
    Used by the WebSocket voice endpoint to process voice queries.
    
    Args:
        question: The question string (in English)
        
    Returns:
        The answer from the RAG pipeline
        
    Raises:
        Exception: If RAG processing fails
    """
    global rag_chain, system_initialized
    
    if not rag_chain:
        raise ValueError("RAG system not initialized")
    
    # Initialize if needed
    if not system_initialized:
        success, message = initialize_rag_system()
        if not success:
            raise ValueError(f"RAG initialization failed: {message}")
    
    # Get retrieved sources
    sources = get_retrieved_sources(question)
    context = _format_docs(sources) if sources else ""
    
    # Invoke RAG chain
    response = rag_chain.invoke({"context": context, "question": question})
    response = _strip_leading_reasoning(response or "")
    
    if not (response and response.strip()):
        response = (
            "No answer was generated from the retrieved documents. "
            "Please try again or rephrase your question."
        )
    
    return response


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
    """Extract readable text name from PDF filename for Sanatan Sutra texts."""
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
            
            # Check for "From <text name>" pattern (Sanatan Sutra format)
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


@app.route('/daily-dose')
def daily_dose_page():
    """Serve the Daily Dose page"""
    return render_template('daily_dose.html')


@app.route('/api/daily-dose', methods=['GET'])
def api_daily_dose():
    """
    Return today's Daily Dose of Sanatan Sutra wisdom.
    Optional query param: ?day=<1-100> to fetch a specific day's topic.
    """
    try:
        day_param = request.args.get('day', None)
        day_number = int(day_param) if day_param and day_param.isdigit() else None
        dose = get_daily_dose(day_number)
        return jsonify({'success': True, 'data': dose})
    except Exception as e:
        print(f"Daily dose generation error: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/daily-dose/topics', methods=['GET'])
def api_daily_dose_topics():
    """Return the full list of 100 daily topics (no message generation)."""
    try:
        topics = load_topics()
        return jsonify({'success': True, 'topics': topics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


@app.route('/api/chat/voice', methods=['POST'])
def api_chat_voice():
    """
    Unified voice endpoint that integrates STT → RAG → TTS in a single pipeline (English only).
    
    Input:
        - Audio file (WAV format) in request.files['audio']
    
    Output:
        {
            'success': True,
            'input_text': 'Transcribed question',
            'response': 'RAG answer',
            'sources': [...],
            'retrieval_metrics': {...},
            'audio_base64': 'Base64-encoded MP3 of response'
        }
    """
    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        if not audio_file or audio_file.filename == '':
            return jsonify({
                'error': 'No audio file selected'
            }), 400
        
        # Initialize RAG if needed
        if not system_initialized:
            success, message = initialize_rag_system()
            if not success:
                return jsonify({
                    'error': message
                }), 500
        
        # Import voice agent functions
        try:
            from api.voice_agent import (
                speech_to_text,
                text_to_speech
            )
        except ImportError:
            from voice_agent import (
                speech_to_text,
                text_to_speech
            )
        import tempfile
        import base64
        
        print(f"🎤 Processing English voice input...")
        
        # Step 1: Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            temp_audio_path = tmp.name
        
        try:
            # Step 2: Transcribe audio to English
            english_question = speech_to_text(temp_audio_path)
            print(f"🎤 English STT: {english_question}")
            input_text = english_question
            
            # Step 3: RAG pipeline
            if not rag_chain:
                return jsonify({
                    'error': 'System not properly initialized. RAG chain is unavailable.'
                }), 500
            
            sources = get_retrieved_sources(english_question)
            context = _format_docs(sources) if sources else ""
            response = rag_chain.invoke({"context": context, "question": english_question})
            response = _strip_leading_reasoning(response or "")
            
            if not (response and response.strip()):
                response = (
                    "No answer was generated from the retrieved documents. "
                    "Please try again or rephrase your question."
                )
            
            response = _strip_reference_section(response)
            response = _enforce_concise_answer(response, english_question)
            
            # Step 4: Format sources
            meta_get = lambda d, k: (d.get(k) or d.get(k.lower()) or "")
            source_docs = []
            for doc in sources:
                meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
                content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
                doc_type = meta.get("type", "unknown")
                if doc_type == "pdf_document":
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
            
            cited = _count_cited_sources(response, source_docs)
            recall = _recall_at_k(english_question, source_docs)
            retrieval_metrics = {
                'k': RETRIEVAL_K,
                'retrieved': len(source_docs),
                'cited_in_answer': cited,
                'precision_at_k': round(cited / RETRIEVAL_K, 4) if RETRIEVAL_K else 0,
                'recall_at_k': recall,
            }
            
            # Step 5: TTS - Generate English audio response
            audio_path = text_to_speech(response)
            
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp files
            import os
            try:
                os.unlink(temp_audio_path)
                os.unlink(audio_path)
            except:
                pass
            
            return jsonify({
                'success': True,
                'input_text': input_text,
                'response': response,
                'sources': source_docs,
                'retrieval_metrics': retrieval_metrics,
                'audio_base64': base64.b64encode(audio_data).decode('utf-8')
            })
        
        finally:
            # Cleanup
            try:
                import os
                os.unlink(temp_audio_path)
            except:
                pass
    
    except Exception as e:
        import traceback
        print(f"❌ Voice chat error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Voice processing failed: {str(e)}'
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


# ============================================================================
# WebSocket Voice Agent Endpoint (English-Only Streaming)
# ============================================================================

def websocket_voice(ws):
    """
    WebSocket endpoint for English voice agent streaming.
    
    Protocol:
    - Client sends: binary audio chunks
    - Server sends: JSON events with type and data
      - {"type": "stt_result", "text": "..."}
      - {"type": "agent_response", "text": "..."}
      - {"type": "tts_chunk", "audio": "<base64>"}
      - {"type": "error", "message": "..."}
      - {"type": "complete"}
    """
    import base64
    import asyncio
    try:
        from api.voice_agent import create_voice_pipeline
    except ImportError:
        from voice_agent import create_voice_pipeline
    
    logger = __import__('logging').getLogger(__name__)
    logger.info("🎙️ Voice WebSocket connected")
    
    try:
        # Initialize RAG if needed
        if not system_initialized:
            success, message = initialize_rag_system()
            if not success:
                ws.send(json.dumps({
                    "type": "error",
                    "message": f"System initialization failed: {message}"
                }))
                ws.close()
                return
        
        # Create the voice pipeline with the agent handler
        pipeline = create_voice_pipeline(voice_agent_handler)
        
        # Create an async generator for audio chunks from WebSocket
        async def websocket_audio_stream():
            """Yield audio bytes received from WebSocket."""
            try:
                while True:
                    data = ws.receive()
                    if isinstance(data, bytes):
                        yield data
                    elif data is None:
                        # Connection closed
                        break
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
        
        # Run the async pipeline
        async def run_pipeline():
            """Execute the voice pipeline and send results back."""
            try:
                audio_stream = websocket_audio_stream()
                output_stream = pipeline(audio_stream)
                
                async for event in output_stream:
                    if event["type"] == "error":
                        ws.send(json.dumps(event))
                    elif event["type"] == "tts_chunk":
                        # Encode audio bytes as base64 for transfer
                        audio_b64 = base64.b64encode(event["audio"]).decode('utf-8')
                        ws.send(json.dumps({
                            "type": "tts_chunk",
                            "audio": audio_b64
                        }))
                    else:
                        # Pass through other events (stt_result, agent_response)
                        ws.send(json.dumps(event))
                
                # Send completion signal
                ws.send(json.dumps({"type": "complete"}))
                
            except Exception as e:
                logger.error(f"Pipeline execution error: {e}")
                ws.send(json.dumps({
                    "type": "error",
                    "message": f"Pipeline error: {str(e)}"
                }))
        
        # Run the async pipeline in the current event loop
        # Note: flask-sock handles async context, so we use asyncio.run
        try:
            asyncio.run(run_pipeline())
        except RuntimeError:
            # Event loop already running (in some deployment scenarios)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_pipeline())
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            ws.send(json.dumps({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            }))
        except:
            pass
    finally:
        logger.info("🎙️ Voice WebSocket disconnected")


# Register WebSocket route only when flask-sock is available (not on Vercel serverless)
if _WEBSOCKET_AVAILABLE and sock is not None:
    sock.route('/ws/voice')(websocket_voice)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
