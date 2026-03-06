"""
AI agent that evaluates RAG pipeline answers for Krishna Wisdom spiritual questions.
Checks faithfulness to Hindu scripture sources, completeness, relevance, and citation quality.
"""

import json
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator of answers grounded in Hindu sacred texts — specifically the Bhagavad Gita, Upanishads, Mahabharata, and Srimad Bhagavatam (Bhagavata Purana). Your job is to evaluate whether an AI assistant's answer to a spiritual or life-guidance question is correct, well-grounded, and helpful.

You will receive:
1. The user's question
2. The RAG assistant's answer
3. The source document excerpts that were retrieved (from PDFs of Hindu scriptures)

IMPORTANT CONTEXT — THE VALID SOURCE TEXTS:
The system's knowledge base contains ONLY these four texts:
- Bhagavad Gita (file: Bhagavad-gita-As-It-Is.pdf)
- 108 Upanishads (file: 108upanishads.pdf)
- Mahabharata (file: Mahabharata (Unabridged in English).pdf)
- Srimad Bhagavatam / Bhagavata Purana (file: SB3.1.pdf)

Do NOT penalise the answer for referencing concepts that are intrinsic to these four texts, even if those exact passages were not in the retrieved excerpts. For example:
- "Swadharma", "Karma Yoga", "Nishkama Karma", "Arjuna's dilemma" are core Bhagavad Gita concepts — acceptable references.
- "Atman", "Brahman", "Self-realisation", "Maya" are central Upanishadic concepts — acceptable references.
- Stories from the Mahabharata or Bhagavatam are valid to reference by name.
The retrieved excerpts are a SAMPLE of what was found; the model may legitimately draw on the deeper philosophical spirit of these texts.

Evaluate the answer on these criteria:

**Faithfulness**: Are the philosophical claims consistent with the spirit and teachings of the four source texts? Flag any clear misrepresentation, invented scripture, or concepts that contradict these traditions. Do NOT flag authentic Hindu philosophical concepts just because the exact verse is unseen in the retrieved chunks.

**Answer relevance**: Does the answer directly address the person's real-world problem? A good answer applies scripture practically to modern life — not just quotes philosophy in the abstract.

**Completeness**: Does the answer acknowledge the person's struggle, connect it to relevant teachings, and give practical guidance? Generic platitudes with no real application are a weakness.

**Appropriate scope**: If the question is clearly out of scope (e.g. cooking recipes, sports, coding, finance), the answer should politely decline and not invent spiritual guidance. Penalise answers that fabricate a spiritual connection to completely unrelated topics.

**Citation quality**: Are relevant source texts named (e.g. "From the Bhagavad Gita", "The Upanishads teach...") where appropriate? Over-citing or robotically listing PDF filenames is not required — natural, readable source attribution is preferred.

You must also provide:
- **faithfulness** (0-100): How well does the answer align with the authentic teachings of these four Hindu scriptures? 100 = perfectly grounded; 0 = contradicts or fabricates scripture.
- **answer_relevance** (0-100): How directly and practically does the answer address the user's question? 100 = fully on-point with actionable guidance; 0 = irrelevant or off-topic.

Output a JSON object only, no other text, with this exact structure:
{
  "verdict": "pass" | "warning" | "fail",
  "score": <number 0-100>,
  "faithfulness": <number 0-100>,
  "answer_relevance": <number 0-100>,
  "summary": "<one sentence overall assessment>",
  "issues": ["<issue 1>", "<issue 2>", ...],
  "suggestions": ["<suggestion 1>", ...],
  "strengths": ["<strength 1>", ...]
}

Use "pass" when the answer is spiritually accurate, practically helpful, and well-grounded; "warning" when mostly good but with minor gaps, vague application, or one questionable claim; "fail" when the answer clearly misrepresents scripture, gives harmful or irresponsible guidance, or fabricates concepts not found in Hindu philosophy.

Respond with ONLY a single JSON object. No markdown, no code fence, no explanation before or after. Valid JSON only.
"""



def _format_sources(sources: List[Dict[str, Any]]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        if s.get("type") == "case_record":
            parts.append(f"[Source {i}] CaseID: {s.get('case_id', '')}, Job: {s.get('job_name', '')}\n{s.get('content', '')}")
        elif s.get("type") == "pdf_document":
            parts.append(f"[Source {i}] PDF: {s.get('filename', '')}\n{s.get('content', '')}")
        else:
            parts.append(f"[Source {i}]\n{s.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def _parse_evaluation_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    raw = _extract_json_object(text)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
    return {
        "verdict": "warning",
        "score": 50,
        "faithfulness": 50,
        "answer_relevance": 50,
        "summary": "Evaluation could not be parsed; manual review recommended.",
        "issues": [text[:500] if text else "No structured evaluation returned."],
        "suggestions": [],
        "strengths": [],
    }


def evaluate_answer(question: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run the evaluation agent on a RAG answer.

    Args:
        question: The user's spiritual or life-guidance question.
        response: The RAG pipeline's answer.
        sources: List of source dicts with type, content, and type-specific fields (filename, etc.).

    Returns:
        Dict with verdict, score, faithfulness, answer_relevance, summary, issues, suggestions, strengths.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "verdict": "fail",
            "score": 0,
            "summary": "Evaluation unavailable: GROQ_API_KEY not set.",
            "issues": ["Missing GROQ_API_KEY"],
            "suggestions": [],
            "strengths": [],
        }

    llm = ChatGroq(
        api_key=api_key,
        model_name="qwen/qwen3-32b",
        temperature=0,
        max_tokens=1024,
    )

    sources_text = _format_sources(sources) if sources else "(No sources provided.)"
    user_content = f"""Question:
{question}

RAG Assistant's Answer:
{response}

Source Documents:
{sources_text}

Evaluate the answer and respond with only the JSON object."""

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    result = llm.invoke(messages)
    response_text = result.content if hasattr(result, "content") else str(result)
    parsed = _parse_evaluation_response(response_text)

    if parsed.get("summary") == "Evaluation could not be parsed; manual review recommended.":
        fallback_content = f"""Question: {question[:200]}
Answer (excerpt): {response[:600]}...

Reply with ONLY this JSON, no other text:
{{"verdict":"pass|warning|fail","score":0-100,"faithfulness":0-100,"answer_relevance":0-100,"summary":"one sentence"}}"""
        retry_result = llm.invoke([
            SystemMessage(content="Reply with only a JSON object: verdict, score, faithfulness, answer_relevance, summary. No markdown."),
            HumanMessage(content=fallback_content),
        ])
        retry_text = retry_result.content if hasattr(retry_result, "content") else str(retry_result)
        retry_parsed = _parse_evaluation_response(retry_text)
        if retry_parsed.get("score") is not None or retry_parsed.get("faithfulness") is not None:
            parsed = retry_parsed

    if "verdict" not in parsed or parsed["verdict"] not in ("pass", "warning", "fail"):
        parsed["verdict"] = "warning"
    else:
        parsed["verdict"] = str(parsed["verdict"]).lower()
    if "score" not in parsed or not isinstance(parsed["score"], (int, float)):
        parsed["score"] = 50
    parsed["score"] = max(0, min(100, int(parsed["score"])))
    for key in ("faithfulness", "answer_relevance"):
        if key not in parsed or not isinstance(parsed[key], (int, float)):
            parsed[key] = 50
        parsed[key] = max(0, min(100, int(parsed[key])))
    parsed.setdefault("summary", "")
    parsed.setdefault("issues", [])
    parsed.setdefault("suggestions", [])
    parsed.setdefault("strengths", [])

    return parsed
