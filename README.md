# Krishna Wisdom Guide (Flask + RAG)

Krishna Wisdom Guide is a Flask-based Retrieval-Augmented Generation (RAG) application that answers spiritual questions using only the sacred texts available in the local `data/` folder.

The system retrieves relevant passages from Astra DB, then generates a concise and conversational answer with natural source mention (for example: “From Bhagavad Gita…”).

## What this project does

- Answers questions from:
	- Bhagavad Gita
	- Upanishads
	- Mahabharata
	- Srimad Bhagavatam
- Uses semantic retrieval from Astra DB vector store
- Uses Groq-hosted LLM for final response generation
- Uses Hugging Face embeddings (`sentence-transformers/all-mpnet-base-v2`)
- Includes a Krishna-themed web chat UI
- **NEW: Gujarati Voice Agent** - speak your question in Gujarati, get spoken answers back!

## Tech stack

- Flask + Flask-CORS
- LangChain (`langchain`, `langchain-core`, `langchain-community`)
- Astra DB (`langchain-astradb`, `astrapy`)
- Groq (`langchain-groq`)
- Hugging Face Inference API (`huggingface-hub`)

## Project structure

- [app.py](app.py) — Flask entrypoint (imports app from API module)
- [api/app.py](api/app.py) — Main API, RAG pipeline, prompting, answer processing
- [ingest_astra.py](ingest_astra.py) — Ingests PDFs (and optional Excel data) into Astra DB
- [templates/index.html](templates/index.html) — Main chat page
- [static/css/style.css](static/css/style.css) — UI styling
- [static/js/app.js](static/js/app.js) — Frontend chat logic
- [data/](data/) — Source PDFs used for retrieval

## Prerequisites

- Python 3.10+
- Astra DB account + vector-capable database
- Groq API key
- Hugging Face token with access to inference API

## Environment variables

Create a `.env` file in the project root with:

```env
# Astra DB
ASTRA_DB_API_ENDPOINT=...
ASTRA_DB_APPLICATION_TOKEN=...
ASTRA_DB_NAMESPACE=...
ASTRA_DB_COLLECTION=elevator_cases

# LLM / Embeddings
GROQ_API_KEY=...
HF_TOKEN=...

# Optional retrieval tuning
RAG_FETCH_K=15
RAG_USE_TOP_K=8
RAG_RERANK_MAX=15
RAG_SKIP_RERANK=false
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data ingestion

1. Put your sacred text PDFs in `data/`.
2. Run ingestion:

```bash
python ingest_astra.py
```

This script extracts text, splits into chunks, and uploads embeddings/documents to Astra DB.

## Run the app

```bash
source .venv/bin/activate
python app.py
```

Open: `http://127.0.0.1:5000`

## API endpoints

- `GET /` — chat UI
- `GET /api/status` — initialization status
- `POST /api/initialize` — initialize RAG chain and retriever
- `POST /api/chat` — ask a question (text)
- `POST /api/voice-chat` — ask a question (Gujarati voice)
- `POST /api/evaluate` — run answer evaluation helper

## Voice Agent Feature

The system now supports Gujarati voice interaction:

**How it works:**
1. Click the microphone button in the chat UI
2. Speak your question in Gujarati
3. System transcribes speech → translates to English → runs RAG → translates answer to Gujarati → generates Gujarati speech
4. Hear the answer spoken back in Gujarati

**Pipeline:**
- **Speech-to-Text**: Whisper (via Groq) for Gujarati transcription
- **Translation**: Groq LLM for Gujarati ↔ English
- **Text-to-Speech**: AI4Bharat Indic-TTS for Gujarati speech synthesis

**Requirements:**
- Microphone access in browser
- Additional dependencies: `transformers`, `torch`, `soundfile`, `TTS`

## Response behavior

- Answers are concise by default.
- If user asks for detailed explanation, longer output is allowed.
- Internal reasoning text is stripped from output.
- Mechanical `References:` blocks are removed from final answer formatting.

## Troubleshooting

- **Embedding errors / dimension issues**
	- Check `HF_TOKEN`
	- Ensure model access and valid network connectivity
- **Astra DB init failure**
	- Verify endpoint, token, namespace, and collection values
- **Empty or weak answers**
	- Re-run ingestion after updating files in `data/`
	- Tune `RAG_FETCH_K` / `RAG_USE_TOP_K`

## Notes

- Primary knowledge source is the indexed content from `data/` PDFs.
- Optional Excel ingestion exists in [ingest_astra.py](ingest_astra.py) for legacy records.


# Krishna-wisdom
