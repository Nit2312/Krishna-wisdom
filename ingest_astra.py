import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from astrapy import DataAPIClient
import fitz  # PyMuPDF
from pathlib import Path

load_dotenv()

def resolve_data_source():
    data_url = os.getenv("DATA_FILE_URL")
    if data_url:
        return data_url

    data_path = os.getenv("DATA_FILE_PATH")
    if data_path:
        return data_path

    default_paths = ["10yearsdata.xls"]
    for path in default_paths:
        if os.path.exists(path):
            return path

    return None

def get_astra_config():
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
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("HF_TOKEN is required for endpoint embeddings.")
        self._client = InferenceClient(model=model_name, token=api_key)

    def embed_documents(self, texts):
        result = self._client.feature_extraction(texts)
        if isinstance(result, list) and result and isinstance(result[0], float):
            return [result]
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def build_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    backend = os.getenv("EMBEDDINGS_BACKEND", "local").strip().lower()

    if backend == "endpoint":
        embeddings = RouterHuggingFaceEmbeddings(
            api_key=os.getenv("HF_TOKEN"),
            model_name=model_name,
        )
        try:
            embeddings.embed_query("health check")
            return embeddings
        except Exception as e:
            print(f"Warning: endpoint embeddings failed ({e}); falling back to local.")

    return HuggingFaceEmbeddings(model_name=model_name)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def ingest_pdfs_from_data_folder(data_folder="data"):
    """Ingest all PDF files from the data folder"""
    pdf_documents = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"Data folder {data_folder} not found")
        return pdf_documents
    
    # Find all PDF files
    pdf_files = list(data_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        
        if text and text.strip():  # Only add if text is not empty
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(pdf_file),
                    "filename": pdf_file.name,
                    "type": "pdf_document"
                }
            )
            pdf_documents.append(doc)
        else:
            print(f"Warning: No text extracted from {pdf_file.name}")
    
    return pdf_documents

def clean_excel_data(df):
    """Comprehensive data cleaning for Excel data - removes ANY row with empty data"""
    print("Original data shape:", df.shape)
    print("Original null counts:")
    print(df.isnull().sum())
    
    # Select relevant columns
    df_clean = df[["CaseID", "Job_Name", "Case_Problem", "Case_Resolution_Notes"]].copy()
    
    # Remove ANY row that has empty/NaN data in ANY column
    df_clean = df_clean.dropna()
    
    # Also remove rows with empty strings after stripping whitespace
    for col in df_clean.columns:
        df_clean = df_clean[df_clean[col].astype(str).str.strip() != '']
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with ANY empty data")
    print("\nNull counts after cleaning:")
    print(df_clean.isnull().sum())
    
    return df_clean

def main():
    # Ingest PDFs from data folder (primary source)
    pdf_docs = ingest_pdfs_from_data_folder()
    print(f"Successfully ingested {len(pdf_docs)} PDF documents")
    
    # Load and clean Excel data (optional)
    data_source = resolve_data_source()
    if data_source:
        df = pd.read_excel(data_source)
        df_clean = clean_excel_data(df)
    else:
        print("No Excel data source found. Proceeding with PDF documents only.")
        df_clean = pd.DataFrame()  # Empty DataFrame for Excel documents

    # Process documents (larger chunks keep procedure steps together; overlap preserves continuity)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    excel_documents = []
    if not df_clean.empty:
        for _, row in df_clean.iterrows():
            combined_text = f"Problem: {row['Case_Problem']}\n\nResolution: {row['Case_Resolution_Notes']}"
            doc = Document(
                page_content=combined_text,
                metadata={
                    "CaseID": row["CaseID"],
                    "Job_Name": row["Job_Name"],
                    "source": "excel_data",
                    "type": "case_record"
                },
            )
            excel_documents.append(doc)

    # Combine all documents
    all_documents = excel_documents + pdf_docs
    print(f"Total documents to process: {len(all_documents)}")
    print(f"- Excel case records: {len(excel_documents)}")
    print(f"- PDF documents: {len(pdf_docs)}")
    
    if not all_documents:
        raise ValueError("No documents found. Please ensure PDFs are in the 'data' folder and/or Excel data is available.")

    # Split documents into chunks
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks from {len(all_documents)} documents")

    embeddings = build_embeddings()

    astra_config = get_astra_config()

    # Always reset collection to ensure fresh data
    print("Resetting Astra DB collection...")
    client = DataAPIClient(astra_config["token"])
    
    # Use keyspace instead of deprecated namespace
    try:
        db = client.get_database(astra_config["api_endpoint"], keyspace=astra_config["namespace"])
        db.drop_collection(astra_config["collection_name"])
        print(f"Dropped collection: {astra_config['collection_name']}")
    except Exception as e:
        print(f"Warning: failed to drop collection '{astra_config['collection_name']}': {e}")
        # Try with namespace as fallback
        try:
            db = client.get_database(astra_config["api_endpoint"], namespace=astra_config["namespace"])
            db.drop_collection(astra_config["collection_name"])
            print(f"Dropped collection with namespace fallback: {astra_config['collection_name']}")
        except Exception as e2:
            print(f"Warning: failed to drop collection with fallback: {e2}")
    
    # Create new collection with AstraDBVectorStore (same as app.py for schema compatibility)
    print("Creating new Astra DB collection with AstraDBVectorStore...")
    AstraDBVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        api_endpoint=astra_config["api_endpoint"],
        token=astra_config["token"],
        namespace=astra_config["namespace"],
        collection_name=astra_config["collection_name"],
    )

    print(f"Successfully ingested {len(excel_documents)} Excel cases and {len(pdf_docs)} PDF documents into Astra DB.")
    print(f"Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()
