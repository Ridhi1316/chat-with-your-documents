from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import shutil
from typing import List
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# -------------------- FastAPI --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Constants --------------------
VECTOR_STORE_PATH = "faiss_vector_store"
UPLOAD_FOLDER = "uploaded_documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Embeddings --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- Vector Store --------------------
vector_store = None


def load_vector_store():
    """Load FAISS index if exists, otherwise initialize empty."""
    global vector_store
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded from disk.")
    except Exception:
        vector_store = None
        print("No existing vector store found. Waiting for upload.")


load_vector_store()

# -------------------- Models --------------------
class QueryRequest(BaseModel):
    question: str


# -------------------- Core Query Logic --------------------
def query(question: str):
    global vector_store

    if vector_store is None:
        return {"answer": "Please upload a document first."}

    # Retrieve top chunks
    relevant_docs = vector_store.similarity_search(question, k=5)

    if not relevant_docs:
        return {"answer": "No relevant content found in the document."}

    # Pick the best matching chunk (highest similarity)
    best_doc = relevant_docs[0]
    answer_text = best_doc.page_content.strip()

    # Trim overly long responses
    if len(answer_text) > 1200:
        answer_text = answer_text[:1200]

    return {
        "answer": answer_text,
        "sources": [best_doc.metadata.get("source", "Unknown")]
    }


# -------------------- Document Processing --------------------
def process_document(file_path: str) -> List[str]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_documents(documents)

        cleaned_chunks = []
        for chunk in chunks:
            text = chunk.page_content.replace("\n", " ")
            text = " ".join(text.split())
            cleaned_chunks.append(text)

        return cleaned_chunks

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {e}"
        )


# -------------------- Upload API --------------------
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        texts = process_document(file_path)

        # Reset old FAISS index (single-document mode)
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)

        vector_store = FAISS.from_texts(texts, embedding=embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)

        print(f"FAISS index size: {vector_store.index.ntotal}")

        return {
            "filename": file.filename,
            "message": "PDF uploaded and indexed successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()


# -------------------- Query API --------------------
@app.post("/query/")
async def ask_question(request: QueryRequest):
    return query(request.question)


# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
