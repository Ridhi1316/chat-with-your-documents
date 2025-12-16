# Chat-with-Your-Documents (React Frontend & FastAPI Backend)

An AI-powered document chatbot that lets you chat with your own PDF documents, featuring a modern React.js frontend and a robust FastAPI backend.

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![React](https://img.shields.io/badge/React-Frontend-blue)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-blue)](https://fastapi.tiangolo.com/)

## Overview

This project provides a local solution for interacting with PDF documents using natural language. It combines a React.js frontend for a user-friendly experience with a FastAPI backend for efficient document processing and question answering using Retrieval-Augmented Generation (RAG).

**App Showcase:**

![Chat-with-Your-Documents Screenshot](images\Ss3.png)


*See the application in action!*

## Key Features

*   **Modern React.js Frontend:** A clean and responsive user interface.
*   **FastAPI Backend:** Handles document processing and question answering efficiently.
*   **Retrieval-Based Question Answering (RAG-ready):** Document embeddings are stored in FAISS and the most relevant document chunks are retrieved and returned as answers. The system can be extended to full RAG with LLMs.
*   **Local Document Q&A:** Ask questions and get answers based on your PDF documents, all running locally.
*   **Persistent Vector Store:** Uses FAISS (via Langchain) for efficient storage and retrieval of document embeddings.
*   **LLM-agnostic Architecture:** The system currently uses retrieval-based question answering for accuracy and free local execution.

## How It Works

1. User uploads a PDF document via the React frontend.
2. The FastAPI backend extracts text from the PDF.
3. Text is split into chunks and converted into embeddings using Hugging Face Sentence Transformers.
4. Embeddings are stored in a FAISS vector store.
5. When a user asks a question, the backend retrieves the most relevant document chunk using semantic search.
6. The retrieved content is returned as the answer.

## Technology Stack

*   **Frontend:**
    *   [React.js](https://reactjs.org/)
    *   [styled-components](https://styled-components.com/)
    *   [axios](https://axios-http.com/)
    *   [react-dropzone](https://react-dropzone.js.org/)
*   **Backend:**
    *   [FastAPI](https://fastapi.tiangolo.com/)
    *   [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
    *   [FAISS](https://github.com/facebookresearch/faiss) (via Langchain)
    *   [Langchain](https://www.langchain.com/)
    *   [Python](https://www.python.org/) 3.10+

## Prerequisites

*   [Node.js](https://nodejs.org/) and npm (Node Package Manager)
*   Python 3.10 or higher

## Setup and Installation

**1. Clone the repository:**

```bash
git clone 
cd Chat-with-Your-Documents
```

## Backend Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install fastapi uvicorn langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf
uvicorn backend_chatdoc:app --reload

```
---

## Frontend Setup missing Vite mention

### Fix frontend section

## Frontend Setup

```bash
cd front-chatdoc
npm install
npm run dev
