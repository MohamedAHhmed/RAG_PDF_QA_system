# 📚 PDF Q&A with Hybrid Search + LLM

## 🚀 Overview
This project is a **Question Answering (QA) system** that allows users to:
1. Upload a **PDF document**.  
2. Automatically process and chunk the text.  
3. Store embeddings in **Qdrant Vector Database** and build a **hybrid retriever** (BM25 + Qdrant).  
4. Ask **natural language questions**, and the model will retrieve the relevant context from the PDF and generate an answer using a **Large Language Model (LLM)**.  

It combines **semantic search (dense)** + **keyword search (BM25)** for better retrieval accuracy.

---

## 🛠️ Tech Stack
- **LangChain** → Orchestration of retrievers and chains.  
- **HuggingFace + Together API** → LLM endpoint (`Qwen3-235B-A22B-Instruct-2507`).  
- **Qdrant** → Vector database for storing embeddings.  
- **BM25** → Keyword-based retriever.  
- **Docling** → Loader to extract text from PDF into Markdown.  
- **Transformers** → Tokenizer for chunking text.  
- **Gradio** → Web interface.  
- **dotenv** → Secure API key management.  

---

## ⚙️ Workflow
1. **Upload PDF**  
   - The file is loaded with `DoclingLoader`.  
   - Text is split into **chunks** using HuggingFace tokenizer.  

2. **Build Hybrid Search**  
   - Embeddings are created using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.  
   - Chunks are stored in **Qdrant**.  
   - **Dense retriever** (embeddings) + **BM25 retriever** (keywords) are combined with weights `0.6` (dense) and `0.4` (BM25).  

3. **Ask Questions**  
   - User writes a question.  
   - Relevant chunks are retrieved.  
   - A **prompt** is built with context + question.  
   - The **LLM** generates the answer (max 3 sentences).  

---

## 📋 Features
- Upload any **PDF document**.  
- Hybrid search ensures **more accurate retrieval** than only embeddings or BM25.  
- Context-aware **Q&A** answers.  
- **Caching retriever** so you only upload once (no need to re-process for every question).  
- Simple **Gradio UI** with upload + question box.  

---

## 🔑 Requirements
- Python 3.10+  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
