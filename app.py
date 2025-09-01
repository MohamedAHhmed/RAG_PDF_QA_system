import os
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from transformers import AutoTokenizer

# ========== Load API KEYS ==========
load_dotenv()
huggingfacehub_api_token = os.getenv("HF_TOKEN")
Qdrant_api_key = os.getenv("QDRANT_API_KEY")

# ========== LLM ==========
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
        provider="together",
        huggingfacehub_api_token=huggingfacehub_api_token,
        task="conversational"
    )
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


retriever_cache = {}

# ========== Prepare Data ==========
def prepare_data(filepath):
    loader = DoclingLoader(file_path=filepath, export_type=ExportType.MARKDOWN).load()
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=300, chunk_overlap=20
    )
    normal_chunks = text_splitter.create_documents(
        [loader[0].model_dump()['page_content']], 
        metadatas=[loader[0].model_dump()['metadata']]
    )
    return normal_chunks

# ========== Hybrid Search ==========
def Hybrid_search(normal_chunks):
    embedding_llm = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    qdrant_store = Qdrant.from_documents(
        documents=normal_chunks,
        embedding=embedding_llm,
        url="https://3464a78e-425b-4e6b-bc10-5b0333dc9ad1.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=Qdrant_api_key,
        collection_name="my_collection",
        force_recreate=True
    )

    dense_retriever = qdrant_store.as_retriever(
        search_kwargs={"k": 8, "score_threshold": 0.25}
    )
    bm25_retriever = BM25Retriever.from_documents(normal_chunks)
    bm25_retriever.k = 8

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]
    )
    return hybrid_retriever

# ========== Call Model ==========
def call_model(question, retriever):
    qna_template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    from langchain.prompts import PromptTemplate
    qna_prompt = PromptTemplate(
        template=qna_template,
        input_variables=['context', 'question']
    )

    stuff_chain = create_stuff_documents_chain(llm, prompt=qna_prompt)
    retrieved_docs = retriever.get_relevant_documents(question)
    
    answer = stuff_chain.invoke(
        {
            "context": retrieved_docs,
            "question": question
        }
    )
    return answer

# ========== Gradio App ==========
def upload_pdf(file_path, progress=gr.Progress()):
    progress(0, desc="Preparing data...")
    chunks = prepare_data(file_path)
    progress(0.5, desc="Building retrievers...")
    retriever_cache["retriever"] = Hybrid_search(chunks)
    progress(1.0, desc="Done ✅")
    return "✅ PDF uploaded successfully! Now ask your questions."

def qa_interface(question):
    if "retriever" not in retriever_cache:
        return "❌ Please upload a PDF first."
    return call_model(question, retriever_cache["retriever"])

with gr.Blocks() as demo:
    gr.Markdown("## 📚 PDF Q&A with Hybrid Search + LLM")
    
    with gr.Row():
        file_input = gr.File(label="Upload PDF", type="filepath")
        upload_output = gr.Textbox(label="Upload Status")

    upload_btn = gr.Button("Upload PDF")
    upload_btn.click(
        fn=upload_pdf,
        inputs=[file_input],
        outputs=[upload_output]
    )
    
    question_input = gr.Textbox(label="Ask a question")
    output = gr.Markdown()
    submit_btn = gr.Button("Get Answer")

    submit_btn.click(
        fn=qa_interface,
        inputs=[question_input],
        outputs=output
    )

demo.launch(share=True)
