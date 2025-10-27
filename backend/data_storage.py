import os
import re
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        if not os.path.exists(pdf_path):
            print("[WARN] File not found:", pdf_path)
            continue
        print("[INFO] Reading:", pdf_path)
        extracted_text = ""
        try:
            pdf_reader = PdfReader(pdf_path)
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    print(f"[WARN] Page {i+1} read error: {e}")
                    page_text = None
                if page_text and page_text.strip():
                    extracted_text += page_text + "\n"
                else:
                    print(f"[INFO] Using OCR for page {i+1}...")
                    extracted_text += extract_text_ocr(pdf_path, i)
        except Exception as e:
            print(f"[WARN] PyPDF2 failed for '{pdf_path}': {e}")
            print("[INFO] Running full OCR for file...")
            extracted_text = extract_text_ocr(pdf_path)
        text += extracted_text + "\n"
    return text.strip()


def extract_text_ocr(pdf_path, page_index=None):
    text = ""
    try:
        pages = convert_from_path(pdf_path)
        if page_index is not None:
            pages = [pages[page_index]]
        for img in pages:
            text += pytesseract.image_to_string(img, lang="eng") + "\n"
    except Exception as e:
        print(f"[ERROR] OCR failed for '{pdf_path}': {e}")
    return text


def clean_text(text):
    """
    Preserve paragraph breaks: collapse 2+ newlines to a marker, remove single newlines,
    collapse spaces, then restore double newlines.
    """
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "<<PARA>>", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"<<PARA>>", "\n\n", text)
    return text.strip()


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n\n",  # Split on paragraphs for better context
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    print(f"[INFO] Created {len(chunks)} text chunks")
    return chunks


def get_or_load_vectorstore(text_chunks, path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if os.path.exists(path):
        print(f"[INFO] Loading existing vectorstore from '{path}'...")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        if text_chunks:
            print("[INFO] Adding new chunks to existing vectorstore...")
            vectorstore.add_texts(text_chunks)
            vectorstore.save_local(path)
            print(f"[INFO] Updated vectorstore saved to '{path}'")
        return vectorstore

    print("[INFO] Creating new vectorstore...")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local(path)
    print(f"[INFO] Vectorstore saved to '{path}'")
    return vectorstore


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=False,
        return_source_documents=False
    )
    print("[INFO] Created conversation chain with RAG")
    return chain

# RAG helper functions

def format_history_from_db(chat_messages_db):
    """Convert DB rows to LangChain messages, skipping blanks."""
    msgs = []
    for msg in chat_messages_db:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if msg.get("sender") == "user":
            msgs.append(HumanMessage(content=content))
        elif msg.get("sender") == "bot":
            msgs.append(AIMessage(content=content))
    return msgs

def generate_general_response(question, history_messages):
    """LLM-only response (no retrieval), concise + fallback."""
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    messages = [
        SystemMessage(content="You are a helpful, concise assistant."),
        *history_messages,
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    answer = (getattr(resp, "content", "") or "").strip()
    if not answer:
        # Single retry without history if blank
        resp = llm.invoke([SystemMessage(content="You are a helpful, concise assistant."), HumanMessage(content=question)])
        answer = (getattr(resp, "content", "") or "").strip()
    return answer

def generate_rag_response(vectorstore, question, history_messages, k=5):
    """Retrieve -> prompt with context -> LLM, with safe fallback to non-RAG."""
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    try:
        # Use MMR for diverse coverage (selects chunks that are relevant to the query but dissimilar to each other)
        docs = vectorstore.max_marginal_relevance_search(
            question, k=k, fetch_k=max(20, k * 5), lambda_mult=0.5
        )
        print(f"[INFO] Retrieved {len(docs)} documents for RAG (MMR)")
    except Exception as e:
        print(f"[WARN] MMR search failed ({e}); falling back to similarity_search")
        docs = vectorstore.similarity_search(question, k=k)

    # Do not truncate context — we already cap k and chunk size
    context_text = "\n\n---\n\n".join(getattr(d, "page_content", "") for d in docs)
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question in a concise manner.\n\n"
        f"Context:\n{context_text if context_text else '[No relevant context found]'}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        *history_messages,
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    print(f"[CHAT] Retrieved response from RAG chain: {resp}")
    answer = (getattr(resp, "content", "") or "").strip()

    if not answer:
        print("[INFO] Empty RAG answer; retrying without context...")
        answer = generate_general_response(question, history_messages)

    return answer


def run_conversational_agent(pdf_files):
    print("[INFO] Checking for existing FAISS vectorstore...")
    if not os.path.exists(index_path):
        print("[INFO] No vectorstore found. Extracting from PDFs...")
        raw_text = get_pdf_text(pdf_files)
        if not raw_text.strip():
            print("[ERROR] No text extracted. Exiting.")
            return
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
    else:
        text_chunks = []

    vectorstore = get_or_load_vectorstore(text_chunks, path=index_path)
    # conversation_chain = create_conversation_chain(vectorstore)  # no longer used in local run

    print("\n[READY] Ask questions about the document. Type 'exit' to quit.\n")

    # Maintain chat history as LangChain messages for follow-ups
    history_msgs = []

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("[INFO] Exiting conversation.")
            break

        # RAG response using the latest pipeline
        answer = generate_rag_response(vectorstore, query, history_messages=history_msgs, k=5)
        print("Bot:", answer)

        # Update history for follow-up questions
        history_msgs.append(HumanMessage(content=query))
        history_msgs.append(AIMessage(content=answer))


if __name__ == "__main__":
    pdf_files = ["pdfs/Ads cookbook.pdf"]
    run_conversational_agent(pdf_files)