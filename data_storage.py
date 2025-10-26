import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class InMemoryChatHistory(BaseChatMessageHistory, BaseModel):
    """In-memory chat history."""
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []


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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def get_text_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


def get_or_load_vectorstore(text_chunks, path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if os.path.exists(path):
        print(f"[INFO] Loading existing vectorstore from '{path}'...")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    print("[INFO] Creating new vectorstore...")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local(path)
    print(f"[INFO] Vectorstore saved to '{path}'")
    return vectorstore


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question. If you don't know, say so.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    return {"retriever": retriever, "llm": llm, "prompt": prompt}


def run_conversational_agent(pdf_files):
    print("[INFO] Checking for existing FAISS vectorstore...")
    if not os.path.exists("faiss_index"):
        print("[INFO] No vectorstore found. Extracting from PDFs...")
        raw_text = get_pdf_text(pdf_files)
        if not raw_text.strip():
            print("[ERROR] No text extracted. Exiting.")
            return
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
    else:
        text_chunks = []

    vectorstore = get_or_load_vectorstore(text_chunks)
    chain_components = create_conversation_chain(vectorstore)
    retriever = chain_components["retriever"]
    llm = chain_components["llm"]
    prompt = chain_components["prompt"]
    
    chat_history = InMemoryChatHistory()

    print("\n[READY] Ask questions about the document. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            print("[INFO] Exiting conversation.")
            break
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt with context and history
        messages = prompt.format_messages(
            context=context,
            chat_history=chat_history.messages,
            question=query
        )
        
        # Get response
        response = llm.invoke(messages)
        
        # Update history
        chat_history.add_message(HumanMessage(content=query))
        chat_history.add_message(AIMessage(content=response.content))
        
        print("Bot:", response.content)


if __name__ == "__main__":
    pdf_files = ["CALuxuryGuide_FY24_25.pdf"]
    run_conversational_agent(pdf_files)