# -*- coding: utf-8 -*-
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""

    for pdf_path in pdf_docs:
        if not os.path.exists(pdf_path):
            print("[WARN] File not found:", pdf_path)
            continue

        print("[INFO] Reading:", pdf_path)
        extracted_text = ""

        try:
            # Try text-based extraction first
            pdf_reader = PdfReader(pdf_path)
            for i, page in enumerate(pdf_reader.pages):
                page_text = None
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    print("[WARN] Page {} read error: {}".format(i + 1, e))

                # If this page has text, append
                if page_text and page_text.strip():
                    extracted_text += page_text + "\n"
                else:
                    # Fallback: use OCR for this page
                    print("[INFO] Using OCR for page {}...".format(i + 1))
                    extracted_text += extract_text_ocr(pdf_path, i)

        except Exception as e:
            # If PyPDF2 completely fails, fallback to full OCR
            print("[WARN] PyPDF2 failed for '{}': {}".format(pdf_path, e))
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
            page_text = pytesseract.image_to_string(img, lang="eng")
            text += page_text + "\n"
    except Exception as e:
        print("[ERROR] OCR failed for '{}': {}".format(pdf_path, e))
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


def get_vectorstore(text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not text_chunks:
        raise ValueError("[ERROR] No text chunks found. Check PDF extraction or OCR output.")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore



if __name__ == "__main__":
    pdf_files = ["CALuxuryGuide_FY24_25.pdf"]

    print("[INFO] Reading PDF(s)...")
    raw_text = get_pdf_text(pdf_files)

    if not raw_text.strip():
        print("[ERROR] No text extracted from PDF. Please check if it's encrypted or unreadable.")
        exit()

    print("[INFO] Cleaning text...")
    cleaned_text = clean_text(raw_text)
    print("[INFO] Text length after cleaning:", len(cleaned_text))

    print("[INFO] Splitting into chunks...")
    text_chunks = get_text_chunks(cleaned_text)
    print("[INFO] Total chunks:", len(text_chunks))

    print("[INFO] Building FAISS vectorstore...")
    vectorstore = get_vectorstore(text_chunks)
