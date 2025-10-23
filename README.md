# Custom Q&A Chatbot

This project builds a Question-and-Answer (Q&A) chatbot powered by Large Language Models (LLMs) and embedding models. It aims to understand uploaded PDF documents and provide accurate, context-aware responses.

The team prepared and processed text data by extracting content from PDFs, creating embeddings, and constructing a vector datastore as the chatbot’s knowledge base. A user-friendly web interface was developed using HTML, CSS, and JavaScript to support real-time interaction and ensure a smooth user experience.

## Features
- PDF Text Extraction
- Text Chunk Generation
- Vector Datastore Creation
- Conversation Chain Construction
- PDF Upload and Analysis
- Interactive Chat Window with Input Field

## Usage

Create and activate a virtual environment:  
`python -m venv venv`  
`source venv/bin/activate`  

Install dependencies:  
`pip install -r requirements.txt`  

Put database login info in the same directory:  
`.env`  

Run data storage script:  
`python data_storage.py`



