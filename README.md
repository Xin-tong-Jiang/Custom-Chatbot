# AI Chatbot with RAG Pipeline

A full-stack AI chatbot application with Retrieval-Augmented Generation (RAG) capabilities. Upload PDF documents and chat with an AI that can answer questions based on the document content.

## Features

- **PDF Upload**: Upload PDF documents through a web interface
- **Text Extraction**: Advanced PDF text extraction with OCR fallback
- **Vector Database**: FAISS-based vector storage for document embeddings
- **RAG Pipeline**: Retrieval-Augmented Generation for context-aware responses
- **Interactive Chat**: Real-time chat interface with the AI
- **Modern UI**: Clean, responsive React frontend

## Architecture

- **Frontend**: React.js with modern UI components
- **Backend**: FastAPI with Python
- **RAG Pipeline**: LangChain + OpenAI + FAISS
- **Vector Store**: FAISS for efficient similarity search

## Setup Instructions

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your OpenAI API key:
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. Frontend Setup

```bash
# Install Node.js dependencies
cd frontend
npm install
cd ..
```

### 3. Running the Application

**Terminal 1 - Start Backend:**
```bash
python start_backend.py
```
Backend will be available at: http://localhost:8000

**Terminal 2 - Start Frontend:**
```bash
./start_frontend.sh
```
Frontend will be available at: http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. Upload a PDF document using the "Upload PDF" button
3. Wait for the document to be processed (vectorized)
4. Start chatting! Ask questions about the document content
5. The AI will provide context-aware answers based on the uploaded document

## API Endpoints

- `POST /upload-pdf` - Upload and process a PDF file
- `POST /chat` - Send a message to the chatbot
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

## Technical Details

### RAG Pipeline
1. **PDF Processing**: Extract text using PyPDF2 with OCR fallback
2. **Text Chunking**: Split documents into manageable chunks
3. **Embedding**: Generate embeddings using HuggingFace sentence transformers
4. **Vector Storage**: Store embeddings in FAISS vector database
5. **Retrieval**: Find relevant chunks for user queries
6. **Generation**: Generate responses using OpenAI GPT with retrieved context

### Frontend Features
- File upload with drag-and-drop support
- Real-time chat interface
- Typing indicators
- Responsive design
- Error handling

## Environment Variables

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Dependencies

### Backend
- FastAPI: Web framework
- LangChain: RAG pipeline
- OpenAI: LLM integration
- FAISS: Vector database
- PyPDF2: PDF processing
- pdf2image: OCR support

### Frontend
- React: UI framework
- Axios: HTTP client
- Modern CSS: Styling

## Troubleshooting

1. **PDF Upload Issues**: Ensure the PDF is not password-protected
2. **API Errors**: Check that your OpenAI API key is valid
3. **CORS Issues**: Make sure the backend is running on port 8000
4. **Memory Issues**: Large PDFs may require more RAM for processing

## Development

The application supports hot reloading:
- Backend: Automatic restart on code changes
- Frontend: Hot reloading with React development server