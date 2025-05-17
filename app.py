import os
import time
import atexit
import shutil
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings

# Load environment variables
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Store for the current vector index
app.current_vector_store: Optional[Chroma] = None

# Cleanup at exit
def _cleanup_files_and_store():
    if app.current_vector_store:
        try:
            app.current_vector_store._client.delete_collection(
                app.current_vector_store._collection.name
            )
            logger.info('Vector store collection deleted successfully on exit.')
        except Exception as err:
            logger.warning('Failed to delete vector store on exit: %s', err)

    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
        logger.info('Upload folder cleaned successfully on exit.')
    except Exception as err:
        logger.warning('Failed to clean upload folder on exit: %s', err)

atexit.register(_cleanup_files_and_store)


def validate_environment():
    """Ensure required environment variables are set."""
    host = os.getenv('OLLAMA_HOST')
    model = os.getenv('OLLAMA_MODEL')
    if not host or not model:
        logger.error('OLLAMA_HOST and OLLAMA_MODEL must be set.')
        raise EnvironmentError('OLLAMA_HOST and OLLAMA_MODEL must be configured in .env')
    return host, model


def clean_previous_store():
    """Delete the existing vector store, if any."""
    store = app.current_vector_store
    if store:
        try:
            store.delete_collection()
            logger.info('Previous vector store collection deleted.')
        except Exception as err:
            logger.warning('Error deleting previous vector store: %s', err)
        finally:
            app.current_vector_store = None


def split_pdf_to_chunks(filepath: Path, chunk_size: int, chunk_overlap: int):
    """Load a PDF and split into text chunks."""
    loader = PyPDFLoader(str(filepath))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(pages)


def create_vector_store(
    documents, host: str, model: str, prefix: str = 'pdf', telemetry: bool = False
) -> Chroma:
    """Create a Chroma vector store from documents."""
    collection_name = f"{prefix}_{int(time.time())}"
    store = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(base_url=host, model=model),
        collection_name=collection_name,
        client_settings=Settings(anonymized_telemetry=telemetry)
    )
    logger.info('Created new vector store: %s', collection_name)
    return store


def build_retrieval_qa(store: Chroma, host: str, model: str, temperature: float, k: int):
    """Instantiate a RetrievalQA chain from a vector store."""
    retriever = store.as_retriever(search_kwargs={'k': k})
    llm = OllamaLLM(base_url=host, model=model, temperature=temperature)
    return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)


def process_pdf(
    filepath: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 4,
    temperature: float = 0.3
) -> RetrievalQA:
    """
    Process a PDF file to produce a RetrievalQA agent.

    Steps:
    1. Clean any existing vector store.
    2. Split the PDF into text chunks.
    3. Build a new vector store.
    4. Return a RetrievalQA chain.
    """
    if not filepath.is_file() or filepath.suffix.lower() != '.pdf':
        logger.error('Invalid PDF file: %s', filepath)
        raise ValueError(f"Invalid PDF file: {filepath}")

    clean_previous_store()
    host, model = validate_environment()

    documents = split_pdf_to_chunks(filepath, chunk_size, chunk_overlap)
    store = create_vector_store(documents, host, model)
    app.current_vector_store = store

    return build_retrieval_qa(store, host, model, temperature, top_k)


@app.route('/', methods=['GET'])
def upload_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def handle_upload():
    # Reset existing store
    clean_previous_store()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    save_path = app.config['UPLOAD_FOLDER'] / filename
    file.save(str(save_path))
    logger.info('File uploaded: %s', filename)
    return jsonify({'filename': filename}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json() or {}
    filename = data.get('filename')
    query = data.get('query')

    if not filename or not query:
        return jsonify({'error': 'Missing parameters'}), 400

    pdf_path = app.config['UPLOAD_FOLDER'] / filename
    if not pdf_path.exists():
        return jsonify({'error': 'File not found'}), 404

    qa_agent = process_pdf(pdf_path)
    response = qa_agent.invoke({'query': query})
    answer = response.get('result', '')

    return jsonify({'answer': answer}), 200


@app.route('/chat/<filename>', methods=['GET', 'POST'])
def chat_page(filename):
    pdf_path = app.config['UPLOAD_FOLDER'] / filename
    if not pdf_path.exists():
        return jsonify({'error': 'File not found'}), 404

    if request.method == 'POST':
        query = request.form.get('query', '')
        qa_agent = process_pdf(pdf_path)
        response = qa_agent.invoke({'query': query})
        answer = response.get('result', '')
        return render_template('chat.html', filename=filename, query=query, answer=answer)

    return render_template('chat.html', filename=filename)


@app.route('/uploads/<path:filename>')
def serve_pdf(filename):
    return send_from_directory(str(app.config['UPLOAD_FOLDER']), filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)