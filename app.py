import os
import time
import atexit
import shutil
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings

# Configurazione
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variabile globale per lo store vettoriale
current_vector_store = None

# Funzione di pulizia
@atexit.register
def cleanup():
    global current_vector_store
    try:
        if current_vector_store:
            current_vector_store._client.delete_collection(current_vector_store._collection.name)
    except Exception as e:
        print(f"Cleanup error: {e}")
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except Exception as e:
        print(f"Cleanup error: {e}")

def process_pdf(filepath):
    global current_vector_store
    
    # Pulisci la cache precedente
    if current_vector_store:
        try:
            current_vector_store.delete_collection()
        except Exception as e:
            print(f"Error cleaning previous collection: {e}")
    
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(pages)
        
        current_vector_store = Chroma.from_documents(
            documents=texts,
            embedding=OllamaEmbeddings(
                base_url=os.getenv("OLLAMA_HOST"),
                model=os.getenv("OLLAMA_MODEL")
            ),
            collection_name=f"pdf_{os.path.basename(filepath)}_{int(time.time())}",
            client_settings=Settings(anonymized_telemetry=False)
        )
        
        return RetrievalQA.from_chain_type(
            llm=Ollama(
                base_url=os.getenv("OLLAMA_HOST"),
                model=os.getenv("OLLAMA_MODEL"),
                temperature=0.3
            ),
            chain_type="stuff",
            retriever=current_vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
        )
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

@app.route('/', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    global current_vector_store
    
    # Resetta lo store esistente
    if current_vector_store:
        try:
            current_vector_store.delete_collection()
        except:
            pass
        current_vector_store = None

    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file fornito'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename}), 200
    
    return jsonify({'error': 'Formato file non supportato'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    filename = data.get('filename')
    query = data.get('query')
    
    if not filename or not query:
        return jsonify({'error': 'Parametri mancanti'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File non trovato'}), 404
    
    qa_agent = process_pdf(filepath)
    answer = qa_agent.invoke({"query": query})["result"]
    
    return jsonify({'answer': answer}), 200


@app.route('/chat/<filename>', methods=['GET', 'POST'])
def chat_page(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        qa_agent = process_pdf(filepath)
        answer = qa_agent.invoke({"query": query})["result"]
        return render_template('chat.html', 
                            filename=filename,
                            query=query,
                            answer=answer)
    
    return render_template('chat.html', filename=filename)

@app.route('/uploads/<filename>')
def serve_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)