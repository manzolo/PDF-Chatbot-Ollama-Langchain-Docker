# PDF Chatbot with Ollama and Langchain

This project implements a chatbot that allows you to ask questions about the content of a PDF file. It utilizes [Ollama](https://ollama.ai/) to run a language model locally and [Langchain](https://www.langchain.com/) to orchestrate the process of loading, splitting, embedding, and searching the PDF text. The user interface is provided by [Flask](https://flask.palletsprojects.com/).

## Features

* **PDF File Upload:** Allows uploading PDF files through a web interface.
* **PDF Processing:** The uploaded PDF is split into text chunks and converted into embeddings using Ollama.
* **Vector Storage:** Embeddings are stored in a Chroma vector database for efficient semantic search.
* **Question Answering:** Users can ask natural language questions about the PDF content.
* **Responses:** The chatbot uses the Ollama language model to generate answers based on the relevant content of the PDF.
* **Web Interface:** Provides a simple web interface for uploading files and chatting with the PDF.
* **Automatic Cleanup:** Upon completion, temporary uploaded files and the ChromaDB collection are deleted.


## Requirements

To run this application using Docker, you will need:

* **Docker:** Ensure you have Docker installed on your system. You can download it from [Docker Desktop](https://www.docker.com/products/docker-desktop/).
* **Docker Compose:** Docker Compose simplifies the management of multi-container Docker applications. You can install it by following the instructions on the [official Docker website](https://docs.docker.com/compose/install/).
* **Ollama (Local):** This setup assumes you have Ollama running locally on your machine, accessible by the Docker container. You will need to download and run Ollama separately, ensuring the desired language model is available. You can find installation instructions on the [Ollama website](https://ollama.ai/).


## Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/manzolo/PDF-Chatbot-Ollama-Langchain-Docker
    cd PDF-Chatbot-Ollama-Langchain-Docker
    ```

2.  **`.env` file:** Configure the following variables:

    ```env
    OLLAMA_HOST=http://172.17.0.1:11434  # The address of your Ollama server
    OLLAMA_MODEL=llama2                 # The name of the language model you want to use
    ```

    Ensure that `OLLAMA_HOST` and `OLLAMA_MODEL` match your Ollama setup.

3.  **Run the application:**

    ```bash
    docker compose up --build
    ```

    This command will build the Docker image (if necessary) and start the application container.

The application will be available at `http://127.0.0.1:5000/` in your browser. Ensure that Ollama is running locally and is accessible at `http://localhost:11434` from your host machine, which the Docker container will reach via `host.docker.internal`. The `/app/uploads` path inside the container is mounted to the `./uploads` volume on your host.

## Screenshot

![immagine](https://github.com/user-attachments/assets/f066e7db-ef13-4f88-9cd2-c68a4a67fe8e)

![immagine](https://github.com/user-attachments/assets/96237422-ac16-4878-8b61-52761d3856c9)

