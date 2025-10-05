# Chat with PDF using AWS Bedrock

**Chat with PDF using AWS Bedrock** is a Streamlit app that allows users to interactively query PDF documents. It uses **AWS Bedrock** to generate embeddings with the Titan model and retrieve context with **Llama 3**, storing document vectors in **FAISS** for fast similarity search. Users can upload PDFs, update the vector store, and get detailed, context-aware answers directly from their documents.

---

## Features

- **PDF Data Ingestion:** Automatically loads PDFs from a directory and splits them into manageable chunks.
- **Vector Store Creation:** Generates embeddings and stores them in FAISS for efficient retrieval.
- **Interactive Q&A:** Users can ask questions and get detailed answers based on PDF content.
- **Streamlit Interface:** Simple web interface to upload PDFs, update vectors, and query documents.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/OmRajput17/Chat-PDF-with-BedRock.git
    ```

2. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure you have AWS credentials configured for Bedrock access.

---

## Usage
1. Run the Streamlit App
    
    streamlit run app.py

2. Use the sidebar to update or create vector stores from your PDFs.

3. Enter a question in the input box and click Output to get answers.

---

### Project Structure
```graphql
.
├── app.py                # Main Streamlit application
├── data/                 # Directory to store PDF files
├── faiss_index/          # Local FAISS vector store (auto-generated)
├── requirements.txt      # Python dependencies
└── README.md
```
---

### Dependencies
    - Python 3.9+
    - Streamlit
    - Boto3
    - LangChain
    - FAISS (faiss-cpu or faiss-gpu)
    - Other Python libraries as listed in requirements.txt

---

## Notes
- Make sure to configure your AWS Bedrock credentials properly.
- The app currently uses Titan for embeddings and Llama 3 for text generation.
- FAISS is used for vector storage and retrieval.