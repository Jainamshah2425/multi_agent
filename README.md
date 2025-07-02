# Multi-Agent System

A sophisticated multi-agent system built with Python that combines document processing, vector search, and intelligent question-answering capabilities with web search integration.

## 🚀 Features

- **Document Processing**: Advanced document ingestion and processing pipeline
- **Vector Search**: FAISS-based vector storage for efficient similarity search
- **Question-Answering Engine**: Intelligent QA system powered by LLMs
- **Web Search Integration**: Enhanced search capabilities with web integration
- **Multi-Agent Architecture**: Coordinated agents for complex task handling
- **Streamlit Interface**: User-friendly web interface
- **Docker Support**: Containerized deployment for easy setup

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.x** - Main programming language
- **Streamlit** - Web application framework
- **FAISS** - Vector similarity search and clustering
- **LangChain** - LLM application development framework

### AI/ML Components
- **Large Language Models (LLMs)** - For natural language processing
- **Vector Embeddings** - For document similarity and retrieval
- **Document Processing** - Text extraction and preprocessing

### Infrastructure
- **Docker** - Containerization and deployment
- **Docker Compose** - Multi-container orchestration

### Storage
- **FAISS Vector Store** - Efficient vector storage and retrieval
- **Pickle Files** - Serialized data storage
- **JSON** - Configuration and cache storage

## 📁 Project Structure

```
multi-agent/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration management
├── document_processor.py           # Document processing logic
├── ingest.py                       # Data ingestion pipeline
├── llm_manager.py                  # LLM integration and management
├── qa_engine.py                    # Question-answering engine
├── web_search.py                   # Web search functionality
├── utils.py                        # Utility functions
├── test_retrieval.py               # Testing and validation
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose setup
├── cache/                          # Cache directory
│   └── file_hashes.json           # File hash cache
└── vectorstore_*/                  # FAISS vector stores
    ├── index.faiss                # FAISS index files
    └── index.pkl                  # Pickle metadata
```

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

#### Option 1: Local Installation
```bash
# Clone the repository
git clone https://github.com/Jainamshah2425/multi_agent.git
cd multi_agent

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Option 2: Docker Deployment
```bash
# Clone the repository
git clone https://github.com/Jainamshah2425/multi_agent.git
cd multi_agent

# Build and run with Docker Compose
docker-compose up --build
```

### Configuration

1. **Environment Setup**: Configure your environment variables in `config.py`
2. **LLM Configuration**: Set up your LLM API keys and endpoints
3. **Document Ingestion**: Use `ingest.py` to process and index your documents

## 🔧 Usage

### Document Processing
```python
# Process and ingest documents
python ingest.py
```

### Running the Application
```bash
# Start the Streamlit interface
streamlit run app.py
```

### Testing
```bash
# Run retrieval tests
python test_retrieval.py
```

## 🏗️ Architecture

The system follows a multi-agent architecture with the following components:

1. **Document Processor Agent**: Handles document ingestion and preprocessing
2. **Vector Store Agent**: Manages vector embeddings and similarity search
3. **QA Engine Agent**: Processes questions and generates answers
4. **Web Search Agent**: Integrates external web search capabilities
5. **Coordination Layer**: Orchestrates agent interactions

## 🔍 Key Components

### Document Processing
- Text extraction from various document formats
- Content preprocessing and chunking
- Vector embedding generation

### Vector Search
- FAISS-based similarity search
- Multiple vector store support
- Efficient indexing and retrieval

### Question Answering
- Context-aware answer generation
- Multi-step reasoning capabilities
- Source attribution and confidence scoring

### Web Search Integration
- Real-time web search capabilities
- Search result processing and integration
- Enhanced context for better answers

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries and tools
- Special thanks to the LangChain and FAISS communities
- Streamlit team for the excellent web framework
