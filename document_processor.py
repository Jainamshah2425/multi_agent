import fitz  # PyMuPDF
import os
import pickle
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from utils import logger, get_file_hash

def replace_newlines(text: str) -> str:
    return text.replace('\n', ' ')

class DocumentProcessor:
    def __init__(self):
        # Initialize embeddings with better error handling
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embeddings model: {str(e)}")
            raise
        
        # Optimized text splitter for better retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Reduced for better precision
            chunk_overlap=40,  # Reduced overlap to minimize noise
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # SESSION-BASED ISOLATION - CRITICAL FIX
        self._init_session_isolation()
        
    def _init_session_isolation(self):
        """Initialize session-based document isolation."""
        # Create unique session ID for document isolation
        if 'doc_session_id' not in st.session_state:
            st.session_state.doc_session_id = str(uuid.uuid4())
            logger.info(f"Created new document session: {st.session_state.doc_session_id}")
        
        self.session_id = st.session_state.doc_session_id
        
        # Use session-specific paths to prevent contamination
        self.vectorstore_path = f"vectorstore_{self.session_id}"
        self.documents_info_path = f"documents_info_{self.session_id}.pkl"
        
        logger.info(f"Document processor initialized for session: {self.session_id}")

    def reset_session(self):
        """Reset to a new session - clears all previous documents."""
        try:
            # Clear current session data
            self.clear_vectorstore()
            
            # Create new session
            st.session_state.doc_session_id = str(uuid.uuid4())
            self.session_id = st.session_state.doc_session_id
            self.vectorstore_path = f"vectorstore_{self.session_id}"
            self.documents_info_path = f"documents_info_{self.session_id}.pkl"
            
            logger.info(f"Reset to new document session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting session: {str(e)}")
            return False

    def process_uploaded_files(self, uploaded_files, force_new_session: bool = False) -> Dict[str, Any]:
        """Process uploaded files with session isolation."""
        if not uploaded_files:
            return {"success": False, "message": "No files uploaded"}
        
        # Option to force new session (isolate from previous uploads)
        if force_new_session:
            self.reset_session()
        
        try:
            new_documents_to_process = []
            new_processed_files_info = []
            
            # For session isolation, we don't load existing vectorstore from other sessions
            # Only load if it's from the current session
            existing_vectorstore = self.load_vectorstore()
            existing_documents_info = self.get_documents_info()
            
            # Create a map of existing file names to their hashes
            existing_file_hashes = {doc['name']: doc.get('hash') for doc in existing_documents_info}

            # Create temporary directory for uploaded files
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            
            upload_timestamp = datetime.now().isoformat()
            
            for uploaded_file in uploaded_files:
                logger.info(f"Processing file: {uploaded_file.name} in session {self.session_id}")
                
                # Save uploaded file temporarily to calculate hash
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                current_file_hash = get_file_hash(temp_file_path)

                # Check if file already processed in current session
                if uploaded_file.name in existing_file_hashes and existing_file_hashes[uploaded_file.name] == current_file_hash:
                    logger.info(f"File {uploaded_file.name} already processed in current session. Skipping.")
                    new_processed_files_info.append({
                        'name': uploaded_file.name,
                        'size': len(uploaded_file.getbuffer()),
                        'chunks': next((doc['chunks'] for doc in existing_documents_info if doc['name'] == uploaded_file.name), 0),
                        'hash': current_file_hash,
                        'session_id': self.session_id,
                        'upload_timestamp': upload_timestamp
                    })
                    continue

                # Process the file with session metadata
                documents = self._process_single_file(temp_file_path, uploaded_file.name, upload_timestamp)
                if documents:
                    new_documents_to_process.extend(documents)
                    new_processed_files_info.append({
                        'name': uploaded_file.name,
                        'size': len(uploaded_file.getbuffer()),
                        'chunks': len(documents),
                        'hash': current_file_hash,
                        'session_id': self.session_id,
                        'upload_timestamp': upload_timestamp
                    })
                    logger.info(f"Successfully processed {uploaded_file.name} with {len(documents)} chunks")
                else:
                    logger.warning(f"No content extracted from {uploaded_file.name}")
                
                # Clean up temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
            # Clean up temp directory
            try:
                os.rmdir(temp_dir)
            except:
                pass
            
            if not new_documents_to_process and not existing_vectorstore:
                return {"success": False, "message": "No new content to process and no existing vectorstore."}
            
            if new_documents_to_process:
                if existing_vectorstore:
                    logger.info(f"Adding {len(new_documents_to_process)} new documents to existing session vectorstore.")
                    existing_vectorstore.add_documents(new_documents_to_process)
                    vectorstore_to_save = existing_vectorstore
                else:
                    logger.info(f"Creating new vectorstore with {len(new_documents_to_process)} documents.")
                    vectorstore_to_save = FAISS.from_documents(new_documents_to_process, self.embeddings)
                
                # Update document info for current session
                final_processed_files = [doc for doc in existing_documents_info if doc['name'] not in {f['name'] for f in new_processed_files_info}] + new_processed_files_info
                
                # Save the updated vectorstore and document info
                self._save_vectorstore(vectorstore_to_save)
                self._save_documents_info(final_processed_files)
            else:
                logger.info("No new documents to add. Vectorstore remains unchanged.")
                final_processed_files = existing_documents_info

            # Verify the vectorstore
            test_vectorstore = self.load_vectorstore()
            if test_vectorstore is None:
                return {"success": False, "message": "Failed to save or update vectorstore"}
            
            total_chunks_in_vectorstore = len(test_vectorstore.index_to_docstore_id) if test_vectorstore else 0
            
            logger.info(f"Session {self.session_id}: Processed {len(new_documents_to_process)} new chunks. Total files: {len(final_processed_files)}, Total chunks: {total_chunks_in_vectorstore}")
            
            return {
                "success": True,
                "message": f"Successfully processed {len(new_processed_files_info)} files in session {self.session_id[:8]}...",
                "files": final_processed_files,
                "total_chunks": total_chunks_in_vectorstore,
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return {"success": False, "message": f"Error processing files: {str(e)}"}

    def _process_single_file(self, file_path: str, original_name: str, upload_timestamp: str) -> List[Document]:
        """Process a single file with enhanced metadata."""
        try:
            file_extension = original_name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                try:
                    loader = PyPDFLoader(file_path, extract_images=False)
                    raw_documents = loader.load()
                except Exception as e:
                    logger.warning(f"PyPDFLoader failed for {original_name}: {e}. Falling back to PyMuPDF.")
                    try:
                        doc = fitz.open(file_path)
                        raw_documents = [Document(page_content=page.get_text(), metadata={"source": original_name, "page": page.number}) for page in doc]
                    except Exception as e:
                        logger.error(f"Failed to load PDF {original_name} with both PyPDFLoader and PyMuPDF: {e}")
                        return []
            elif file_extension == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"Unsupported file type: {original_name}")
                return []
            
            
            
            if not raw_documents:
                logger.warning(f"No content loaded from {original_name}")
                return []
            
            logger.info(f"Loaded {len(raw_documents)} pages from {original_name}")
            
            for doc in raw_documents:
                doc.page_content = replace_newlines(doc.page_content)
            for i, doc in enumerate(raw_documents):
                doc.metadata.update({
                    'source': original_name,
                    'file_type': file_extension,
                    'page_number': i + 1,
                    'session_id': self.session_id,  # CRITICAL: Session tracking
                    'upload_timestamp': upload_timestamp,
                    'processing_timestamp': datetime.now().isoformat()
                })
            
            # Split documents into chunks
            split_documents = self.text_splitter.split_documents(raw_documents)
            
            if not split_documents:
                logger.warning(f"No chunks created from {original_name}")
                return []
            
            # Add comprehensive chunk metadata
            for i, doc in enumerate(split_documents):
                doc.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(split_documents),
                    'chunk_size': len(doc.page_content),
                    'session_id': self.session_id,  # CRITICAL: Ensure session tracking
                    'upload_timestamp': upload_timestamp,
                    'chunk_hash': hash(doc.page_content[:100])  # For deduplication
                })
                
                # Content validation
                if not doc.page_content.strip():
                    logger.warning(f"Empty chunk {i} in {original_name}")
            
            # Filter out empty chunks and very short chunks
            split_documents = [doc for doc in split_documents 
                              if doc.page_content.strip() and len(doc.page_content.strip()) > 20]
            
            logger.info(f"Created {len(split_documents)} valid chunks from {original_name}")
            return split_documents
            
        except Exception as e:
            logger.error(f"Error processing file {original_name}: {str(e)}")
            return []

    def load_vectorstore(self) -> Optional[FAISS]:
        """Load the saved vectorstore for current session."""
        try:
            if os.path.exists(self.vectorstore_path):
                vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Vectorstore loaded for session {self.session_id}")
                return vectorstore
            else:
                logger.info(f"No vectorstore found for session {self.session_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            return None

    def _save_vectorstore(self, vectorstore: FAISS):
        """Save vectorstore to disk."""
        try:
            # Remove existing vectorstore directory if it exists
            if os.path.exists(self.vectorstore_path):
                import shutil
                shutil.rmtree(self.vectorstore_path)
            
            vectorstore.save_local(self.vectorstore_path)
            logger.info(f"Vectorstore saved for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving vectorstore: {str(e)}")
            raise

    def _save_documents_info(self, documents_info: List[Dict]):
        """Save document information for current session."""
        try:
            with open(self.documents_info_path, 'wb') as f:
                pickle.dump(documents_info, f)
            logger.info(f"Document info saved for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving document info: {str(e)}")

    def get_documents_info(self) -> List[Dict]:
        """Get information about processed documents in current session."""
        try:
            if os.path.exists(self.documents_info_path):
                with open(self.documents_info_path, 'rb') as f:
                    return pickle.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading document info: {str(e)}")
            return []

    def clear_vectorstore(self):
        """Clear the vectorstore and document info for current session."""
        try:
            if os.path.exists(self.vectorstore_path):
                import shutil
                shutil.rmtree(self.vectorstore_path)
                logger.info(f"Vectorstore cleared for session {self.session_id}")
            if os.path.exists(self.documents_info_path):
                os.remove(self.documents_info_path)
                logger.info(f"Document info cleared for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error clearing vectorstore: {str(e)}")

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data."""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            # Clean up old vectorstore directories
            for item in os.listdir('.'):
                if item.startswith('vectorstore_') and item != f'vectorstore_{self.session_id}':
                    try:
                        import shutil
                        shutil.rmtree(item)
                        cleaned_count += 1
                    except:
                        pass
                        
                # Clean up old document info files
                if item.startswith('documents_info_') and item.endswith('.pkl') and item != f'documents_info_{self.session_id}.pkl':
                    try:
                        os.remove(item)
                        cleaned_count += 1
                    except:
                        pass
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old session files")
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {str(e)}")

    def test_retrieval(self, query: str, k: int = 5) -> List[Dict]:
        """Test document retrieval for current session."""
        vectorstore = self.load_vectorstore()
        if not vectorstore:
            logger.warning(f"No vectorstore available for session {self.session_id}")
            return []
        
        try:
            logger.info(f"Testing retrieval for query: '{query}' in session {self.session_id}")
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
            results = []
            
            for doc, score in docs_with_scores:
                # Verify document belongs to current session
                doc_session = doc.metadata.get('session_id', 'unknown')
                results.append({
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'score': float(score),
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page_number', 'N/A'),
                    'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                    'session_id': doc_session,
                    'is_current_session': doc_session == self.session_id,
                    'upload_timestamp': doc.metadata.get('upload_timestamp', 'N/A')
                })
            
            logger.info(f"Retrieved {len(results)} documents for session {self.session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error testing retrieval: {str(e)}")
            return []

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session's vectorstore."""
        vectorstore = self.load_vectorstore()
        documents_info = self.get_documents_info()
        
        stats = {
            'session_id': self.session_id,
            'vectorstore_exists': vectorstore is not None,
            'vectorstore_path': self.vectorstore_path,
            'total_files': len(documents_info),
            'total_chunks': sum(file_info.get('chunks', 0) for file_info in documents_info),
            'files': documents_info
        }
        
        if vectorstore:
            try:
                # Test search and verify session isolation
                test_docs = vectorstore.similarity_search("test", k=5)
                stats['test_search_works'] = len(test_docs) > 0
                stats['all_chunks_current_session'] = all(
                    doc.metadata.get('session_id') == self.session_id for doc in test_docs
                )
                if test_docs:
                    stats['sample_content'] = test_docs[0].page_content[:100] + "..."
                    stats['sample_session_id'] = test_docs[0].metadata.get('session_id')
            except Exception as e:
                stats['test_search_works'] = False
                stats['error'] = str(e)
        
        return stats
