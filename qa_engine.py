# from typing import Dict, List, Any
# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from document_processor import DocumentProcessor
# from llm_manager import LLMManager
# from web_search import WebSearcher
# from config import config
# from utils import logger

# class QAEngine:
#     def __init__(self):
#         self.document_processor = DocumentProcessor()
#         self.llm_manager = LLMManager()
#         self.web_searcher = WebSearcher()
        
#         # Custom prompt template
#         self.prompt_template = PromptTemplate(
#             template="""Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.

# Context from Documents:
# {doc_context}

# Web Context:
# {web_context}

# Question: {question}

# Answer: Provide a comprehensive answer based on the available context. If information comes from multiple sources, synthesize them coherently. Cite your sources when possible.""",
#             input_variables=["doc_context", "web_context", "question"]
#         )
    
#     def answer_query(self, query: str, llm_type: str) -> Dict[str, Any]:
#         """Answer a query using both document and web context."""
#         try:
#             # Get document context
#             doc_context, doc_sources = self._get_document_context(query)
            
#             # Get web context
#             web_context, web_sources = self._get_web_context(query)
            
#             # Generate answer
#             llm = self.llm_manager.get_llm(llm_type)
            
#             prompt = self.prompt_template.format(
#                 doc_context=doc_context,
#                 web_context=web_context,
#                 question=query
#             )
            
#             answer = llm.generate(prompt)
            
#             return {
#                 'answer': answer,
#                 'sources': doc_sources + web_sources,
#                 'doc_context': doc_context,
#                 'web_context': web_context
#             }
            
#         except Exception as e:
#             logger.error(f"Error answering query: {str(e)}")
#             return {
#                 'answer': f"Error processing query: {str(e)}",
#                 'sources': [],
#                 'doc_context': '',
#                 'web_context': ''
#             }
    
#     def _get_document_context(self, query: str) -> tuple[str, List[str]]:
#         """Get relevant context from documents."""
#         vectorstore = self.document_processor.load_vectorstore()
        
#         if not vectorstore:
#             return "", []
        
#         try:
#             # Retrieve relevant documents
#             docs = vectorstore.similarity_search(query, k=config.MAX_RETRIEVAL_DOCS)
            
#             # Combine context
#             context_parts = []
#             sources = []
            
#             for doc in docs:
#                 context_parts.append(doc.page_content)
#                 source = doc.metadata.get('source', 'Unknown')
#                 if source not in sources:
#                     sources.append(source)
            
#             context = "\n\n".join(context_parts)
#             return context, sources
            
#         except Exception as e:
#             logger.error(f"Error retrieving document context: {str(e)}")
#             return "", []
    
#     def _get_web_context(self, query: str) -> tuple[str, List[str]]:
#         """Get relevant context from web search."""
#         try:
#             results = self.web_searcher.search(query)
            
#             context_parts = []
#             sources = []
            
#             for result in results:
#                 context_parts.append(f"{result['title']}: {result['snippet']}")
#                 sources.append(result['url'])
            
#             context = "\n\n".join(context_parts)
#             return context, sources
            
#         except Exception as e:
#             logger.error(f"Error getting web context: {str(e)}")
#             return "", []



from typing import Dict, List, Any, Tuple
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from document_processor import DocumentProcessor
from llm_manager import LLMManager
from web_search import WebSearcher
from utils import logger

class QAEngine:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.llm_manager = LLMManager()
        self.web_searcher = WebSearcher()
        
        # Enhanced prompt template that prioritizes document content
        self.prompt_template = PromptTemplate(
            template="""
CRITICAL:You are an intelligent expert that answers questions using uploaded documents first, then web search when use_web_search=true or when documents lack relevant content. You MUST provide reliable responses by citing all sources clearly - use [Doc: "title" p.X] for documents and [Web: "source" date] for web sources. Strictly avoid speculation beyond provided context and never make uncited claims.
📄 DOCUMENT CONTENT (PRIMARY SOURCE):
{doc_context}
🌐 WEB INFORMATION (SUPPLEMENTARY):
{web_context}
❓ USER QUESTION: {question}
ANALYSIS:chu
- Document chunks found: {num_doc_chunks}
- Document relevance: {doc_relevance_score}/10

✅ ANSWER:
Based on the available information, here is my response:

{answer_instruction}
""",
            input_variables=["doc_context", "web_context", "question", "doc_relevance_score", "num_doc_chunks", "answer_instruction"]
        )

    def answer_query(self, query: str, llm_type: str = None, use_web_search: bool = True) -> Dict[str, Any]:
        """Main entry: answers query prioritizing documents first, then supplementing with web."""
        try:
            # Get LLM type from session state or parameter
            if llm_type is None:
                llm_type = st.session_state.get("llm_type", "local")
            
            logger.info(f"Processing query: '{query[:50]}...' with LLM backend: {llm_type}")

            # Step 1: Get document context first
            doc_context, doc_sources, doc_relevance_score, num_doc_chunks = self._get_enhanced_document_context(query)
            
            # Step 2: Get web context (conditional based on document completeness)
            web_context, web_sources = self._get_conditional_web_context(query, doc_relevance_score, num_doc_chunks, use_web_search)
            
            # Step 3: Create dynamic answer instruction
            answer_instruction = self._create_answer_instruction(doc_relevance_score, num_doc_chunks)
            
            # Step 4: Generate answer
            llm = self.llm_manager.get_llm(llm_type)

            prompt = self.prompt_template.format(
                doc_context=doc_context if doc_context else "No relevant document content found.",
                web_context=web_context if web_context else "No web information retrieved.",
                question=query,
                doc_relevance_score=doc_relevance_score,
                num_doc_chunks=num_doc_chunks,
                answer_instruction=answer_instruction
            )

            logger.info(f"Document chunks: {num_doc_chunks}, Relevance: {doc_relevance_score}/10")
            logger.info(f"Prompt length: {len(prompt)} characters")

            answer = llm.generate(prompt)

            return {
                'answer': answer,
                'sources': doc_sources + web_sources,
                'doc_context': doc_context,
                'web_context': web_context,
                'doc_relevance_score': doc_relevance_score,
                'num_doc_chunks': num_doc_chunks,
                'strategy_used': self._get_strategy_description(doc_relevance_score, num_doc_chunks)
            }

        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                'answer': f"❌ Error processing query: {str(e)}",
                'sources': [],
                'doc_context': '',
                'web_context': '',
                'doc_relevance_score': 0,
                'num_doc_chunks': 0,
                'strategy_used': 'Error occurred'
            }

    def _get_enhanced_document_context(self, query: str) -> Tuple[str, List[str], int, int]:
        """Enhanced document retrieval with better relevance scoring."""
        vectorstore = self.document_processor.load_vectorstore()
        
        if not vectorstore:
            logger.warning("No vectorstore available - no documents processed")
            return "", [], 0, 0

        try:
            # Increase number of documents retrieved for better coverage
            k = 4  # Reduced for shorter prompts
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
            
            if not docs_with_scores:
                logger.warning("No documents retrieved from vectorstore")
                return "", [], 0, 0

            logger.info(f"Retrieved {len(docs_with_scores)} documents from vectorstore")

            # Log the scores to understand the distribution
            scores = [score for _, score in docs_with_scores]
            logger.info(f"Document scores: {scores}")

            # More lenient relevance threshold (FAISS uses cosine distance, lower is better)
            relevance_threshold = 1.2  # Increased threshold to include more documents
            relevant_docs = [(doc, score) for doc, score in docs_with_scores if score < relevance_threshold]
            
            if not relevant_docs:
                # If no docs meet threshold, take the top 5 anyway
                relevant_docs = docs_with_scores[:5]
                logger.info(f"No docs met relevance threshold, using top 5. Best score: {docs_with_scores[0][1]:.3f}")
            else:
                logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            # Sort by relevance score and take top documents
            relevant_docs = sorted(relevant_docs, key=lambda x: x[1])[:3]  # Take top 3
            
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(relevant_docs):
                # Add document content with enhanced metadata
                source = doc.metadata.get('source', 'Unknown Document')
                page = doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))
                chunk_id = doc.metadata.get('chunk_id', i)
                
                # Create a more informative context entry
                context_part = f"""
--- Document Chunk {i+1} (Relevance Score: {score:.3f}) ---
Source: {source} | Page: {page} | Chunk: {chunk_id}
Content:
{doc.page_content}
"""
                context_parts.append(context_part)
                
                source_info = f"📄 {source} (Page {page})"
                if source_info not in sources:
                    sources.append(source_info)
            
            context = "\n".join(context_parts)
            
            # Calculate relevance score (0-10 scale)
            if relevant_docs:
                avg_score = sum(score for _, score in relevant_docs) / len(relevant_docs)
                # Convert to 0-10 scale (FAISS cosine distance: 0 = perfect match, 2 = orthogonal)
                # Map [0, 2] to [10, 0] for intuitive scoring
                relevance_score = max(0, min(10, int(10 * (1 - avg_score / 2))))
            else:
                relevance_score = 0
            
            num_chunks = len(relevant_docs)
            
            logger.info(f"Document context created: {num_chunks} chunks, relevance score: {relevance_score}/10")
            
            return context, sources, relevance_score, num_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document context: {str(e)}")
            return "", [], 0, 0

    def _get_conditional_web_context(self, query: str, doc_relevance_score: int, num_doc_chunks: int, use_web_search: bool) -> Tuple[str, List[str]]:
        if not use_web_search:
            logger.info("Web search disabled by user.")
            return "", []
        """Get web context conditionally based on document completeness."""
        try:
            # Determine web search strategy based on document availability and relevance
            if num_doc_chunks == 0:
                # No documents - comprehensive web search
                max_results = 5
                logger.info("No documents found - comprehensive web search")
            elif doc_relevance_score >= 7 and num_doc_chunks >= 3:
                # High document relevance - minimal web search for verification
                max_results = 2
                logger.info("High document relevance - minimal web search")
            elif doc_relevance_score >= 4:
                # Medium document relevance - moderate web search
                max_results = 3
                logger.info("Medium document relevance - moderate web search")
            else:
                # Low document relevance - more comprehensive web search
                max_results = 4
                logger.info("Low document relevance - enhanced web search")
            
            results = self.web_searcher.search(query)
            
            # Limit results based on strategy
            limited_results = results[:max_results]
            
            context_parts = []
            sources = []
            
            for i, result in enumerate(limited_results):
                context_part = f"""
--- Web Result {i+1} ---
Title: {result['title']}
Source: {result['url']}
Content: {result['snippet']}
"""
                context_parts.append(context_part)
                sources.append(f"🌐 {result['url']}")
            
            context = "\n".join(context_parts)
            logger.info(f"Web context created with {len(limited_results)} results")
            return context, sources
            
        except Exception as e:
            logger.error(f"Error getting web context: {str(e)}")
            return "", []

    def _create_answer_instruction(self, doc_relevance_score: int, num_doc_chunks: int) -> str:
        """Create dynamic instructions based on available content."""
        if num_doc_chunks == 0:
            return """Since no relevant documents were found in the uploaded files, base your answer on the web search results. 
                     Clearly indicate that the response is based on web sources only."""
        
        elif doc_relevance_score >= 7:
            return """The uploaded documents contain highly relevant information for this query. 
                     Prioritize the document content in your answer and use web sources only to supplement 
                     or provide additional recent context. Clearly cite which information comes from documents vs web."""
        
        elif doc_relevance_score >= 4:
            return """The uploaded documents contain moderately relevant information. 
                     Synthesize information from both documents and web sources, giving preference to 
                     document content where applicable. Clearly distinguish between document and web sources in your answer."""
        
        else:
            return """The uploaded documents have limited relevance to this query. 
                     Use web sources as the primary information source, but include any relevant 
                     document information if applicable. Clearly indicate the source of each piece of information."""

    def _get_strategy_description(self, doc_relevance_score: int, num_doc_chunks: int) -> str:
        """Get a description of the strategy used for answering."""
        if num_doc_chunks == 0:
            return "Web-only search (no relevant documents found)"
        elif doc_relevance_score >= 7:
            return f"Document-primary strategy ({num_doc_chunks} highly relevant chunks)"
        elif doc_relevance_score >= 4:
            return f"Balanced document-web synthesis ({num_doc_chunks} moderately relevant chunks)"
        else:
            return f"Web-primary with document context ({num_doc_chunks} low-relevance chunks)"

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information."""
        try:
            vectorstore = self.document_processor.load_vectorstore()
            documents_info = self.document_processor.get_documents_info()
            vectorstore_stats = self.document_processor.get_vectorstore_stats()
            
            debug_info = {
                'vectorstore_available': vectorstore is not None,
                'documents_processed': len(documents_info),
                'total_chunks': sum(doc.get('chunks', 0) for doc in documents_info),
                'files_info': documents_info,
                'vectorstore_stats': vectorstore_stats
            }
            
            # Test retrieval if vectorstore exists
            if vectorstore:
                test_results = self.document_processor.test_retrieval("test query", k=3)
                debug_info['test_retrieval'] = {
                    'results_count': len(test_results),
                    'sample_results': test_results[:2] if test_results else []
                }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error getting debug info: {str(e)}")
            return {'error': str(e)}

    def test_document_retrieval(self, query: str) -> Dict[str, Any]:
        """Test document retrieval for a specific query."""
        try:
            doc_context, doc_sources, doc_relevance_score, num_doc_chunks = self._get_enhanced_document_context(query)
            
            return {
                'query': query,
                'num_chunks_found': num_doc_chunks,
                'relevance_score': doc_relevance_score,
                'sources': doc_sources,
                'context_preview': doc_context[:500] + "..." if len(doc_context) > 500 else doc_context,
                'full_context_length': len(doc_context)
            }
        except Exception as e:
            logger.error(f"Error testing document retrieval: {str(e)}")
            return {'error': str(e)}
        
      