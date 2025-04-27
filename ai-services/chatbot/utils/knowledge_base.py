import os
import glob
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader, 
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENT_TYPES
)

class KnowledgeBase:
    """
    Knowledge Base manager for the WMS Chatbot.
    Uses Chroma DB for vector storage and OpenAI embeddings.
    """
    
    def __init__(self):
        """Initialize the knowledge base with OpenAI embeddings and Chroma DB."""
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    def load_documents(self, directory: str, doc_type: str) -> List[Document]:
        """
        Load documents from a directory and tag them with metadata.
        
        Args:
            directory: Path to directory containing documents
            doc_type: Type of documents (sop, product, etc.)
            
        Returns:
            List of loaded documents
        """
        # Validate document type
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(f"Invalid document type: {doc_type}. Must be one of {list(DOCUMENT_TYPES.keys())}")
        
        # Check if directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Create loaders for different file types
        loaders = []
        
        # Text files
        if glob.glob(os.path.join(directory, "*.txt")):
            loaders.append(DirectoryLoader(
                directory, 
                glob="**/*.txt",
                loader_cls=TextLoader
            ))
        
        # PDF files
        if glob.glob(os.path.join(directory, "*.pdf")):
            loaders.append(DirectoryLoader(
                directory, 
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            ))
        
        # CSV files
        if glob.glob(os.path.join(directory, "*.csv")):
            loaders.append(DirectoryLoader(
                directory, 
                glob="**/*.csv",
                loader_cls=CSVLoader
            ))
        
        # Load documents
        documents = []
        for loader in loaders:
            docs = loader.load()
            # Add metadata
            for doc in docs:
                doc.metadata["type"] = doc_type
                doc.metadata["description"] = DOCUMENT_TYPES[doc_type]
            documents.extend(docs)
        
        return documents
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Process documents and add them to the knowledge base.
        
        Args:
            documents: List of documents to add
        """
        # Split documents into chunks
        chunks = self.splitter.split_documents(documents)
        
        # Add documents to the vector store
        self.db.add_documents(chunks)
        
        # Persist the database
        self.db.persist()
        
        print(f"Added {len(chunks)} document chunks to the knowledge base")
    
    def query(self, query: str, n_results: int = 5, 
              doc_type: Optional[str] = None) -> List[Document]:
        """
        Query the knowledge base for relevant documents.
        
        Args:
            query: The query string
            n_results: Number of results to return
            doc_type: Optional filter by document type
            
        Returns:
            List of relevant documents
        """
        # Build filter if doc_type is provided
        filter_dict = None
        if doc_type:
            if doc_type not in DOCUMENT_TYPES:
                raise ValueError(f"Invalid document type: {doc_type}")
            filter_dict = {"type": doc_type}
        
        # Query the vector store
        results = self.db.similarity_search(
            query=query,
            k=n_results,
            filter=filter_dict
        )
        
        return results
    
    def add_text(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        Add a text snippet to the knowledge base with metadata.
        
        Args:
            text: Text content
            metadata: Metadata to attach to the document
        """
        # Create a document
        document = Document(page_content=text, metadata=metadata)
        
        # Split into chunks
        chunks = self.splitter.split_documents([document])
        
        # Add to vector store
        self.db.add_documents(chunks)
        
        # Persist the database
        self.db.persist()
        
        print(f"Added text with {len(chunks)} chunks to the knowledge base")

# Initialize singleton instance
knowledge_base = KnowledgeBase()