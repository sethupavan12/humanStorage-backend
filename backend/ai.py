"""
This file handles the AI part of our document storing and retrieval blockchain platform.


The user asks a question, then following happens
1. User's question is sent to backend to analyze what category of documents the user is asking for.
2. Frontend gets what to search for and retrieves all the documents of that category.
3. The documents are then sent to the backend to be vectorized along with the user's question.
4. AI will answer the question like RAG.
5. The answer is then sent to the frontend to be displayed.


Types of documents:
1. Medical Records
2. Passports
3. Emergency Information
4. Education Transcripts
5. Legal Contracts 
6. Misc - Birth Certificate, Intellectual Property Documentation
"""
import sqlite3
import chromadb

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain import hub
PERSISTENT_DIR="./data"
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Chroma(persist_directory=PERSISTENT_DIR, embedding_function=embedding_function)

class AI:
    """
    This class handles the AI part.
    """

    def __init__(self,openai=True):
        # self.client = app.config["client"]
        # self.client = chromadb.HttpClient(host='localhost', port='8080')

        if openai:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
            self.embeddings = OpenAIEmbeddings()
        else:
            self.llm=None
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_documents_from_unstrctured_data(self, list_of_paths):
        """
        Loads the documents from the unstructured data.
        """
        loader = UnstructuredFileLoader(list_of_paths)
        self.docs = loader.load_and_split()
        return self.docs
    
    def check_if_collection_exists(self, vector_db,collection_name):
        """
        Checks if the given collection exists.
        """
        if vectordb._collection.count() > 0:
            return True
        return False
    
    def create_collection_and_put_it_in_db(self,list_of_paths, collection_name):
        docs = load_documents_from_unstrctured_data(list_of_paths)

        vectordb = Chroma.from_documents(docs, embedding=self.embeddings, persist_directory = PERSISTENT_DIR, collection_name=collection_name)
        vectordb.persist()
        return vectordb
        print("Collection created and persisted")

    def load_collection_from_db(self, collection_name):
        vectordb = Chroma(persist_directory= PERSISTENT_DIR, embedding_function=self.embeddings, collection_name=collection_name)
        print(f"Number of collections in {collection_name} is {vectordb._collection.count()}")
        return vectordb
    
    def get_retriever_for_given_vectordb(self, vectordb):
        return vectordb.as_retriever()

    def get_answer(self, question, db):
        """
        Gets the answer to the given question.
        """
        rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
        chain = RetrievalQA.from_chain_type(llm=self.llm,
                                    chain_type="stuff",
                                    chain_type_kwargs={"prompt": rag_prompt_llama},
                                    retriever=db)
        response = chain(question)
        return response

    
    



