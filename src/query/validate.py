import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class MuRAGDatabase:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initializes the MuRAG Database for encoding and retrieval.

        Args:
            model_name (str): Pretrained model for encoding text.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embeddings = []
        self.document_ids = []

    def encode_text(self, text):
        """
        Encodes text into an embedding using the specified model.

        Args:
            text (str): The text to encode.

        Returns:
            np.ndarray: Encoded embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def add_document(self, doc_id, text):
        """
        Encodes and adds a document to the database.

        Args:
            doc_id (str): Document ID.
            text (str): Text content of the document.
        """
        embedding = self.encode_text(text)
        self.embeddings.append(embedding)
        self.document_ids.append(doc_id)

    def build_index(self):
        """
        Builds a nearest neighbor index for efficient retrieval.
        """
        self.embeddings = np.vstack(self.embeddings)
        self.index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.embeddings)

    def query(self, text):
        """
        Retrieves the top-5 most relevant documents for a query.

        Args:
            text (str): Query text.

        Returns:
            list: List of document IDs for the top-5 results.
        """
        query_embedding = self.encode_text(text)
        distances, indices = self.index.kneighbors(query_embedding)
        return [self.document_ids[i] for i in indices[0]]


def build_rag_dataset(dataset_path, model_name="bert-base-uncased"):
    """
    Builds a RAG database from the Hugging Face dataset.

    Args:
        dataset_path (str): Path to the Hugging Face dataset.
        model_name (str): Model for encoding text.

    Returns:
        MuRAGDatabase: The built RAG database object.
    """
    rag_db = MuRAGDatabase(model_name=model_name)
    dataset = load_from_disk(dataset_path)

    for record in dataset:
        doc_id = record["id"]
        text_content = " ".join(
            [content["value"] for content in record["document"] if content["type"] == "text"]
        )
        rag_db.add_document(doc_id, text_content)

    rag_db.build_index()
    return rag_db


def validate_query(rag_db, query):
    """
    Validates a query using the MuRAG database.

    Args:
        rag_db (MuRAGDatabase): The MuRAG database object.
        query (str): The query to validate.

    Returns:
        bool: True if the query is valid, False otherwise.
    """
    top_documents = rag_db.query(query)
    unique_sources = set(doc_id.split("-")[0] for doc_id in top_documents)

    # A query is valid if it retrieves documents from more than one source
    return len(unique_sources) > 1