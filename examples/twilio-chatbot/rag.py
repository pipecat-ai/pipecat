"""This module contains the DocumentDBVectorStore class, which is a vector store implementation for DocumentDB."""

import uuid
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from loguru import logger
from pymongo.errors import PyMongoError

from pipecat.services.documentdb.client import DocumentDBStore


class DocumentDBVectorStore:
    """DocumentDB implementation of vector store."""

    def __init__(self):
        # Initialize model
        self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize DocumentDB vector store
        self.vector_store = DocumentDBStore("vector_store")
        self.collection_name = "knowledge_base"
        self.vector_store.create_collection(
            collection_name=self.collection_name,
            vector_size=384,
        )

    def add_vectors(
        self,
        collection_name: str,
        knowledge_base_id: str,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """Add vectors to the collection."""
        chunk_ids = chunk_ids or [str(uuid.uuid4()) for _ in vectors]
        payloads = payloads or [{} for _ in vectors]

        if len(payloads) != len(vectors):
            raise ValueError("vectors and payloads must be of same size")

        for payload in payloads:
            payload["knowledge_base_id"] = knowledge_base_id

        documents = [
            {"chunk_id": chunk_id, "vectorEmbedding": vector, **payload}
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
        ]

        try:
            self.vector_store.get_collection(collection_name=collection_name).insert_many(documents)
            logger.info(f"Successfully added {knowledge_base_id} to DocumentDB")
        except PyMongoError as e:
            logger.error(
                f"Failed to add vectors to the '{collection_name}' collection: {str(e)}",
                exc_info=True,
            )

    def _search(
        self,
        collection_name: str,
        query_vector: List[float],
        knowledge_base_id: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection.

        Args:
            collection_name: Name of the collection to search in
            query_vector: Vector to search for
            knowledge_base_id: ID of the knowledge base to search in
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold (between -1 and 1 for cosine similarity)
                        Note: In DocumentDB, score filtering happens client-side

        Returns:
            List of documents matching the search criteria
        """
        pipeline = [
            {"$match": {"knowledge_base_id": knowledge_base_id}},
            {
                "$search": {
                    "vectorSearch": {
                        "vector": query_vector,
                        "path": "vectorEmbedding",
                        "similarity": "cosine",
                        "k": (
                            limit if score_threshold is None else limit * 2
                        ),  # Request more results for filtering
                    }
                }
            },
            {"$project": {"_id": 0}},
        ]

        try:
            results = list(
                self.vector_store.get_collection(collection_name=collection_name).aggregate(
                    pipeline
                )
            )

            # If score threshold is set, calculate scores and filter
            if score_threshold is not None:
                if not -1 <= score_threshold <= 1:
                    raise ValueError(
                        "score_threshold must be between -1 and 1 for cosine similarity"
                    )

                filtered_results = []
                for result in results:
                    # Calculate cosine similarity
                    embedding = result.pop("vectorEmbedding")  # Remove embedding after calculation
                    score = self._calculate_cosine_similarity(query_vector, embedding)
                    if score >= score_threshold:
                        result["score"] = score
                        filtered_results.append(result)

                # Sort by score and limit results
                filtered_results.sort(key=lambda x: x["score"], reverse=True)
                return filtered_results[:limit]

            # If no threshold, just remove the embeddings from results
            for result in results:
                result.pop("vectorEmbedding")
            return results

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}", exc_info=True)
            return []

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(a * a for a in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

    def retrieve(
        self,
        knowledge_base_id: str,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = 0.1,
        call_id: Optional[str] = None,
    ) -> List[str]:
        """Retrieve text from the knowledge base."""
        try:
            results = self._search(
                collection_name=self.collection_name,
                query_vector=list(self.model.embed(query.strip()))[0].tolist(),
                knowledge_base_id=knowledge_base_id,
                limit=limit,
                score_threshold=score_threshold,
            )
            return [result["text"] for result in results]

        except Exception as e:
            logger.error(f"Failed to retrieve text from knowledge base: {str(e)}", exc_info=True)
            raise
