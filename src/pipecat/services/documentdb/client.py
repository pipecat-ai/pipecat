"""This module contains the DocumentDBStore class, which is a vector store implementation for DocumentDB."""

import os
from typing import Any, Dict, List, Mapping

import pymongo
from loguru import logger
from pymongo.collection import Collection as MongoCollection
from pymongo.errors import PyMongoError


class DocumentDBStore:
    """DocumentDB implementation of vector store."""

    def __init__(self, db_name: str):
        try:
            mongodb_url = os.getenv("MONGODB_URL")
            assert mongodb_url, "MONGODB_URL is not set"
            self.client = pymongo.MongoClient(mongodb_url)
            self.db = self.client[db_name]
        except PyMongoError as e:
            logger.error(f"Failed to connect to DocumentDB: {str(e)}")
            raise

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """Create a new collection in DocumentDB."""
        if collection_name in self.db.list_collection_names():
            logger.info(f"Collection '{collection_name}' already exists.")
            return

        available_distance_metrics = ["euclidean", "cosine", "dotProduct"]

        if distance_metric not in available_distance_metrics:
            raise ValueError(f"Distance metric {distance_metric} is not available")

        index_options = {
            "vectorOptions": {
                "type": kwargs.get("index_type", "hnsw"),
                "dimensions": vector_size,
                "similarity": distance_metric.lower(),
            }
        }

        try:
            self.db[collection_name].create_index("vectorEmbedding", **index_options)
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}", exc_info=True)

    def get_collection(self, collection_name: str) -> MongoCollection:
        """Get a collection from DocumentDB."""
        return self.db[collection_name]

    def scroll(
        self,
        collection_name: str,
        filter: Mapping[str, Any],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Scroll through the collection."""
        try:
            return list(self.db[collection_name].find(filter).limit(limit))
        except Exception as e:
            logger.error(f"Failed to scroll: {str(e)}", exc_info=True)
            return

    def delete_records(self, collection_name: str, filter: Mapping[str, Any]) -> None:
        """Delete a knowledge base from the collection."""
        try:
            result = self.db[collection_name].delete_many(filter)
            logger.info(f"Deleted {result.deleted_count} records from '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to delete records: {str(e)}", exc_info=True)
