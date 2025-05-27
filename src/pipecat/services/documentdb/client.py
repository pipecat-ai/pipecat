"""This module contains the DocumentDBStore class, which is a vector store implementation for DocumentDB."""

import os
from typing import Any, Dict, List, Mapping, Union

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase


class DocumentDBStore:
    """DocumentDB implementation of vector store."""

    def __init__(self, db_name: str):
        self.mongodb_url = os.getenv("MONGODB_URL")
        assert self.mongodb_url, "MONGODB_URL is not set"
        self.client: Union[None, AsyncIOMotorClient] = None
        self.db: Union[None, AsyncIOMotorDatabase] = None
        self.db_name = db_name

    async def ensure_connection(self):
        """Ensure that the connection to DocumentDB is established."""
        try:
            if self.client is None:
                logger.debug("Creating new MongoDB client connection")
                self.client = AsyncIOMotorClient(self.mongodb_url)
                self.db = self.client[self.db_name]
                logger.debug(f"Connected to database: {self.db_name}")

            if self.db is None:
                self.db = self.client[self.db_name]

            # Test the connection
            await self.db.command("ping")
            logger.debug("Successfully connected to MongoDB")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}", exc_info=True)
            raise

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """Create a new collection in DocumentDB."""
        await self.ensure_connection()
        if collection_name in await self.db.list_collection_names():
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
            await self.db[collection_name].create_index("vectorEmbedding", **index_options)
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}", exc_info=True)

    async def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection from DocumentDB."""
        try:
            await self.ensure_connection()
            return self.db[collection_name]
        except Exception as e:
            logger.error(f"Failed to get collection: {str(e)}", exc_info=True)
            return None

    async def scroll(
        self,
        collection_name: str,
        filter: Mapping[str, Any],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Scroll through the collection."""
        try:
            await self.ensure_connection()
            cursor = self.db[collection_name].find(filter).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Failed to scroll: {str(e)}", exc_info=True)
            return []

    async def delete_records(self, collection_name: str, filter: Mapping[str, Any]) -> None:
        """Delete a knowledge base from the collection."""
        try:
            await self.ensure_connection()
            result = await self.db[collection_name].delete_many(filter)
            logger.info(f"Deleted {result.deleted_count} records from '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to delete records: {str(e)}", exc_info=True)
