#!/usr/bin/env python3
"""
Database migration script for WMS Chatbot enhanced features.
This script sets up the MongoDB collections and indexes for persistent chat storage.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pymongo import ASCENDING, DESCENDING, TEXT
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection URL
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot_migration")

async def create_collections_and_indexes():
    """Create MongoDB collections and indexes for the enhanced chatbot."""
    
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        
        logger.info(f"Connected to MongoDB: {DATABASE_NAME}")
        
        # Collection definitions
        collections = {
            "chat_conversations": {
                "description": "User conversation metadata and settings",
                "indexes": [
                    ([("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
                    ([("conversation_id", ASCENDING)], {"unique": True}),
                    ([("status", ASCENDING)], {}),
                    ([("agent_role", ASCENDING)], {}),
                    ([("tags", ASCENDING)], {}),
                    ([("last_message_at", DESCENDING)], {}),
                ]
            },
            "chat_messages": {
                "description": "Individual chat messages with full content",
                "indexes": [
                    ([("conversation_id", ASCENDING), ("timestamp", ASCENDING)], {}),
                    ([("user_id", ASCENDING), ("timestamp", DESCENDING)], {}),
                    ([("message_type", ASCENDING)], {}),
                    ([("content", TEXT)], {}),  # Full-text search
                    ([("has_attachments", ASCENDING)], {}),
                ]
            },
            "chat_attachments": {
                "description": "File attachments for chat messages",
                "indexes": [
                    ([("message_id", ASCENDING)], {}),
                    ([("conversation_id", ASCENDING)], {}),
                    ([("user_id", ASCENDING)], {}),
                    ([("file_type", ASCENDING)], {}),
                    ([("processing_status", ASCENDING)], {}),
                ]
            },
            "chat_user_preferences": {
                "description": "User preferences for chat experience",
                "indexes": [
                    ([("user_id", ASCENDING)], {"unique": True}),
                ]
            },
            "chat_analytics": {
                "description": "Analytics data for chat conversations and user behavior",
                "indexes": [
                    ([("user_id", ASCENDING), ("period_start", DESCENDING)], {}),
                    ([("period_type", ASCENDING)], {}),
                    ([("period_start", DESCENDING)], {}),
                ]
            },
            "chat_audit_logs": {
                "description": "Audit log for chat actions and data access",
                "indexes": [
                    ([("user_id", ASCENDING), ("created_at", DESCENDING)], {}),
                    ([("action_type", ASCENDING)], {}),
                    ([("resource_type", ASCENDING)], {}),
                    ([("conversation_id", ASCENDING)], {}),
                    ([("created_at", DESCENDING)], {}),
                ]
            }
        }
        
        # Create collections and indexes
        for collection_name, config in collections.items():
            logger.info(f"Setting up collection: {collection_name}")
            
            # Get or create collection
            collection = db[collection_name]
            
            # Create indexes
            for index_spec, index_options in config["indexes"]:
                try:
                    index_name = await collection.create_index(index_spec, **index_options)
                    logger.info(f"  Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"  Index already exists or failed: {str(e)}")
            
            # Add collection metadata
            await collection.update_one(
                {"_id": "_metadata"},
                {
                    "$set": {
                        "description": config["description"],
                        "created_at": datetime.utcnow(),
                        "version": "1.0.0"
                    }
                },
                upsert=True
            )
        
        logger.info("All collections and indexes created successfully")
        
        # Test the setup
        await test_collections(db)
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"Failed to create collections and indexes: {str(e)}")
        raise

async def test_collections(db):
    """Test the created collections with sample data."""
    
    logger.info("Testing collections with sample data...")
    
    try:
        # Test conversations collection
        conversations_col = db["chat_conversations"]
        test_conversation = {
            "conversation_id": "test_conversation_001",
            "user_id": "test_user",
            "title": "Test Conversation",
            "agent_role": "manager",
            "status": "active",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0,
            "user_message_count": 0,
            "assistant_message_count": 0,
            "total_tokens_used": 0,
            "has_attachments": False,
            "attachment_count": 0,
            "languages_detected": [],
            "user_preferences": {},
            "conversation_context": {},
            "tags": ["test"],
            "metadata": {"test": True}
        }
        
        result = await conversations_col.insert_one(test_conversation)
        logger.info(f"  Test conversation inserted: {result.inserted_id}")
        
        # Test messages collection
        messages_col = db["chat_messages"]
        test_message = {
            "conversation_id": "test_conversation_001",
            "user_id": "test_user",
            "message_type": "user",
            "content": "This is a test message for indexing",
            "context": {},
            "metadata": {"test": True},
            "timestamp": datetime.utcnow(),
            "has_attachments": False,
            "attachment_count": 0
        }
        
        result = await messages_col.insert_one(test_message)
        logger.info(f"  Test message inserted: {result.inserted_id}")
        
        # Test text search
        search_results = await messages_col.find({"$text": {"$search": "test message"}}).to_list(length=10)
        logger.info(f"  Text search test: found {len(search_results)} results")
        
        # Clean up test data
        await conversations_col.delete_one({"conversation_id": "test_conversation_001"})
        await messages_col.delete_one({"conversation_id": "test_conversation_001"})
        
        logger.info("Collection tests completed successfully")
        
    except Exception as e:
        logger.error(f"Collection tests failed: {str(e)}")
        raise

async def migrate_existing_conversations():
    """Migrate existing conversations from the old in-memory format to MongoDB."""
    
    logger.info("Checking for existing conversations to migrate...")
    
    try:
        # This would be called if there are existing conversations to migrate
        # For now, we'll just create the infrastructure
        
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        
        # Check if there are any existing conversations
        conversations_col = db["chat_conversations"]
        existing_count = await conversations_col.count_documents({})
        
        logger.info(f"Found {existing_count} existing conversations in database")
        
        if existing_count == 0:
            logger.info("No existing conversations found. Migration not needed.")
        else:
            logger.info("Existing conversations found. Use the API migration endpoint to migrate from in-memory storage.")
        
        client.close()
        
    except Exception as e:
        logger.error(f"Migration check failed: {str(e)}")
        raise

def main():
    """Main migration function."""
    
    logger.info("Starting WMS Chatbot database migration...")
    logger.info(f"Target database: {DATABASE_NAME}")
    logger.info(f"MongoDB URL: {MONGODB_URL}")
    
    try:
        # Run async operations
        asyncio.run(create_collections_and_indexes())
        asyncio.run(migrate_existing_conversations())
        
        logger.info("Migration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update your application to use the enhanced conversation service")
        logger.info("2. Test the new chat functionality")
        logger.info("3. Use the API migration endpoint if you have existing conversations")
        logger.info("4. Monitor the chat_audit_logs collection for system activity")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
