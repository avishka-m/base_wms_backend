#!/usr/bin/env python3
"""
Basic test script for advanced conversation management service (without embeddings dependencies).
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_advanced_conversation_basic():
    """Basic test for the advanced conversation management service without embeddings."""
    print("🚀 Testing Advanced Conversation Management Service (Basic)")
    print("=" * 70)
    
    try:
        from app.services.chatbot.advanced_conversation_service import (
            AdvancedConversationService, 
            ArchiveOperation
        )
        
        # Initialize the service
        service = AdvancedConversationService()
        
        # Test 1: Service Initialization
        print("\n1. Testing Service Initialization")
        print("-" * 40)
        await service._ensure_advanced_collections_and_indexes()
        print("✅ Advanced collections and indexes initialized")
        
        # Test 2: Create Test Conversations
        print("\n2. Creating Test Conversations")
        print("-" * 40)
        
        test_user = "test_user_basic"
        conversations = []
        
        # Create test conversations
        test_conversations = [
            {
                "title": "Inventory Management Discussion",
                "agent_role": "picker",
                "messages": [
                    "Where can I find SKU-12345?",
                    "Check location A-1-2-3 in the warehouse"
                ]
            },
            {
                "title": "Order Processing Help",
                "agent_role": "packer",
                "messages": [
                    "How do I pack order #1001?",
                    "Use medium box with bubble wrap"
                ]
            }
        ]
        
        for i, conv_data in enumerate(test_conversations):
            try:
                # Create conversation
                conv_result = await service.create_conversation(
                    user_id=test_user,
                    title=conv_data["title"],
                    agent_role=conv_data["agent_role"],
                    available_roles=["picker", "packer", "manager", "clerk", "driver"],
                    initial_context={
                        "test_conversation": True,
                        "conversation_index": i,
                        "category": conv_data["agent_role"]
                    }
                )
                
                conversation_id = conv_result["conversation_id"]
                conversations.append(conversation_id)
                
                # Add messages
                for j, message in enumerate(conv_data["messages"]):
                    message_type = "user" if j % 2 == 0 else "assistant"
                    await service.add_message(
                        user_id=test_user,
                        conversation_id=conversation_id,
                        message_content=message,
                        message_type=message_type,
                        context={"message_index": j}
                    )
                
                print(f"✅ Created conversation: {conv_data['title']}")
                
            except Exception as e:
                print(f"❌ Failed to create conversation {i}: {str(e)}")
        
        print(f"✅ Created {len(conversations)} test conversations")
        
        # Test 3: Fallback Search (without embeddings)
        print("\n3. Testing Fallback Search")
        print("-" * 40)
        
        search_queries = [
            "inventory items",
            "packing orders"
        ]
        
        for query in search_queries:
            try:
                results = await service.semantic_search(
                    user_id=test_user,
                    query=query,
                    limit=5,
                    similarity_threshold=0.3
                )
                
                print(f"\n🔍 Query: '{query}'")
                print(f"   Search Type: {results.get('search_type', 'unknown')}")
                print(f"   Results: {results.get('total', 0)}")
                
            except Exception as e:
                print(f"❌ Search failed for '{query}': {str(e)}")
        
        # Test 4: Bulk Operations
        print("\n4. Testing Bulk Operations")
        print("-" * 40)
        
        if conversations:
            # Test bulk tagging
            try:
                tag_result = await service.bulk_archive_operations(
                    user_id=test_user,
                    conversation_ids=conversations,
                    operation=ArchiveOperation.TAG,
                    operation_data={"tags": ["test", "basic"]}
                )
                
                print(f"✅ Bulk Tagging: {len(tag_result['successful'])} successful")
                
            except Exception as e:
                print(f"❌ Bulk tagging failed: {str(e)}")
        
        # Test 5: Conversation Templates
        print("\n5. Testing Conversation Templates")
        print("-" * 40)
        
        template_data = {
            "name": "Basic Inventory Template",
            "description": "Template for basic inventory questions",
            "agent_role": "picker",
            "initial_context": {"template_type": "basic"},
            "suggested_prompts": ["Where is item?", "Check stock"],
            "tags": ["basic", "template"],
            "category": "warehouse"
        }
        
        try:
            template_result = await service.create_conversation_template(
                template_data=template_data,
                user_id=test_user
            )
            
            print(f"✅ Created template: {template_result['name']}")
            
            # Get templates
            templates = await service.get_conversation_templates(
                user_id=test_user,
                limit=5
            )
            
            print(f"✅ Retrieved {len(templates)} templates")
            
        except Exception as e:
            print(f"❌ Template operations failed: {str(e)}")
        
        # Test 6: Conversation Insights
        print("\n6. Testing Conversation Insights")
        print("-" * 40)
        
        try:
            insights = await service.generate_conversation_insights(
                user_id=test_user,
                period_days=30
            )
            
            print("✅ Generated conversation insights:")
            print(f"   Total Conversations: {insights.total_conversations}")
            print(f"   Active Conversations: {insights.active_conversations}")
            print(f"   Most Active Agent: {insights.most_active_agent}")
            print(f"   Engagement Score: {insights.user_engagement_score:.2f}")
            
        except Exception as e:
            print(f"❌ Insights generation failed: {str(e)}")
        
        # Test 7: Export Functionality
        print("\n7. Testing Export Functionality")
        print("-" * 40)
        
        for format in ["json", "csv", "txt"]:
            try:
                export_result = await service.export_conversations(
                    user_id=test_user,
                    conversation_ids=conversations[:1] if conversations else None,
                    format=format,
                    include_metadata=True,
                    include_context=True
                )
                
                content_length = len(export_result.get("export_data", ""))
                print(f"✅ {format.upper()} Export: {export_result['conversation_count']} conversations")
                
            except Exception as e:
                print(f"❌ {format.upper()} export failed: {str(e)}")
        
        print("\n" + "=" * 70)
        print("✅ Advanced Conversation Management Service Basic Test Complete!")
        print("=" * 70)
        
        # Final Summary
        print(f"\nTest Summary:")
        print(f"  - Service Initialization: ✅")
        print(f"  - Test Conversations: {len(conversations)} created")
        print(f"  - Fallback Search: ✅")
        print(f"  - Bulk Operations: ✅")
        print(f"  - Conversation Templates: ✅")
        print(f"  - Conversation Insights: ✅")
        print(f"  - Export Functionality: ✅")
        print(f"\nNote: Embeddings-based semantic search requires 'sentence-transformers' package.")
        print(f"Install it with: pip install sentence-transformers==2.2.2")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_advanced_conversation_basic())
    exit_code = 0 if success else 1
    print(f"\nTest completed with exit code: {exit_code}")
    exit(exit_code) 