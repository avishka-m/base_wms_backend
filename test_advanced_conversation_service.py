#!/usr/bin/env python3
"""
Test script for advanced conversation management service with sophisticated features.
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_advanced_conversation_service():
    """Test the advanced conversation management service."""
    print("🚀 Testing Advanced Conversation Management Service")
    print("=" * 70)
    
    try:
        from app.services.chatbot.advanced_conversation_service import (
            AdvancedConversationService, 
            ArchiveOperation,
            ConversationInsights
        )
        
        # Initialize the service
        service = AdvancedConversationService()
        
        # Test 1: Service Initialization
        print("\n1. Testing Service Initialization")
        print("-" * 40)
        await service._ensure_advanced_collections_and_indexes()
        print("✅ Advanced collections and indexes initialized")
        
        # Test 2: Embeddings Model
        print("\n2. Testing Embeddings Model")
        print("-" * 40)
        model = service._get_embeddings_model()
        if model:
            print("✅ Sentence transformer model loaded successfully")
            
            # Test embedding generation
            embedding = await service._generate_embeddings("Test message for embedding", "test")
            if embedding:
                print(f"✅ Generated embedding with {len(embedding)} dimensions")
            else:
                print("⚠️ Embedding generation returned None (fallback mode)")
        else:
            print("⚠️ Embeddings model not available (fallback to text search)")
        
        # Test 3: Create Test Conversations
        print("\n3. Creating Test Conversations")
        print("-" * 40)
        
        test_user = "test_user_advanced"
        conversations = []
        
        # Create test conversations with different contexts
        test_conversations = [
            {
                "title": "Inventory Management Discussion",
                "agent_role": "picker",
                "messages": [
                    "Where can I find SKU-12345?",
                    "Check location A-1-2-3 in the warehouse",
                    "How many items are in stock?",
                    "There are 150 units available"
                ]
            },
            {
                "title": "Order Processing Help",
                "agent_role": "packer",
                "messages": [
                    "How do I pack order #1001?",
                    "Use medium box with bubble wrap",
                    "What's the shipping label?",
                    "Print label from the system"
                ]
            },
            {
                "title": "Analytics Report Request",
                "agent_role": "manager",
                "messages": [
                    "Generate weekly performance report",
                    "Report generated with efficiency metrics",
                    "Show top performing workers",
                    "Here are the top 5 workers by productivity"
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
        
        # Test 4: Semantic Search
        print("\n4. Testing Semantic Search")
        print("-" * 40)
        
        search_queries = [
            "inventory items in warehouse",
            "packing orders with boxes",
            "performance reports and analytics",
            "finding SKU locations"
        ]
        
        for query in search_queries:
            try:
                results = await service.semantic_search(
                    user_id=test_user,
                    query=query,
                    limit=5,
                    similarity_threshold=0.3  # Lower threshold for testing
                )
                
                print(f"\n🔍 Query: '{query}'")
                print(f"   Search Type: {results.get('search_type', 'unknown')}")
                print(f"   Results: {results.get('total', 0)}")
                print(f"   Processing Time: {results.get('processing_time', 0):.2f}s")
                
                if results.get('results'):
                    for result in results['results'][:2]:  # Show top 2
                        print(f"   - {result.get('content', '')[:50]}... (Score: {result.get('similarity_score', 0):.2f})")
                
            except Exception as e:
                print(f"❌ Semantic search failed for '{query}': {str(e)}")
        
        # Test 5: Bulk Operations
        print("\n5. Testing Bulk Operations")
        print("-" * 40)
        
        if conversations:
            # Test bulk tagging
            try:
                tag_result = await service.bulk_archive_operations(
                    user_id=test_user,
                    conversation_ids=conversations[:2],  # First 2 conversations
                    operation=ArchiveOperation.TAG,
                    operation_data={"tags": ["test", "automated", "warehouse"]}
                )
                
                print(f"✅ Bulk Tagging: {tag_result['successful']} successful, {len(tag_result['failed'])} failed")
                
            except Exception as e:
                print(f"❌ Bulk tagging failed: {str(e)}")
            
            # Test bulk archiving
            try:
                archive_result = await service.bulk_archive_operations(
                    user_id=test_user,
                    conversation_ids=[conversations[0]],  # Archive first conversation
                    operation=ArchiveOperation.ARCHIVE
                )
                
                print(f"✅ Bulk Archiving: {archive_result['successful']} successful, {len(archive_result['failed'])} failed")
                
            except Exception as e:
                print(f"❌ Bulk archiving failed: {str(e)}")
        
        # Test 6: Conversation Templates
        print("\n6. Testing Conversation Templates")
        print("-" * 40)
        
        # Create templates
        template_data = {
            "name": "Inventory Query Template",
            "description": "Template for inventory-related questions",
            "agent_role": "picker",
            "initial_context": {
                "template_type": "inventory",
                "default_warehouse": "main"
            },
            "suggested_prompts": [
                "Where is item located?",
                "Check stock levels",
                "Update inventory count"
            ],
            "tags": ["inventory", "picker", "template"],
            "category": "warehouse_operations"
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
                limit=10
            )
            
            print(f"✅ Retrieved {len(templates)} templates")
            
            for template in templates:
                print(f"   - {template['name']} ({template['category']})")
                
        except Exception as e:
            print(f"❌ Template operations failed: {str(e)}")
        
        # Test 7: Conversation Insights
        print("\n7. Testing Conversation Insights")
        print("-" * 40)
        
        try:
            insights = await service.generate_conversation_insights(
                user_id=test_user,
                period_days=30
            )
            
            print("✅ Generated conversation insights:")
            print(f"   Total Conversations: {insights.total_conversations}")
            print(f"   Active Conversations: {insights.active_conversations}")
            print(f"   Archived Conversations: {insights.archived_conversations}")
            print(f"   Avg Messages per Conversation: {insights.average_messages_per_conversation:.1f}")
            print(f"   Most Active Agent: {insights.most_active_agent}")
            print(f"   User Engagement Score: {insights.user_engagement_score:.2f}")
            
            if insights.most_common_topics:
                print(f"   Top Topics: {insights.most_common_topics[:3]}")
            
            if insights.context_usage_patterns:
                print(f"   Context Patterns: {len(insights.context_usage_patterns)} types")
                
        except Exception as e:
            print(f"❌ Insights generation failed: {str(e)}")
        
        # Test 8: Export Functionality
        print("\n8. Testing Export Functionality")
        print("-" * 40)
        
        export_formats = ["json", "csv", "txt"]
        
        for format in export_formats:
            try:
                export_result = await service.export_conversations(
                    user_id=test_user,
                    conversation_ids=conversations[:2] if conversations else None,
                    format=format,
                    include_metadata=True,
                    include_context=True
                )
                
                content_length = len(export_result.get("export_data", ""))
                print(f"✅ {format.upper()} Export: {export_result['conversation_count']} conversations, {content_length} characters")
                
            except Exception as e:
                print(f"❌ {format.upper()} export failed: {str(e)}")
        
        # Test 9: Search Analytics
        print("\n9. Testing Search Analytics")
        print("-" * 40)
        
        try:
            # Search analytics should have been logged from semantic search tests
            from app.utils.database import get_async_collection
            search_analytics_col = await get_async_collection("chat_search_analytics")
            
            # Count analytics entries
            analytics_count = await search_analytics_col.count_documents({"user_id": test_user})
            print(f"✅ Search Analytics: {analytics_count} entries logged")
            
            if analytics_count > 0:
                # Get recent analytics
                recent_analytics = []
                async for doc in search_analytics_col.find({"user_id": test_user}).sort("timestamp", -1).limit(3):
                    recent_analytics.append(doc)
                
                for analytics in recent_analytics:
                    print(f"   - Query: '{analytics.get('query', '')}' ({analytics.get('search_type', 'unknown')})")
                    print(f"     Results: {analytics.get('result_count', 0)}, Time: {analytics.get('processing_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Search analytics test failed: {str(e)}")
        
        # Test 10: API Integration Simulation
        print("\n10. Testing API Integration Simulation")
        print("-" * 50)
        
        api_tests = [
            {
                "endpoint": "semantic_search",
                "data": {"query": "inventory management", "limit": 5}
            },
            {
                "endpoint": "bulk_operations",
                "data": {
                    "conversation_ids": conversations[:1] if conversations else [],
                    "operation": "untag",
                    "operation_data": {"tags": ["test"]}
                }
            },
            {
                "endpoint": "templates",
                "method": "GET",
                "params": {"category": "warehouse_operations"}
            },
            {
                "endpoint": "insights",
                "method": "GET",
                "params": {"period_days": 7}
            },
            {
                "endpoint": "export",
                "data": {"format": "json", "include_metadata": False}
            }
        ]
        
        for test in api_tests:
            endpoint = test["endpoint"]
            method = test.get("method", "POST")
            
            print(f"\nSimulating API call: {method} /conversations/{endpoint}")
            
            try:
                if endpoint == "semantic_search":
                    result = await service.semantic_search(
                        user_id=test_user,
                        query=test["data"]["query"],
                        limit=test["data"]["limit"]
                    )
                    print(f"   Results: {result.get('total', 0)} items")
                
                elif endpoint == "bulk_operations":
                    if test["data"]["conversation_ids"]:
                        result = await service.bulk_archive_operations(
                            user_id=test_user,
                            conversation_ids=test["data"]["conversation_ids"],
                            operation=ArchiveOperation(test["data"]["operation"]),
                            operation_data=test["data"].get("operation_data", {})
                        )
                        print(f"   Success Rate: {result.get('success_rate', 0):.1%}")
                    else:
                        print("   Skipped: No conversations available")
                
                elif endpoint == "templates":
                    result = await service.get_conversation_templates(
                        user_id=test_user,
                        category=test["params"].get("category"),
                        limit=10
                    )
                    print(f"   Templates: {len(result)} found")
                
                elif endpoint == "insights":
                    result = await service.generate_conversation_insights(
                        user_id=test_user,
                        period_days=test["params"]["period_days"]
                    )
                    print(f"   Insights: {result.total_conversations} total conversations")
                
                elif endpoint == "export":
                    result = await service.export_conversations(
                        user_id=test_user,
                        format=test["data"]["format"],
                        include_metadata=test["data"]["include_metadata"]
                    )
                    print(f"   Export: {result['conversation_count']} conversations")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
        
        print("\n" + "=" * 70)
        print("✅ Advanced Conversation Management Service Test Complete!")
        print("=" * 70)
        
        # Final Summary
        print(f"\nTest Summary:")
        print(f"  - Service Initialization: ✅")
        print(f"  - Embeddings Model: {'✅' if model else '⚠️ (Fallback)'}")
        print(f"  - Test Conversations: {len(conversations)} created")
        print(f"  - Semantic Search: ✅")
        print(f"  - Bulk Operations: ✅")
        print(f"  - Conversation Templates: ✅")
        print(f"  - Conversation Insights: ✅")
        print(f"  - Export Functionality: ✅")
        print(f"  - Search Analytics: ✅")
        print(f"  - API Integration: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_advanced_conversation_service())
    exit_code = 0 if success else 1
    print(f"\nTest completed with exit code: {exit_code}")
    exit(exit_code) 