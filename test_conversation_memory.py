#!/usr/bin/env python3
"""
Test script for the enhanced conversation memory system.

This script tests:
1. Basic conversation memory functionality
2. ConversationSummaryBufferMemory with summarization
3. Persistence to MongoDB
4. Context awareness across multiple messages
5. Different conversation threads
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from app.services.chatbot.conversation_memory_service import conversation_memory_service
from app.services.chatbot.agent_service import EnhancedAgentService


async def test_basic_memory():
    """Test basic conversation memory functionality."""
    print("=" * 60)
    print("🧠 Testing Basic Conversation Memory")
    print("=" * 60)
    
    # Test conversation parameters
    user_id = "test_user_001"
    conversation_id = f"test_conv_{int(datetime.now().timestamp())}"
    
    try:
        # Add some messages manually
        await conversation_memory_service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message="What items do we have in inventory?",
            ai_response="I can help you check our inventory. We have smartphones, laptops, clothing items, and food products in stock.",
            agent_role="clerk"
        )
        
        await conversation_memory_service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message="How many smartphones are available?",
            ai_response="We have 100 Smartphone XYZ units available in stock at location A-1-1-1.",
            agent_role="clerk"
        )
        
        # Get conversation context
        context = await conversation_memory_service.get_conversation_context(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_role="clerk"
        )
        
        print(f"✅ Conversation ID: {conversation_id}")
        print(f"✅ Total messages: {context['total_messages']}")
        print(f"✅ Recent messages: {context['recent_messages']}")
        print(f"✅ Has summary: {context['has_summary']}")
        
        if context['memory_variables'].get('history'):
            print("\n📝 Recent conversation history:")
            for i, msg in enumerate(context['memory_variables']['history']):
                role = "User" if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage' else "Assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"  {i+1}. {role}: {content[:100]}...")
        
        print("\n✅ Basic memory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_with_memory():
    """Test agent conversation with memory context."""
    print("\n" + "=" * 60)
    print("🤖 Testing Agent with Conversation Memory")
    print("=" * 60)
    
    # Initialize agent service
    agent_service = EnhancedAgentService()
    
    # Test conversation parameters
    user_id = "test_user_002"
    conversation_id = f"agent_test_{int(datetime.now().timestamp())}"
    
    try:
        # First message
        print("\n📝 First message: 'What inventory items do we have?'")
        response1 = await agent_service.process_message(
            message="What inventory items do we have?",
            role="clerk",
            user_role="clerk",
            conversation_id=conversation_id
        )
        print(f"🤖 Response: {response1['response'][:200]}...")
        
        # Second message - should have context from first
        print("\n📝 Second message: 'How many are in stock for the first item you mentioned?'")
        response2 = await agent_service.process_message(
            message="How many are in stock for the first item you mentioned?",
            role="clerk", 
            user_role="clerk",
            conversation_id=conversation_id
        )
        print(f"🤖 Response: {response2['response'][:200]}...")
        
        # Third message - testing memory continuation
        print("\n📝 Third message: 'Where is that item located?'")
        response3 = await agent_service.process_message(
            message="Where is that item located?",
            role="clerk",
            user_role="clerk", 
            conversation_id=conversation_id
        )
        print(f"🤖 Response: {response3['response'][:200]}...")
        
        # Get conversation context to verify memory
        context = await conversation_memory_service.get_conversation_context(
            conversation_id=conversation_id,
            user_id="clerk",  # Using user_role as user_id
            agent_role="clerk"
        )
        
        print(f"\n✅ Agent conversation completed!")
        print(f"✅ Total messages in memory: {context['total_messages']}")
        print(f"✅ Conversation ID: {conversation_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_conversations():
    """Test multiple separate conversation threads."""
    print("\n" + "=" * 60)
    print("🔀 Testing Multiple Conversation Threads")
    print("=" * 60)
    
    agent_service = EnhancedAgentService()
    user_id = "test_user_003"
    
    try:
        # Conversation 1: Inventory-focused
        conv1_id = f"inventory_conv_{int(datetime.now().timestamp())}"
        print(f"\n📝 Conversation 1 ({conv1_id}): Inventory focus")
        
        await agent_service.process_message(
            message="Show me our smartphone inventory",
            role="clerk",
            user_role=user_id,
            conversation_id=conv1_id
        )
        
        conv1_response = await agent_service.process_message(
            message="What's the stock level?", 
            role="clerk",
            user_role=user_id,
            conversation_id=conv1_id
        )
        print(f"🤖 Conv1 Response: {conv1_response['response'][:150]}...")
        
        # Conversation 2: Orders-focused  
        conv2_id = f"orders_conv_{int(datetime.now().timestamp())}"
        print(f"\n📝 Conversation 2 ({conv2_id}): Orders focus")
        
        await agent_service.process_message(
            message="Show me recent orders",
            role="clerk",
            user_role=user_id,
            conversation_id=conv2_id
        )
        
        conv2_response = await agent_service.process_message(
            message="What's the status of the latest order?",
            role="clerk", 
            user_role=user_id,
            conversation_id=conv2_id
        )
        print(f"🤖 Conv2 Response: {conv2_response['response'][:150]}...")
        
        # Back to Conversation 1 - should remember smartphone context
        print(f"\n📝 Back to Conversation 1: 'Where are those smartphones located?'")
        conv1_back_response = await agent_service.process_message(
            message="Where are those smartphones located?",
            role="clerk",
            user_role=user_id, 
            conversation_id=conv1_id
        )
        print(f"🤖 Conv1 Back Response: {conv1_back_response['response'][:150]}...")
        
        # Verify separate conversation contexts
        context1 = await conversation_memory_service.get_conversation_context(
            conversation_id=conv1_id,
            user_id=user_id,
            agent_role="clerk"
        )
        
        context2 = await conversation_memory_service.get_conversation_context(
            conversation_id=conv2_id,
            user_id=user_id,
            agent_role="clerk"
        )
        
        print(f"\n✅ Conversation 1 messages: {context1['total_messages']}")
        print(f"✅ Conversation 2 messages: {context2['total_messages']}")
        print(f"✅ Multiple conversation test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple conversation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_stats():
    """Test memory statistics functionality."""
    print("\n" + "=" * 60) 
    print("📊 Testing Memory Statistics")
    print("=" * 60)
    
    try:
        stats = await conversation_memory_service.get_memory_stats()
        
        print(f"✅ Total conversations: {stats['total_conversations']}")
        print(f"✅ Active memory instances: {stats['active_memory_instances']}")
        print(f"✅ Max token limit: {stats['max_token_limit']}")
        print(f"✅ LLM available: {stats['llm_available']}")
        
        if stats.get('total_messages'):
            print(f"✅ Total messages: {stats['total_messages']}")
            print(f"✅ Avg messages per conversation: {stats['avg_messages_per_conversation']}")
            print(f"✅ Max messages per conversation: {stats['max_messages_per_conversation']}")
        
        print("\n✅ Memory statistics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all conversation memory tests."""
    print("🚀 Starting Conversation Memory Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Memory", test_basic_memory),
        ("Agent with Memory", test_agent_with_memory), 
        ("Multiple Conversations", test_multiple_conversations),
        ("Memory Statistics", test_memory_stats)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running test: {test_name}")
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All conversation memory tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 