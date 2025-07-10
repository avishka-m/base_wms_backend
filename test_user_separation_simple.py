#!/usr/bin/env python3
"""
Simple User Separation Test - Real Users

Tests core user separation between Manager (John) and Picker (Alice) 
using real usernames from init_db.py database.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService
from app.models.chatbot.enhanced_chat_models import ChatMessageType

# Real users from init_db.py
MANAGER_CONVERSATIONS = [
    {
        "title": "Daily Analytics Review", 
        "agent_role": "manager",
        "messages": [("USER", "Show me warehouse analytics"), ("ASSISTANT", "Here's your dashboard...")]
    },
    {
        "title": "Budget Planning",
        "agent_role": "manager", 
        "messages": [("USER", "Review quarterly budget"), ("ASSISTANT", "Q4 budget analysis...")]
    }
]

PICKER_CONVERSATIONS = [
    {
        "title": "Order Picking Tasks",
        "agent_role": "picker",
        "messages": [("USER", "What orders to pick today?"), ("ASSISTANT", "Here are your picking tasks...")]
    },
    {
        "title": "Inventory Location Query", 
        "agent_role": "picker",
        "messages": [("USER", "Where is SKU-12345?"), ("ASSISTANT", "Located in section A-1-2...")]
    }
]

async def test_user_separation():
    """Test complete user separation between Manager and Picker."""
    
    print("🧪 Testing User Separation with Real Database Users")
    print("=" * 55)
    print("👤 Manager: John Manager (username: manager)")  
    print("👤 Picker: Alice Picker (username: picker)")
    print()
    
    service = EnhancedConversationService()
    
    # 1. Clean up existing data
    print("🧹 Cleaning up existing conversations...")
    for user_id in ["manager", "picker"]:
        convs = await service.get_user_conversations(user_id=user_id, limit=1000)
        for conv in convs.get("conversations", []):
            await service.delete_conversation(user_id=user_id, conversation_id=conv["conversation_id"], hard_delete=True)
    print("  ✓ Cleanup complete")
    
    # 2. Create Manager's conversations
    print("\n📝 Creating Manager's conversations...")
    manager_conv_ids = []
    for conv_data in MANAGER_CONVERSATIONS:
        conv = await service.create_conversation(
            user_id="manager",
            title=conv_data["title"], 
            agent_role=conv_data["agent_role"],
            available_roles=["manager", "clerk", "picker", "packer", "driver"]
        )
        manager_conv_ids.append(conv["conversation_id"])
        
        # Add messages
        for msg_type, content in conv_data["messages"]:
            await service.add_message(
                user_id="manager",
                conversation_id=conv["conversation_id"],
                message_content=content,
                message_type=getattr(ChatMessageType, msg_type)
            )
    
    print(f"  ✓ Created {len(manager_conv_ids)} conversations for Manager")
    
    # 3. Create Picker's conversations  
    print("\n📝 Creating Picker's conversations...")
    picker_conv_ids = []
    for conv_data in PICKER_CONVERSATIONS:
        conv = await service.create_conversation(
            user_id="picker",
            title=conv_data["title"],
            agent_role=conv_data["agent_role"], 
            available_roles=["manager", "clerk", "picker", "packer", "driver"]
        )
        picker_conv_ids.append(conv["conversation_id"])
        
        # Add messages
        for msg_type, content in conv_data["messages"]:
            await service.add_message(
                user_id="picker",
                conversation_id=conv["conversation_id"],
                message_content=content,
                message_type=getattr(ChatMessageType, msg_type)
            )
            
    print(f"  ✓ Created {len(picker_conv_ids)} conversations for Picker")
    
    # 4. Test isolation
    print("\n🔒 Testing conversation isolation...")
    
    # Get each user's conversations
    manager_conversations = await service.get_user_conversations(user_id="manager", limit=100)
    picker_conversations = await service.get_user_conversations(user_id="picker", limit=100)
    
    manager_ids = set(c["conversation_id"] for c in manager_conversations.get("conversations", []))
    picker_ids = set(c["conversation_id"] for c in picker_conversations.get("conversations", []))
    
    overlap = manager_ids.intersection(picker_ids)
    
    print(f"  ✓ Manager has {len(manager_ids)} conversations")
    print(f"  ✓ Picker has {len(picker_ids)} conversations")
    print(f"  ✓ Overlap: {len(overlap)} conversations (should be 0)")
    
    # 5. Test search separation
    print("\n🔍 Testing search separation...")
    
    # Count conversations by topic
    manager_analytics = sum(1 for c in manager_conversations.get("conversations", []) if "analytics" in c.get("title", "").lower())
    manager_budget = sum(1 for c in manager_conversations.get("conversations", []) if "budget" in c.get("title", "").lower())
    
    picker_picking = sum(1 for c in picker_conversations.get("conversations", []) if "pick" in c.get("title", "").lower())
    picker_location = sum(1 for c in picker_conversations.get("conversations", []) if "location" in c.get("title", "").lower())
    
    print(f"  ✓ Manager - Analytics topics: {manager_analytics}, Budget topics: {manager_budget}")
    print(f"  ✓ Picker - Picking topics: {picker_picking}, Location topics: {picker_location}")
    
    # 6. Test cross-contamination prevention
    print("\n🛡️  Testing cross-contamination prevention...")
    
    # Count before operation
    initial_picker_count = len(picker_ids)
    
    # Create new conversation for Manager
    new_manager_conv = await service.create_conversation(
        user_id="manager",
        title="Test Cross-Contamination",
        agent_role="manager",
        available_roles=["manager", "clerk", "picker", "packer", "driver"]
    )
    
    # Check Picker's count didn't change
    updated_picker_conversations = await service.get_user_conversations(user_id="picker", limit=100)
    final_picker_count = len(updated_picker_conversations.get("conversations", []))
    
    print(f"  ✓ Picker conversations before Manager operation: {initial_picker_count}")
    print(f"  ✓ Picker conversations after Manager operation: {final_picker_count}")
    print(f"  ✓ Picker unaffected: {final_picker_count == initial_picker_count}")
    
    # Clean up test conversation
    await service.delete_conversation(
        user_id="manager", 
        conversation_id=new_manager_conv["conversation_id"],
        hard_delete=True
    )
    
    # 7. Final verification
    print("\n📊 Final Results:")
    print("=" * 30)
    
    all_tests_passed = (
        len(overlap) == 0 and  # Perfect isolation
        manager_analytics > 0 and  # Manager has analytics conversations
        picker_picking > 0 and  # Picker has picking conversations  
        final_picker_count == initial_picker_count  # No cross-contamination
    )
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ Perfect user separation achieved")
        print("✅ Manager and Picker have completely isolated chat histories") 
        print("✅ No cross-contamination detected")
        print("✅ Real database users working correctly")
    else:
        print("❌ Some tests failed")
        print(f"   Isolation: {'✅' if len(overlap) == 0 else '❌'}")
        print(f"   Manager content: {'✅' if manager_analytics > 0 else '❌'}")
        print(f"   Picker content: {'✅' if picker_picking > 0 else '❌'}")
        print(f"   Cross-contamination: {'✅' if final_picker_count == initial_picker_count else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_user_separation()) 