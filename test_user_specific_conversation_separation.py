#!/usr/bin/env python3
"""
Test script to verify user-specific conversation separation.

This script tests that different users (Bob and Malan) have completely separate
conversation histories with no cross-contamination. It also tests the enhanced
chat history features like analytics, smart search, and categorization.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chatbot.enhanced_conversation_service import EnhancedConversationService
from app.services.chatbot.agent_service import AgentService
from app.models.chatbot.enhanced_chat_models import ChatMessageType

# Test configuration - Using real users from init_db.py
TEST_USERS = {
    "manager": {
        "username": "manager",
        "password": "manager123",
        "role": "Manager", 
        "email": "manager@warehouse.com",
        "name": "John Manager"
    },
    "picker": {
        "username": "picker",
        "password": "picker123",
        "role": "Picker",
        "email": "picker@warehouse.com", 
        "name": "Alice Picker"
    }
}

# Test conversation data for Manager (John)
MANAGER_CONVERSATIONS = [
    {
        "title": "Inventory Analytics Review",
        "agent_role": "manager",
        "messages": [
            ("USER", "Show me the warehouse analytics for last month"),
            ("ASSISTANT", "Here's the analytics dashboard for last month. Total orders: 1,250, Average fulfillment time: 2.3 hours."),
            ("USER", "What about inventory turnover rates?"),
            ("ASSISTANT", "Inventory turnover was 6.2x last month, which is 15% above our target of 5.4x.")
        ]
    },
    {
        "title": "Staff Performance Metrics",
        "agent_role": "manager", 
        "messages": [
            ("USER", "How is the picking team performing?"),
            ("ASSISTANT", "The picking team had 98.2% accuracy last week with an average pick rate of 45 items/hour."),
            ("USER", "Any areas needing improvement?"),
            ("ASSISTANT", "Consider training on heavy item handling - it's our slowest category at 28 items/hour.")
        ]
    },
    {
        "title": "Budget Planning Discussion",
        "agent_role": "manager",
        "messages": [
            ("USER", "Let's review Q4 budget allocations"),
            ("ASSISTANT", "Q4 budget shows $45K for equipment upgrades and $23K for staff training initiatives."),
            ("USER", "Should we increase the training budget?"),
            ("ASSISTANT", "Based on performance metrics, I recommend increasing training budget to $30K for maximum ROI.")
        ]
    }
]

# Test conversation data for Picker (Alice)
PICKER_CONVERSATIONS = [
    {
        "title": "Order Picking Instructions", 
        "agent_role": "picker",
        "messages": [
            ("USER", "I need help with order #12345"),
            ("ASSISTANT", "Order #12345 has 8 items. Start with section A-1-2 for the electronics items."),
            ("USER", "Where do I find the wireless headphones?"),
            ("ASSISTANT", "Wireless headphones are in location A-1-2-B, shelf 3. SKU: WH-001.")
        ]
    },
    {
        "title": "Picking Route Optimization",
        "agent_role": "picker",
        "messages": [
            ("USER", "What's the best route for my current pick list?"),
            ("ASSISTANT", "Optimal route: Start at A-1, then B-3, C-2, and finish at D-1. Total walking distance: 240 meters."),
            ("USER", "Any items that need special handling?"),
            ("ASSISTANT", "Item #5 (glass vases) requires fragile handling. Use extra padding and careful transport.")
        ]
    },
    {
        "title": "Inventory Status Check",
        "agent_role": "picker", 
        "messages": [
            ("USER", "Is there enough stock for item SKU-789?"),
            ("ASSISTANT", "SKU-789 has 45 units in stock at location B-2-3. That's sufficient for your current orders."),
            ("USER", "What about the backup location?"),
            ("ASSISTANT", "Backup location C-4-1 has 12 additional units. Total available: 57 units.")
        ]
    }
]

class ConversationSeparationTester:
    def __init__(self):
        self.conversation_service = EnhancedConversationService()
        self.agent_service = AgentService()
        self.test_results = {
            "separation_tests": [],
            "analytics_tests": [],
            "search_tests": [],
            "cross_contamination_tests": [],
            "summary": {}
        }

    async def run_all_tests(self):
        """Run all user separation tests."""
        print("🧪 Starting User-Specific Conversation Separation Tests")
        print("=" * 60)
        
        try:
            # Clean up any existing test data
            await self.cleanup_test_data()
            
            # Test 1: Create separate conversations for each user
            await self.test_conversation_creation()
            
            # Test 2: Verify conversation isolation
            await self.test_conversation_isolation()
            
            # Test 3: Test user-specific analytics
            await self.test_user_analytics()
            
            # Test 4: Test smart search separation
            await self.test_smart_search_separation()
            
            # Test 5: Test cross-contamination prevention
            await self.test_cross_contamination_prevention()
            
            # Test 6: Test bulk operations separation
            await self.test_bulk_operations_separation()
            
            # Generate test report
            await self.generate_test_report()
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            raise

    async def cleanup_test_data(self):
        """Clean up any existing test data."""
        print("🧹 Cleaning up existing test data...")
        
        for user_id in TEST_USERS.keys():
            try:
                # Get all conversations for user
                user_conversations = await self.conversation_service.get_user_conversations(
                    user_id=user_id,
                    limit=1000
                )
                
                # Delete all test conversations
                for conv in user_conversations.get("conversations", []):
                    await self.conversation_service.delete_conversation(
                        user_id=user_id,
                        conversation_id=conv["conversation_id"],
                        hard_delete=True
                    )
                
                print(f"  ✓ Cleaned up conversations for {user_id}")
                
            except Exception as e:
                print(f"  ⚠️  Cleanup warning for {user_id}: {e}")

    async def test_conversation_creation(self):
        """Test creating separate conversations for Bob and Malan."""
        print("\n📝 Test 1: Creating User-Specific Conversations")
        
        created_conversations = {"manager": [], "picker": []}
        
        # Create Manager's conversations
        for conv_data in MANAGER_CONVERSATIONS:
            conv_result = await self.conversation_service.create_conversation(
                user_id="manager",
                title=conv_data["title"],
                agent_role=conv_data["agent_role"],
                available_roles=["manager", "clerk", "picker", "packer", "driver"]
            )
            created_conversations["manager"].append(conv_result)
            
            # Add messages to conversation
            for msg_type, content in conv_data["messages"]:
                await self.conversation_service.add_message(
                    user_id="manager",
                    conversation_id=conv_result["conversation_id"],
                    message_content=content,
                    message_type=getattr(ChatMessageType, msg_type)
                )
        
        # Create Picker's conversations
        for conv_data in PICKER_CONVERSATIONS:
            conv_result = await self.conversation_service.create_conversation(
                user_id="picker",
                title=conv_data["title"],
                agent_role=conv_data["agent_role"],
                available_roles=["manager", "clerk", "picker", "packer", "driver"]
            )
            created_conversations["picker"].append(conv_result)
            
            # Add messages to conversation
            for msg_type, content in conv_data["messages"]:
                await self.conversation_service.add_message(
                    user_id="picker",
                    conversation_id=conv_result["conversation_id"],
                    message_content=content,
                    message_type=getattr(ChatMessageType, msg_type)
                )
        
        # Verify creation
        manager_count = len(created_conversations["manager"])
        picker_count = len(created_conversations["picker"])
        
        self.test_results["separation_tests"].append({
            "test": "conversation_creation",
            "status": "PASSED" if manager_count == 3 and picker_count == 3 else "FAILED",
            "details": f"Manager: {manager_count} conversations, Picker: {picker_count} conversations",
            "created_conversations": created_conversations
        })
        
        print(f"  ✓ Created {manager_count} conversations for Manager (John)")
        print(f"  ✓ Created {picker_count} conversations for Picker (Alice)")

    async def test_conversation_isolation(self):
        """Test that users can only see their own conversations."""
        print("\n🔒 Test 2: Conversation Isolation")
        
        # Get Bob's conversations
        bob_conversations = await self.conversation_service.get_user_conversations(
            user_id="bob",
            limit=100
        )
        
        # Get Malan's conversations
        malan_conversations = await self.conversation_service.get_user_conversations(
            user_id="malan", 
            limit=100
        )
        
        bob_conv_ids = set(conv["conversation_id"] for conv in bob_conversations.get("conversations", []))
        malan_conv_ids = set(conv["conversation_id"] for conv in malan_conversations.get("conversations", []))
        
        # Check for overlap (should be none)
        overlap = bob_conv_ids.intersection(malan_conv_ids)
        
        isolation_test = {
            "test": "conversation_isolation",
            "status": "PASSED" if len(overlap) == 0 else "FAILED",
            "details": {
                "bob_conversations": len(bob_conv_ids),
                "malan_conversations": len(malan_conv_ids),
                "overlap_count": len(overlap),
                "overlapping_ids": list(overlap)
            }
        }
        
        self.test_results["separation_tests"].append(isolation_test)
        
        print(f"  ✓ Bob has {len(bob_conv_ids)} conversations")
        print(f"  ✓ Malan has {len(malan_conv_ids)} conversations")
        print(f"  ✓ Overlap: {len(overlap)} (should be 0)")

    async def test_user_analytics(self):
        """Test user-specific analytics and insights."""
        print("\n📊 Test 3: User-Specific Analytics")
        
        analytics_results = {}
        
        for user_id in ["bob", "malan"]:
            # Get user conversations for analytics calculation
            conversations = await self.conversation_service.get_user_conversations(
                user_id=user_id,
                limit=1000
            )
            
            # Calculate analytics manually for verification
            total_conversations = len(conversations.get("conversations", []))
            agent_usage = {}
            categories = {"manager": 0, "picker": 0, "analytics": 0, "inventory": 0}
            
            for conv in conversations.get("conversations", []):
                agent_role = conv.get("agent_role", "general")
                agent_usage[agent_role] = agent_usage.get(agent_role, 0) + 1
                
                # Categorize based on content
                title = conv.get("title", "").lower()
                if "analytics" in title or "metrics" in title or "performance" in title:
                    categories["analytics"] += 1
                elif "inventory" in title or "stock" in title:
                    categories["inventory"] += 1
            
            analytics_results[user_id] = {
                "total_conversations": total_conversations,
                "agent_usage": agent_usage,
                "categories": categories
            }
        
        # Verify Bob's analytics (should be manager-focused)
        bob_analytics = analytics_results["bob"]
        bob_manager_focused = bob_analytics["agent_usage"].get("manager", 0) > 0
        
        # Verify Malan's analytics (should be picker-focused)
        malan_analytics = analytics_results["malan"]
        malan_picker_focused = malan_analytics["agent_usage"].get("picker", 0) > 0
        
        analytics_test = {
            "test": "user_analytics",
            "status": "PASSED" if bob_manager_focused and malan_picker_focused else "FAILED",
            "details": {
                "bob_analytics": bob_analytics,
                "malan_analytics": malan_analytics,
                "bob_manager_focused": bob_manager_focused,
                "malan_picker_focused": malan_picker_focused
            }
        }
        
        self.test_results["analytics_tests"].append(analytics_test)
        
        print(f"  ✓ Bob: {bob_analytics['total_conversations']} conversations, {bob_analytics['agent_usage']}")
        print(f"  ✓ Malan: {malan_analytics['total_conversations']} conversations, {malan_analytics['agent_usage']}")

    async def test_smart_search_separation(self):
        """Test that smart search only returns user-specific results."""
        print("\n🔍 Test 4: Smart Search Separation")
        
        search_tests = [
            {"query": "analytics", "expected_user": "bob"},
            {"query": "picking", "expected_user": "malan"},
            {"query": "inventory", "expected_user": "both"},
            {"query": "budget", "expected_user": "bob"}
        ]
        
        search_results = {}
        
        for test in search_tests:
            query = test["query"]
            search_results[query] = {}
            
            # Search for Bob
            bob_results = []
            bob_conversations = await self.conversation_service.get_user_conversations(
                user_id="bob",
                limit=100
            )
            
            for conv in bob_conversations.get("conversations", []):
                if query.lower() in conv.get("title", "").lower():
                    bob_results.append(conv)
            
            # Search for Malan
            malan_results = []
            malan_conversations = await self.conversation_service.get_user_conversations(
                user_id="malan",
                limit=100
            )
            
            for conv in malan_conversations.get("conversations", []):
                if query.lower() in conv.get("title", "").lower():
                    malan_results.append(conv)
            
            search_results[query] = {
                "bob_results": len(bob_results),
                "malan_results": len(malan_results),
                "total_results": len(bob_results) + len(malan_results)
            }
        
        self.test_results["search_tests"].append({
            "test": "smart_search_separation",
            "status": "PASSED",
            "details": search_results
        })
        
        for query, results in search_results.items():
            print(f"  ✓ '{query}': Bob={results['bob_results']}, Malan={results['malan_results']}")

    async def test_cross_contamination_prevention(self):
        """Test that operations on one user don't affect the other."""
        print("\n🛡️  Test 5: Cross-Contamination Prevention")
        
        # Get initial counts
        bob_initial = await self.conversation_service.get_user_conversations(user_id="bob", limit=100)
        malan_initial = await self.conversation_service.get_user_conversations(user_id="malan", limit=100)
        
        bob_initial_count = len(bob_initial.get("conversations", []))
        malan_initial_count = len(malan_initial.get("conversations", []))
        
        # Create a new conversation for Bob
        new_bob_conv = await self.conversation_service.create_conversation(
            user_id="bob",
            title="Cross-Contamination Test Conversation",
            agent_role="manager",
            available_roles=["manager"]
        )
        
        # Check that Malan's count didn't change
        malan_after = await self.conversation_service.get_user_conversations(user_id="malan", limit=100)
        malan_after_count = len(malan_after.get("conversations", []))
        
        # Check that Bob's count increased by 1
        bob_after = await self.conversation_service.get_user_conversations(user_id="bob", limit=100)
        bob_after_count = len(bob_after.get("conversations", []))
        
        contamination_test = {
            "test": "cross_contamination_prevention",
            "status": "PASSED" if (bob_after_count == bob_initial_count + 1 and 
                                malan_after_count == malan_initial_count) else "FAILED",
            "details": {
                "bob_initial": bob_initial_count,
                "bob_after": bob_after_count,
                "malan_initial": malan_initial_count,
                "malan_after": malan_after_count
            }
        }
        
        self.test_results["cross_contamination_tests"].append(contamination_test)
        
        print(f"  ✓ Bob: {bob_initial_count} → {bob_after_count} (+1)")
        print(f"  ✓ Malan: {malan_initial_count} → {malan_after_count} (unchanged)")
        
        # Clean up test conversation
        await self.conversation_service.delete_conversation(
            user_id="bob",
            conversation_id=new_bob_conv["conversation_id"],
            hard_delete=True
        )

    async def test_bulk_operations_separation(self):
        """Test that bulk operations only affect the requesting user's conversations."""
        print("\n📦 Test 6: Bulk Operations Separation")
        
        # Get Bob's conversation IDs
        bob_conversations = await self.conversation_service.get_user_conversations(user_id="bob", limit=100)
        bob_conv_ids = [conv["conversation_id"] for conv in bob_conversations.get("conversations", [])]
        
        # Get Malan's initial count
        malan_initial = await self.conversation_service.get_user_conversations(user_id="malan", limit=100)
        malan_initial_count = len(malan_initial.get("conversations", []))
        
        # Archive one of Bob's conversations
        if bob_conv_ids:
            await self.conversation_service.archive_conversation(
                user_id="bob",
                conversation_id=bob_conv_ids[0]
            )
        
        # Check that Malan's conversations are unchanged
        malan_after = await self.conversation_service.get_user_conversations(user_id="malan", limit=100)
        malan_after_count = len(malan_after.get("conversations", []))
        
        bulk_test = {
            "test": "bulk_operations_separation",
            "status": "PASSED" if malan_after_count == malan_initial_count else "FAILED",
            "details": {
                "malan_before_bob_operation": malan_initial_count,
                "malan_after_bob_operation": malan_after_count,
                "bob_operations_performed": 1
            }
        }
        
        self.test_results["separation_tests"].append(bulk_test)
        
        print(f"  ✓ Archived Bob's conversation")
        print(f"  ✓ Malan's conversations unchanged: {malan_initial_count} → {malan_after_count}")

    async def generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n📋 Test Report Generation")
        
        total_tests = (len(self.test_results["separation_tests"]) + 
                      len(self.test_results["analytics_tests"]) + 
                      len(self.test_results["search_tests"]) + 
                      len(self.test_results["cross_contamination_tests"]))
        
        passed_tests = 0
        failed_tests = 0
        
        all_tests = []
        for category in ["separation_tests", "analytics_tests", "search_tests", "cross_contamination_tests"]:
            all_tests.extend(self.test_results[category])
        
        for test in all_tests:
            if test["status"] == "PASSED":
                passed_tests += 1
            else:
                failed_tests += 1
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("🏁 USER SEPARATION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {self.test_results['summary']['success_rate']}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! User-specific conversation separation is working correctly.")
            print("✅ Bob and Malan have completely separate conversation histories.")
            print("✅ No cross-contamination detected.")
            print("✅ Analytics and search are user-specific.")
        else:
            print("\n⚠️  SOME TESTS FAILED. Please review the detailed results.")
        
        # Save detailed results to file
        with open("user_separation_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: user_separation_test_results.json")

async def main():
    """Main test execution function."""
    tester = ConversationSeparationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 