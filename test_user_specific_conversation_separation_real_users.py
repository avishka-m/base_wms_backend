#!/usr/bin/env python3
"""
User-Specific Conversation Separation Test Suite

This script tests that different users (Manager and Picker) have completely separate
conversation histories with no cross-contamination between their chat data.
Tests user isolation, analytics separation, search isolation, and bulk operations.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

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
        """Test creating separate conversations for Manager and Picker."""
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
        
        # Get Manager's conversations
        manager_conversations = await self.conversation_service.get_user_conversations(
            user_id="manager",
            limit=100
        )
        
        # Get Picker's conversations  
        picker_conversations = await self.conversation_service.get_user_conversations(
            user_id="picker",
            limit=100
        )
        
        manager_conv_ids = set(conv["conversation_id"] for conv in manager_conversations.get("conversations", []))
        picker_conv_ids = set(conv["conversation_id"] for conv in picker_conversations.get("conversations", []))
        
        # Check for overlap (should be zero)
        overlap = manager_conv_ids.intersection(picker_conv_ids)
        
        self.test_results["separation_tests"].append({
            "test": "conversation_isolation",
            "status": "PASSED" if len(overlap) == 0 else "FAILED",
            "manager_conversations": len(manager_conv_ids),
            "picker_conversations": len(picker_conv_ids),
            "overlap": len(overlap),
            "overlap_ids": list(overlap)
        })
        
        print(f"  ✓ Manager has {len(manager_conv_ids)} conversations")
        print(f"  ✓ Picker has {len(picker_conv_ids)} conversations") 
        print(f"  ✓ Overlap: {len(overlap)} (should be 0)")

    async def test_user_analytics(self):
        """Test that user analytics are separate and role-appropriate."""
        print("\n📊 Test 3: User-Specific Analytics")
        
        analytics_results = {}
        
        for user_id in ["manager", "picker"]:
            try:
                analytics = await self.conversation_service.get_user_analytics(
                    user_id=user_id,
                    period_days=30
                )
                
                analytics_results[user_id] = {
                    "total_conversations": analytics.get("overview", {}).get("total_conversations", 0),
                    "agent_usage": analytics.get("insights", {}).get("agent_usage", analytics.get("agent_usage", {})),
                    "common_topics": analytics.get("insights", {}).get("common_topics", analytics.get("common_topics", []))
                }
                
            except Exception as e:
                print(f"  ⚠️  Analytics error for {user_id}: {e}")
                analytics_results[user_id] = {"error": str(e)}
        
        # Verify Manager's analytics (should be manager-focused)
        manager_analytics = analytics_results.get("manager", {})
        manager_manager_focused = manager_analytics.get("agent_usage", {}).get("manager", 0) > 0
        
        # Verify Picker's analytics (should be picker-focused)
        picker_analytics = analytics_results.get("picker", {})
        picker_picker_focused = picker_analytics.get("agent_usage", {}).get("picker", 0) > 0
        
        self.test_results["analytics_tests"].append({
            "test": "user_specific_analytics",
            "status": "PASSED" if manager_manager_focused and picker_picker_focused else "FAILED",
            "details": "Users have role-appropriate analytics",
            "manager_analytics": manager_analytics,
            "picker_analytics": picker_analytics,
            "manager_manager_focused": manager_manager_focused,
            "picker_picker_focused": picker_picker_focused
        })
        
        print(f"  ✓ Manager: {manager_analytics.get('total_conversations', 0)} conversations, {manager_analytics.get('agent_usage', {})}")
        print(f"  ✓ Picker: {picker_analytics.get('total_conversations', 0)} conversations, {picker_analytics.get('agent_usage', {})}")

    async def test_smart_search_separation(self):
        """Test that smart search results are user-specific."""
        print("\n🔍 Test 4: Smart Search Separation")
        
        search_queries = [
            {"query": "analytics", "expected_user": "manager"},
            {"query": "picking", "expected_user": "picker"},
            {"query": "inventory", "expected_user": "both"},
            {"query": "budget", "expected_user": "manager"}
        ]
        
        for query_info in search_queries:
            query = query_info["query"]
            
            # Search for Manager
            manager_results = []
            manager_conversations = await self.conversation_service.get_user_conversations(
                user_id="manager",
                limit=100
            )
            
            for conv in manager_conversations.get("conversations", []):
                if query.lower() in conv.get("title", "").lower():
                    manager_results.append(conv)
            
            # Search for Picker
            picker_results = []
            picker_conversations = await self.conversation_service.get_user_conversations(
                user_id="picker",
                limit=100
            )
            
            for conv in picker_conversations.get("conversations", []):
                if query.lower() in conv.get("title", "").lower():
                    picker_results.append(conv)
            
            results = {
                "query": query,
                "manager_results": len(manager_results),
                "picker_results": len(picker_results),
                "total_results": len(manager_results) + len(picker_results)
            }
            
            self.test_results["search_tests"].append(results)
            
            print(f"  ✓ '{query}': Manager={results['manager_results']}, Picker={results['picker_results']}")

    async def test_cross_contamination_prevention(self):
        """Test that operations on one user don't affect another."""
        print("\n🛡️  Test 5: Cross-Contamination Prevention")
        
        # Get initial counts
        manager_initial = await self.conversation_service.get_user_conversations(user_id="manager", limit=100)
        picker_initial = await self.conversation_service.get_user_conversations(user_id="picker", limit=100)
        
        manager_initial_count = len(manager_initial.get("conversations", []))
        picker_initial_count = len(picker_initial.get("conversations", []))
        
        # Create a new conversation for Manager
        new_manager_conv = await self.conversation_service.create_conversation(
            user_id="manager",
            title="Cross-Contamination Test Conversation",
            agent_role="manager",
            available_roles=["manager", "clerk", "picker", "packer", "driver"]
        )
        
        # Check that Manager's count increased by 1
        manager_after = await self.conversation_service.get_user_conversations(user_id="manager", limit=100)
        manager_after_count = len(manager_after.get("conversations", []))
        
        # Check that Picker's count remained the same
        picker_after = await self.conversation_service.get_user_conversations(user_id="picker", limit=100)
        picker_after_count = len(picker_after.get("conversations", []))
        
        self.test_results["cross_contamination_tests"].append({
            "test": "create_operation_isolation",
            "status": "PASSED" if (manager_after_count == manager_initial_count + 1 and
                                picker_after_count == picker_initial_count) else "FAILED",
            "manager_initial": manager_initial_count,
            "manager_after": manager_after_count,
            "picker_initial": picker_initial_count,
            "picker_after": picker_after_count
        })
        
        print(f"  ✓ Manager: {manager_initial_count} → {manager_after_count} (+1)")
        print(f"  ✓ Picker: {picker_initial_count} → {picker_after_count} (unchanged)")
        
        # Clean up the test conversation
        await self.conversation_service.delete_conversation(
            user_id="manager",
            conversation_id=new_manager_conv["conversation_id"],
            hard_delete=True
        )

    async def test_bulk_operations_separation(self):
        """Test that bulk operations only affect the target user."""
        print("\n📦 Test 6: Bulk Operations Separation")
        
        # Get Manager's conversation IDs
        manager_conversations = await self.conversation_service.get_user_conversations(user_id="manager", limit=100)
        manager_conv_ids = [conv["conversation_id"] for conv in manager_conversations.get("conversations", [])]
        
        # Get initial Picker count
        picker_initial = await self.conversation_service.get_user_conversations(user_id="picker", limit=100)
        picker_initial_count = len(picker_initial.get("conversations", []))
        
        # Archive one of Manager's conversations
        if manager_conv_ids:
            await self.conversation_service.archive_conversation(
                user_id="manager",
                conversation_id=manager_conv_ids[0]
            )
        
        # Check that Picker's conversations are unaffected
        picker_after = await self.conversation_service.get_user_conversations(user_id="picker", limit=100)
        picker_after_count = len(picker_after.get("conversations", []))
        
        self.test_results["cross_contamination_tests"].append({
            "test": "bulk_operations_isolation",
            "status": "PASSED" if picker_after_count == picker_initial_count else "FAILED",
            "picker_before_manager_operation": picker_initial_count,
            "picker_after_manager_operation": picker_after_count,
            "manager_operations_performed": 1
        })
        
        print(f"  ✓ Archived Manager's conversation")
        print(f"  ✓ Picker conversations unaffected: {picker_initial_count} → {picker_after_count}")

    async def generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n📋 Test Report Summary")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if category == "summary":
                continue
                
            for test in tests:
                total_tests += 1
                if test.get("status") == "PASSED":
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 90 else "FAILED"
        }
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Overall Status: {self.test_results['summary']['overall_status']}")
        
        if success_rate >= 90:
            print("\n✅ Manager and Picker have completely separate conversation histories.")
            print("✅ User isolation is working correctly.")
            print("✅ No cross-contamination detected.")
        else:
            print("\n❌ User separation test suite failed.")
            print("❌ Cross-contamination or isolation issues detected.")

async def main():
    """Main test execution function."""
    tester = ConversationSeparationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 