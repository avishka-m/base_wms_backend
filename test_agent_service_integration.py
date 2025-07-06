#!/usr/bin/env python3
"""
Test script for enhanced agent service integration with role-based selection and management.
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chatbot.agent_service import AgentService
from app.agents.base_agent import BaseAgent

async def test_agent_service_integration():
    """Test the enhanced agent service with role-based selection."""
    print("🚀 Testing Enhanced Agent Service Integration")
    print("=" * 60)
    
    # Initialize the service
    agent_service = AgentService()
    
    # Test 1: System Status
    print("\n1. Testing System Status")
    print("-" * 30)
    status = agent_service.get_system_status()
    print(f"Total Agents: {status['total_agents']}")
    print(f"Active Agents: {status['active_agents']}")
    print(f"Initialization Time: {status['initialization_time']}")
    
    # Test 2: Agent Capabilities
    print("\n2. Testing Agent Capabilities")
    print("-" * 30)
    for role in agent_service.get_available_roles():
        capabilities = agent_service.get_agent_capabilities(role)
        print(f"\n{role.upper()} Agent Capabilities:")
        for cap in capabilities:
            print(f"  - {cap['name']}: {cap['description']}")
            print(f"    Query Types: {cap['query_types']}")
            print(f"    Priority: {cap['priority']}")
    
    # Test 3: Query Classification
    print("\n3. Testing Query Classification")
    print("-" * 30)
    test_queries = [
        "Where can I find item SKU-12345?",
        "What's the status of order #1001?",
        "How do I optimize my picking route?",
        "Pack order #1002 for shipping",
        "Check vehicle maintenance schedule",
        "Generate weekly performance report",
        "Process return for damaged goods"
    ]
    
    for query in test_queries:
        classifications = agent_service.classify_query(query)
        print(f"\nQuery: '{query}'")
        print("Classifications:")
        for query_type, confidence in classifications[:3]:  # Top 3
            print(f"  - {query_type.value}: {confidence:.2f}")
    
    # Test 4: Agent Selection
    print("\n4. Testing Agent Selection")
    print("-" * 30)
    
    test_scenarios = [
        {"query": "Find inventory for SKU-12345", "user_role": "picker"},
        {"query": "Generate analytics report", "user_role": "manager"},
        {"query": "Pack order for shipping", "user_role": "packer"},
        {"query": "Check vehicle status", "user_role": "driver"},
        {"query": "Process return", "user_role": "clerk"},
        {"query": "Where is the inventory located?", "user_role": None}  # Auto-select
    ]
    
    for scenario in test_scenarios:
        query = scenario["query"]
        user_role = scenario["user_role"]
        
        # Get suitable agents
        suitable_agents = agent_service.get_suitable_agents(query, user_role)
        best_agent = agent_service.select_best_agent(query, user_role)
        
        print(f"\nQuery: '{query}' (User: {user_role or 'Auto-select'})")
        print(f"Best Agent: {best_agent}")
        print("Suitable Agents:")
        for agent_role, score in suitable_agents[:3]:  # Top 3
            print(f"  - {agent_role}: {score:.2f}")
    
    # Test 5: Message Processing
    print("\n5. Testing Message Processing")
    print("-" * 30)
    
    test_messages = [
        {"message": "Show me inventory for electronics", "user_role": "picker"},
        {"message": "Create analytics report", "user_role": "manager"},
        {"message": "What orders need packing?", "user_role": "packer"}
    ]
    
    for test_msg in test_messages:
        try:
            print(f"\nProcessing: '{test_msg['message']}' (User: {test_msg['user_role']})")
            
            result = await agent_service.process_message(
                message=test_msg["message"],
                user_role=test_msg["user_role"],
                auto_select=True
            )
            
            print(f"Response: {result['response'][:100]}...")
            print(f"Agent Used: {result['metadata']['agent_role']}")
            print(f"Processing Time: {result['metadata']['processing_time']:.2f}s")
            print(f"Auto-Selected: {result['metadata']['auto_selected']}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    # Test 6: Performance Metrics
    print("\n6. Testing Performance Metrics")
    print("-" * 30)
    
    for role in agent_service.get_available_roles():
        performance = agent_service.get_agent_performance(role)
        print(f"\n{role.upper()} Performance:")
        print(f"  Total Queries: {performance['total_queries']}")
        print(f"  Success Rate: {performance['success_rate']:.1f}%")
        print(f"  Avg Response Time: {performance['avg_response_time']:.2f}s")
        print(f"  Last Used: {performance['last_used'] or 'Never'}")
    
    # Test 7: User Preferences
    print("\n7. Testing User Preferences")
    print("-" * 30)
    
    # Simulate feedback
    agent_service.update_user_preferences("picker", "picker", True)
    agent_service.update_user_preferences("picker", "manager", False)
    agent_service.update_user_preferences("manager", "manager", True)
    
    print("Updated user preferences (simulated feedback)")
    
    # Test preference impact
    query = "Show me inventory levels"
    suitable_without_pref = agent_service.get_suitable_agents(query, None)
    suitable_with_pref = agent_service.get_suitable_agents(query, "picker")
    
    print(f"\nQuery: '{query}'")
    print("Without user preferences (auto-select):")
    for agent_role, score in suitable_without_pref[:3]:
        print(f"  - {agent_role}: {score:.2f}")
    
    print("With picker preferences:")
    for agent_role, score in suitable_with_pref[:3]:
        print(f"  - {agent_role}: {score:.2f}")
    
    # Test 8: API Integration Simulation
    print("\n8. Testing API Integration Simulation")
    print("-" * 30)
    
    # Simulate API calls
    api_tests = [
        {"endpoint": "capabilities", "params": {"role": "manager"}},
        {"endpoint": "performance", "params": {"role": None}},
        {"endpoint": "select", "params": {"query": "Process urgent order", "user_role": "manager"}},
        {"endpoint": "feedback", "params": {"agent_role": "picker", "positive_feedback": True}}
    ]
    
    for test in api_tests:
        endpoint = test["endpoint"]
        params = test["params"]
        
        print(f"\nSimulating API call: {endpoint}")
        print(f"Parameters: {params}")
        
        if endpoint == "capabilities":
            if params["role"]:
                result = agent_service.get_agent_capabilities(params["role"])
                print(f"Capabilities for {params['role']}: {len(result)} items")
            else:
                result = {role: agent_service.get_agent_capabilities(role) for role in agent_service.get_available_roles()}
                print(f"All capabilities: {len(result)} agents")
        
        elif endpoint == "performance":
            if params["role"]:
                result = agent_service.get_agent_performance(params["role"])
                print(f"Performance for {params['role']}: {result['success_rate']:.1f}% success rate")
            else:
                result = {role: agent_service.get_agent_performance(role) for role in agent_service.agents.keys()}
                print(f"All performance: {len(result)} agents")
        
        elif endpoint == "select":
            best_agent = agent_service.select_best_agent(params["query"], params["user_role"])
            suitable_agents = agent_service.get_suitable_agents(params["query"], params["user_role"])
            print(f"Best agent for '{params['query']}': {best_agent}")
            print(f"Suitable agents: {len(suitable_agents)}")
        
        elif endpoint == "feedback":
            agent_service.update_user_preferences("test_user", params["agent_role"], params["positive_feedback"])
            print(f"Feedback processed for {params['agent_role']}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Agent Service Integration Test Complete!")
    print("=" * 60)
    
    # Summary
    final_status = agent_service.get_system_status()
    print(f"\nFinal Status:")
    print(f"  - Total Agents: {final_status['total_agents']}")
    print(f"  - Active Agents: {final_status['active_agents']}")
    print(f"  - Total Queries Processed: {final_status['total_queries_processed']}")
    print(f"  - Capability Coverage: {len(final_status['capability_coverage'])} query types")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_agent_service_integration()) 