#!/usr/bin/env python3
"""
Comprehensive test script to verify order tools implementation and integration.
This script checks that order management functionality is working across all agents.
"""

import sys
import os
from typing import Dict, List

# Add the backend directory to Python path
current_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def test_order_tools_implementation():
    """Test that order tools have been properly implemented."""
    
    print("🧪 Testing Order Tools Implementation")
    print("=" * 50)
    
    try:
        # Import and test order tools
        from app.tools.chatbot.order_tools import (
            order_create_func,
            order_update_func,
            approve_orders_func,
            create_sub_order_func,
            create_picking_task_func,
            update_picking_task_func,
            create_packing_task_func,
            update_packing_task_func
        )
        
        # Test that functions are no longer placeholders
        test_functions = [
            ("order_create_func", order_create_func),
            ("order_update_func", order_update_func),
            ("approve_orders_func", approve_orders_func),
            ("create_sub_order_func", create_sub_order_func),
            ("create_picking_task_func", create_picking_task_func),
            ("update_picking_task_func", update_picking_task_func),
            ("create_packing_task_func", create_packing_task_func),
            ("update_packing_task_func", update_packing_task_func)
        ]
        
        print("✅ Successfully imported all order functions")
        
        for func_name, func in test_functions:
            # Check if function has real implementation (not just a placeholder)
            import inspect
            source = inspect.getsource(func)
            if "not yet implemented" in source or "currently not implemented" in source:
                print(f"❌ {func_name} still contains placeholder implementation")
                return False
            else:
                print(f"✅ {func_name} has real implementation")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import order tools: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing order tools: {e}")
        return False

def test_mongodb_operations():
    """Test that MongoDB operations are available."""
    
    print("\n🧪 Testing MongoDB Operations")
    print("=" * 50)
    
    try:
        from app.utils.chatbot.mongodb_client import chatbot_mongodb_client
        
        # Check if new methods exist
        required_methods = [
            'create_order',
            'update_order', 
            'approve_order',
            'create_sub_order',
            'create_task',
            'get_task_by_id',
            'update_task'
        ]
        
        for method_name in required_methods:
            if hasattr(chatbot_mongodb_client, method_name):
                print(f"✅ MongoDB client has {method_name} method")
            else:
                print(f"❌ MongoDB client missing {method_name} method")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MongoDB operations: {e}")
        return False

def test_agent_tool_integration():
    """Test that agents have the correct order tools integrated."""
    
    print("\n🧪 Testing Agent Tool Integration")
    print("=" * 50)
    
    try:
        # Import agents
        from app.agents.manager_agent import ManagerAgent
        from app.agents.clerk_agent import ClerkAgent
        from app.agents.picker_agent import PickerAgent
        from app.agents.packer_agent_ex import PackerAgent
        
        # Expected tools for each agent
        expected_tools = {
            "ManagerAgent": [
                "check_order", "order_create", "order_update", "approve_orders", 
                "create_sub_order", "create_picking_task", "update_picking_task", 
                "create_packing_task", "update_packing_task"
            ],
            "ClerkAgent": [
                "check_order", "order_create", "order_update", "create_sub_order",
                "create_picking_task", "create_packing_task"
            ],
            "PickerAgent": [
                "check_order", "create_picking_task", "update_picking_task"
            ],
            "PackerAgent": [
                "check_order", "create_packing_task", "update_packing_task"
            ]
        }
        
        agents = {
            "ManagerAgent": ManagerAgent(),
            "ClerkAgent": ClerkAgent(),
            "PickerAgent": PickerAgent(),
            "PackerAgent": PackerAgent()
        }
        
        all_passed = True
        
        for agent_name, agent in agents.items():
            print(f"\n📋 Testing {agent_name}:")
            expected = expected_tools[agent_name]
            actual_tool_names = [tool.name for tool in agent.tools]
            
            found_tools = []
            missing_tools = []
            
            for expected_tool in expected:
                if expected_tool in actual_tool_names:
                    found_tools.append(expected_tool)
                    print(f"  ✅ Has {expected_tool}")
                else:
                    missing_tools.append(expected_tool)
                    print(f"  ❌ Missing {expected_tool}")
            
            if missing_tools:
                print(f"  ❌ {agent_name} missing tools: {missing_tools}")
                all_passed = False
            else:
                print(f"  ✅ {agent_name} has all required order tools ({len(found_tools)} tools)")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing agent integration: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Starting Comprehensive Order Tools Integration Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Order Tools Implementation", test_order_tools_implementation),
        ("MongoDB Operations", test_mongodb_operations),
        ("Agent Tool Integration", test_agent_tool_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Order tools are fully implemented and integrated!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 