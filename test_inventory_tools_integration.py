#!/usr/bin/env python3
"""
Test script to verify inventory tools integration with agents.
This script checks that each agent has the correct inventory tools loaded.
"""

import sys
import os
from typing import Dict, List

# Add the backend directory to Python path
current_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

def test_agent_inventory_tools():
    """Test that agents have the correct inventory tools integrated."""
    
    print("🧪 Testing Inventory Tools Integration")
    print("=" * 50)
    
    try:
        # Import agents
        from app.agents.manager_agent import ManagerAgent
        from app.agents.clerk_agent import ClerkAgent  
        from app.agents.picker_agent import PickerAgent
        from app.agents.packer_agent_ex import PackerAgent
        from app.agents.driver_agent import DriverAgent
        
        # Expected tools for each agent
        expected_tools = {
            "manager": [
                "inventory_query",
                "inventory_add", 
                "inventory_update",
                "inventory_analytics",
                "locate_item",
                "low_stock_alert",
                "stock_movement"
            ],
            "clerk": [
                "inventory_query",
                "inventory_add",
                "inventory_update", 
                "locate_item",
                "low_stock_alert",
                "stock_movement"
            ],
            "picker": [
                "inventory_query",
                "locate_item",
                "low_stock_alert"
            ],
            "packer": [
                "inventory_query",
                "locate_item"
            ],
            "driver": []  # No inventory tools expected
        }
        
        # Test each agent
        agents = {
            "manager": ManagerAgent(),
            "clerk": ClerkAgent(),
            "picker": PickerAgent(),
            "packer": PackerAgent(),
            "driver": DriverAgent()
        }
        
        results = {}
        
        for role, agent in agents.items():
            print(f"\n🔍 Testing {role.title()} Agent:")
            print("-" * 30)
            
            # Get actual tool names from agent
            actual_tool_names = [tool.name for tool in agent.tools]
            inventory_tools = [name for name in actual_tool_names if "inventory" in name or "locate_item" in name or "low_stock" in name or "stock_movement" in name]
            
            expected = expected_tools[role]
            
            print(f"Expected inventory tools: {len(expected)}")
            print(f"Actual inventory tools: {len(inventory_tools)}")
            
            # Check for expected tools
            missing_tools = []
            for expected_tool in expected:
                if expected_tool not in actual_tool_names:
                    missing_tools.append(expected_tool)
            
            # Check for unexpected tools
            unexpected_tools = []
            for tool_name in inventory_tools:
                if tool_name not in expected:
                    unexpected_tools.append(tool_name)
            
            # Report results
            if not missing_tools and not unexpected_tools:
                print("✅ All expected tools present and correct!")
                results[role] = "PASS"
            else:
                results[role] = "FAIL"
                if missing_tools:
                    print(f"❌ Missing tools: {missing_tools}")
                if unexpected_tools:
                    print(f"⚠️  Unexpected tools: {unexpected_tools}")
            
            print(f"📋 Inventory tools: {inventory_tools}")
            print(f"🔧 Total tools: {len(agent.tools)}")
        
        # Summary
        print(f"\n📊 Test Results Summary:")
        print("=" * 30)
        passed = sum(1 for result in results.values() if result == "PASS")
        total = len(results)
        
        for role, result in results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"{status} {role.title()} Agent: {result}")
        
        print(f"\n🎯 Overall: {passed}/{total} agents configured correctly")
        
        if passed == total:
            print("🎉 All agents have correct inventory tools integration!")
            return True
        else:
            print("⚠️  Some agents need tool configuration fixes.")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the backend directory.")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_inventory_tools_functionality():
    """Test that the inventory tools can be imported and have correct functions."""
    
    print(f"\n🔧 Testing Inventory Tools Functionality")
    print("=" * 40)
    
    try:
        from app.tools.chatbot.inventory_tools import (
            inventory_query_tool,
            inventory_add_tool,
            inventory_update_tool,
            inventory_analytics_tool,
            locate_item_tool,
            low_stock_alert_tool,
            stock_movement_tool
        )
        
        tools = {
            "inventory_query": inventory_query_tool,
            "inventory_add": inventory_add_tool,
            "inventory_update": inventory_update_tool,
            "inventory_analytics": inventory_analytics_tool,
            "locate_item": locate_item_tool,
            "low_stock_alert": low_stock_alert_tool,
            "stock_movement": stock_movement_tool
        }
        
        print(f"✅ Successfully imported {len(tools)} inventory tools")
        
        # Check each tool has required attributes
        for name, tool in tools.items():
            print(f"📋 {name}: {tool.description[:50]}...")
            
        print("✅ All tools have proper structure!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import inventory tools: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing tools: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Inventory Tools Integration Test")
    print("=" * 60)
    
    # Test 1: Tool functionality
    tools_ok = test_inventory_tools_functionality()
    
    # Test 2: Agent integration
    agents_ok = test_agent_inventory_tools()
    
    # Final result
    print(f"\n🏁 Final Results:")
    print("=" * 20)
    
    if tools_ok and agents_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Inventory tools are properly integrated into agents")
        print("✅ Ready for production use")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Please check the errors above and fix integration issues")
        sys.exit(1) 