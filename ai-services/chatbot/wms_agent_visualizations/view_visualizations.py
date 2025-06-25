"""
Simple viewer for WMS Agent Workflow Visualizations
Displays the generated visualization images
"""

import os

def display_visualizations():
    """Display all generated visualizations"""
    
    vis_dir = "wms_agent_visualizations"
    
    if not os.path.exists(vis_dir):
        print(f"Visualization directory '{vis_dir}' not found.")
        print("Please run 'python agent_workflow_visualizer.py' first to generate the visualizations.")
        return
    
    visualizations = {
        'wms_overview.png': 'Agent Network Overview',
        'wms_workflow.png': 'Detailed Workflow Process', 
        'wms_tools_matrix.png': 'Agent Tools Access Matrix',
        'wms_architecture.png': 'System Architecture Diagram'
    }
    
    print("WMS Agent Workflow Visualizations Generated:")
    print("=" * 50)
    
    for filename, title in visualizations.items():
        filepath = os.path.join(vis_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {title}")
            print(f"   📁 File: {filepath}")
            print(f"   📊 Description: {get_description(filename)}")
            print()
        else:
            print(f"❌ {title} - File not found: {filepath}")
    
    print("\nTo view the visualizations:")
    print("1. Open the files directly in an image viewer")
    print("2. Use the documentation file: WMS_Agent_Workflow_Documentation.md")
    print("3. Check the 'wms_agent_visualizations' directory")

def get_description(filename):
    """Get description for each visualization"""
    descriptions = {
        'wms_overview.png': 'Shows the network of agents and their relationships with external entities',
        'wms_workflow.png': 'Illustrates the complete order fulfillment, return, and receiving processes',
        'wms_tools_matrix.png': 'Matrix showing which tools each agent has access to',
        'wms_architecture.png': 'Technical architecture diagram showing system layers and components'
    }
    return descriptions.get(filename, 'Workflow visualization')

if __name__ == "__main__":
    display_visualizations()
