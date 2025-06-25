"""
WMS Agent Workflow Visualizer
Generates visual graphs showing how agents interact in the warehouse management system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WMSWorkflowVisualizer:
    def __init__(self):
        self.agents = {
            'Clerk': {
                'role': 'Receiving Clerk',
                'color': '#FF6B6B',
                'tools': [
                    'inventory_query_tool', 'inventory_add_tool', 'inventory_update_tool',
                    'locate_item_tool', 'check_order_tool', 'create_sub_order_tool',
                    'process_return_tool', 'check_supplier_tool'
                ],
                'responsibilities': [
                    'Receive new inventory', 'Process returns', 'Check inventory levels',
                    'Add items to inventory', 'Verify supplier information'
                ]
            },
            'Picker': {
                'role': 'Order Picker',
                'color': '#4ECDC4',
                'tools': [
                    'locate_item_tool', 'check_order_tool', 'create_picking_task_tool',
                    'update_picking_task_tool', 'path_optimize_tool'
                ],
                'responsibilities': [
                    'Optimize picking routes', 'Locate items in warehouse',
                    'Create picking tasks', 'Update picking status'
                ]
            },
            'Packer': {
                'role': 'Order Packer',
                'color': '#45B7D1',
                'tools': [
                    'locate_item_tool', 'check_order_tool', 'create_packing_task_tool',
                    'update_packing_task_tool'
                ],
                'responsibilities': [
                    'Verify order completeness', 'Create packing tasks',
                    'Update packing status', 'Package optimization'
                ]
            },
            'Driver': {
                'role': 'Delivery Driver',
                'color': '#96CEB4',
                'tools': [
                    'check_order_tool', 'calculate_route_tool', 'vehicle_select_tool'
                ],
                'responsibilities': [
                    'Route optimization', 'Vehicle selection',
                    'Delivery management', 'Update shipping status'
                ]
            },
            'Manager': {
                'role': 'Warehouse Manager',
                'color': '#FFEAA7',
                'tools': [
                    'inventory_query_tool', 'inventory_update_tool', 'check_order_tool',
                    'approve_orders_tool', 'worker_manage_tool', 'check_analytics_tool',
                    'system_manage_tool', 'check_anomalies_tool'
                ],
                'responsibilities': [
                    'Oversee all operations', 'Approve orders', 'Manage workers',
                    'Analytics and reporting', 'System management', 'Anomaly detection'
                ]
            }
        }
        
        self.workflow_steps = [
            {'from': 'Customer', 'to': 'Manager', 'action': 'Order Approval', 'type': 'approval'},
            {'from': 'Manager', 'to': 'Picker', 'action': 'Picking Assignment', 'type': 'task_assignment'},
            {'from': 'Picker', 'to': 'Packer', 'action': 'Items Picked', 'type': 'workflow'},
            {'from': 'Packer', 'to': 'Driver', 'action': 'Order Packed', 'type': 'workflow'},
            {'from': 'Driver', 'to': 'Customer', 'action': 'Order Delivered', 'type': 'delivery'},
            {'from': 'Customer', 'to': 'Clerk', 'action': 'Returns Processing', 'type': 'return'},
            {'from': 'Supplier', 'to': 'Clerk', 'action': 'Inventory Receiving', 'type': 'receiving'},
            {'from': 'Manager', 'to': 'All Agents', 'action': 'Analytics & Monitoring', 'type': 'monitoring'}
        ]

    def create_agent_overview_graph(self):
        """Create an overview graph showing all agents and their relationships"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes for agents
        for agent_name, agent_info in self.agents.items():
            G.add_node(agent_name, **agent_info)
        
        # Add external entities
        external_entities = ['Customer', 'Supplier', 'WMS Database', 'Knowledge Base']
        for entity in external_entities:
            G.add_node(entity, color='#DDA0DD', role='External Entity')
        
        # Add edges based on workflow
        workflow_edges = [
            ('Customer', 'Manager'),
            ('Manager', 'Picker'),
            ('Picker', 'Packer'),
            ('Packer', 'Driver'),
            ('Driver', 'Customer'),
            ('Customer', 'Clerk'),
            ('Supplier', 'Clerk'),
            ('Manager', 'Clerk'),
            ('Manager', 'Picker'),
            ('Manager', 'Packer'),
            ('Manager', 'Driver'),
        ]
        
        for edge in workflow_edges:
            G.add_edge(edge[0], edge[1])
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes
        for node in G.nodes():
            color = G.nodes[node].get('color', '#DDA0DD')
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=color, node_size=3000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, 
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # Add labels
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        ax.set_title('WMS Agent Network Overview', fontsize=20, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Create legend
        legend_elements = []
        for agent_name, agent_info in self.agents.items():
            legend_elements.append(mpatches.Patch(color=agent_info['color'], label=agent_name))
        legend_elements.append(mpatches.Patch(color='#DDA0DD', label='External Entity'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return fig

    def create_detailed_workflow_graph(self):
        """Create a detailed workflow showing the complete order processing flow"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        # Define workflow stages
        stages = [
            {'name': 'Order Received', 'agent': None, 'x': 1, 'y': 8},
            {'name': 'Manager Approval', 'agent': 'Manager', 'x': 3, 'y': 8},
            {'name': 'Picking Assignment', 'agent': 'Manager', 'x': 5, 'y': 8},
            {'name': 'Route Optimization', 'agent': 'Picker', 'x': 7, 'y': 8},
            {'name': 'Item Collection', 'agent': 'Picker', 'x': 9, 'y': 8},
            {'name': 'Packing Assignment', 'agent': 'Picker', 'x': 11, 'y': 8},
            {'name': 'Order Verification', 'agent': 'Packer', 'x': 11, 'y': 6},
            {'name': 'Package Creation', 'agent': 'Packer', 'x': 9, 'y': 6},
            {'name': 'Shipping Assignment', 'agent': 'Packer', 'x': 7, 'y': 6},
            {'name': 'Vehicle Selection', 'agent': 'Driver', 'x': 5, 'y': 6},
            {'name': 'Route Planning', 'agent': 'Driver', 'x': 3, 'y': 6},
            {'name': 'Order Delivered', 'agent': 'Driver', 'x': 1, 'y': 6},
            
            # Return process
            {'name': 'Return Request', 'agent': None, 'x': 1, 'y': 4},
            {'name': 'Return Processing', 'agent': 'Clerk', 'x': 3, 'y': 4},
            {'name': 'Inventory Update', 'agent': 'Clerk', 'x': 5, 'y': 4},
            
            # Receiving process
            {'name': 'Supplier Delivery', 'agent': None, 'x': 1, 'y': 2},
            {'name': 'Item Receiving', 'agent': 'Clerk', 'x': 3, 'y': 2},
            {'name': 'Inventory Addition', 'agent': 'Clerk', 'x': 5, 'y': 2},
            {'name': 'Location Assignment', 'agent': 'Clerk', 'x': 7, 'y': 2},
        ]
        
        # Draw workflow stages
        for i, stage in enumerate(stages):
            # Determine color based on agent
            if stage['agent']:
                color = self.agents[stage['agent']]['color']
            else:
                color = '#E8E8E8'
            
            # Create box for stage
            box = FancyBboxPatch((stage['x']-0.4, stage['y']-0.3), 0.8, 0.6,
                               boxstyle="round,pad=0.1", facecolor=color, 
                               edgecolor='black', alpha=0.8)
            ax.add_patch(box)
            
            # Add text
            ax.text(stage['x'], stage['y'], stage['name'], ha='center', va='center',
                   fontsize=8, fontweight='bold', wrap=True)
        
        # Draw arrows between stages
        arrow_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), 
            (7, 8), (8, 9), (9, 10), (10, 11),
            (12, 13), (13, 14),
            (15, 16), (16, 17), (17, 18)
        ]
        
        for start_idx, end_idx in arrow_connections:
            start = stages[start_idx]
            end = stages[end_idx]
            ax.annotate('', xy=(end['x'], end['y']), xytext=(start['x'], start['y']),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(1, 9)
        ax.set_title('Detailed WMS Workflow Process', fontsize=20, fontweight='bold')
        ax.axis('off')
        
        # Add section labels
        ax.text(6, 9, 'ORDER FULFILLMENT PROCESS', ha='center', fontsize=14, 
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        ax.text(3, 5, 'RETURN PROCESS', ha='center', fontsize=12, 
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        ax.text(4, 3, 'RECEIVING PROCESS', ha='center', fontsize=12, 
               fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        plt.tight_layout()
        return fig

    def create_agent_tools_matrix(self):
        """Create a matrix showing which tools each agent has access to"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Collect all unique tools
        all_tools = set()
        for agent_info in self.agents.values():
            all_tools.update(agent_info['tools'])
        all_tools = sorted(list(all_tools))
        
        # Create matrix
        agent_names = list(self.agents.keys())
        matrix = np.zeros((len(agent_names), len(all_tools)))
        
        for i, agent_name in enumerate(agent_names):
            for j, tool in enumerate(all_tools):
                if tool in self.agents[agent_name]['tools']:
                    matrix[i][j] = 1
          # Create heatmap
        ax.imshow(matrix, cmap='RdYlGn', aspect='auto', alpha=0.8)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_tools)))
        ax.set_yticks(np.arange(len(agent_names)))
        ax.set_xticklabels([tool.replace('_tool', '').replace('_', ' ').title() for tool in all_tools], 
                          rotation=45, ha='right')
        ax.set_yticklabels(agent_names)
        
        # Add text annotations
        for i in range(len(agent_names)):
            for j in range(len(all_tools)):
                text = '✓' if matrix[i, j] else ''
                ax.text(j, i, text, ha="center", va="center", 
                       color="black", fontsize=12, fontweight='bold')
        
        ax.set_title('Agent Tools Access Matrix', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Available Tools', fontsize=14)
        ax.set_ylabel('Agents', fontsize=14)
        
        plt.tight_layout()
        return fig

    def create_system_architecture_diagram(self):
        """Create a system architecture diagram showing how agents interact with external systems"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Define system components
        components = {
            'User Interface': {'pos': (2, 8), 'color': '#FFB6C1', 'type': 'interface'},
            'API Gateway': {'pos': (2, 6), 'color': '#DDA0DD', 'type': 'gateway'},
            'Agent Router': {'pos': (2, 4), 'color': '#98FB98', 'type': 'router'},
            
            'Clerk Agent': {'pos': (6, 6), 'color': self.agents['Clerk']['color'], 'type': 'agent'},
            'Picker Agent': {'pos': (8, 6), 'color': self.agents['Picker']['color'], 'type': 'agent'},
            'Packer Agent': {'pos': (10, 6), 'color': self.agents['Packer']['color'], 'type': 'agent'},
            'Driver Agent': {'pos': (12, 6), 'color': self.agents['Driver']['color'], 'type': 'agent'},
            'Manager Agent': {'pos': (9, 8), 'color': self.agents['Manager']['color'], 'type': 'agent'},
            
            'WMS Database': {'pos': (6, 2), 'color': '#F0E68C', 'type': 'database'},
            'Knowledge Base': {'pos': (8, 2), 'color': '#F0E68C', 'type': 'database'},
            'Vector DB': {'pos': (10, 2), 'color': '#F0E68C', 'type': 'database'},
            'OpenAI API': {'pos': (12, 2), 'color': '#87CEEB', 'type': 'external'},
            
            'Inventory Tools': {'pos': (6, 4), 'color': '#FFA07A', 'type': 'tools'},
            'Order Tools': {'pos': (8, 4), 'color': '#FFA07A', 'type': 'tools'},
            'Path Tools': {'pos': (10, 4), 'color': '#FFA07A', 'type': 'tools'},
            'Warehouse Tools': {'pos': (12, 4), 'color': '#FFA07A', 'type': 'tools'},
        }
        
        # Draw components
        for comp_name, comp_info in components.items():
            x, y = comp_info['pos']
            color = comp_info['color']
            
            if comp_info['type'] == 'agent':
                # Draw agents as circles
                circle = plt.Circle((x, y), 0.5, facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(circle)
            elif comp_info['type'] == 'database':
                # Draw databases as cylinders (rectangles with rounded corners)
                rect = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(rect)
            else:
                # Draw other components as rectangles
                rect = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(rect)
            
            # Add text
            ax.text(x, y, comp_name, ha='center', va='center', fontsize=8, 
                   fontweight='bold', wrap=True)
        
        # Draw connections
        connections = [
            ('User Interface', 'API Gateway'),
            ('API Gateway', 'Agent Router'),
            ('Agent Router', 'Clerk Agent'),
            ('Agent Router', 'Picker Agent'),
            ('Agent Router', 'Packer Agent'),
            ('Agent Router', 'Driver Agent'),
            ('Agent Router', 'Manager Agent'),
            
            ('Clerk Agent', 'Inventory Tools'),
            ('Picker Agent', 'Order Tools'),
            ('Packer Agent', 'Order Tools'),
            ('Driver Agent', 'Path Tools'),
            ('Manager Agent', 'Warehouse Tools'),
            
            ('Inventory Tools', 'WMS Database'),
            ('Order Tools', 'WMS Database'),
            ('Path Tools', 'WMS Database'),
            ('Warehouse Tools', 'WMS Database'),
            
            ('Clerk Agent', 'Knowledge Base'),
            ('Picker Agent', 'Vector DB'),
            ('Manager Agent', 'OpenAI API'),
        ]
        
        for start, end in connections:
            start_pos = components[start]['pos']
            end_pos = components[end]['pos']
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_title('WMS Agent System Architecture', fontsize=18, fontweight='bold')
        ax.axis('off')
        
        # Add layer labels
        ax.text(1, 8, 'Presentation\nLayer', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        ax.text(1, 6, 'API\nLayer', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        ax.text(1, 4, 'Agent\nLayer', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
        ax.text(1, 2, 'Data\nLayer', ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        plt.tight_layout()
        return fig

    def generate_all_visualizations(self, save_path=None):
        """Generate all visualization graphs"""
        figures = {}
        
        print("Generating WMS Agent Workflow Visualizations...")
        
        # 1. Agent Overview Graph
        print("1. Creating Agent Network Overview...")
        figures['overview'] = self.create_agent_overview_graph()
        
        # 2. Detailed Workflow Graph
        print("2. Creating Detailed Workflow Process...")
        figures['workflow'] = self.create_detailed_workflow_graph()
        
        # 3. Agent Tools Matrix
        print("3. Creating Agent Tools Matrix...")
        figures['tools_matrix'] = self.create_agent_tools_matrix()
        
        # 4. System Architecture
        print("4. Creating System Architecture Diagram...")
        figures['architecture'] = self.create_system_architecture_diagram()
        
        # Save figures if path is provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/wms_{name}.png", dpi=300, bbox_inches='tight')
                print(f"Saved {name} visualization to {save_path}/wms_{name}.png")
        
        return figures

    def print_workflow_summary(self):
        """Print a text summary of the agent workflow"""
        print("\n" + "="*80)
        print("WAREHOUSE MANAGEMENT SYSTEM - AGENT WORKFLOW SUMMARY")
        print("="*80)
        print("\n🤖 AGENTS OVERVIEW:")
        print("-" * 50)
        for agent_name, agent_info in self.agents.items():
            print(f"\n{agent_name} ({agent_info['role']}):")
            print("  📋 Responsibilities:")
            for resp in agent_info['responsibilities']:
                print(f"    • {resp}")
            print(f"  🔧 Tools ({len(agent_info['tools'])}):")
            for tool in agent_info['tools']:
                print(f"    • {tool.replace('_tool', '').replace('_', ' ').title()}")
        
        print("\n\n🔄 WORKFLOW PROCESS:")
        print("-" * 50)
        print("""
1. ORDER FULFILLMENT FLOW:
   Customer → Manager (Order Approval) → Picker (Route Optimization & Collection) 
   → Packer (Verification & Packaging) → Driver (Vehicle Selection & Delivery) → Customer

2. RETURN PROCESS:
   Customer → Clerk (Return Processing & Inventory Update)

3. RECEIVING PROCESS:
   Supplier → Clerk (Item Receiving, Inventory Addition, Location Assignment)

4. MANAGEMENT OVERSIGHT:
   Manager monitors all processes, manages workers, analyzes performance, and detects anomalies
        """)
        
        print("\n\n🔗 AGENT INTERACTIONS:")
        print("-" * 50)
        print("""
• Manager Agent: Central coordination hub with oversight of all other agents
• Clerk Agent: Handles incoming/outgoing inventory independently  
• Picker Agent: Receives assignments from Manager, coordinates with Packer
• Packer Agent: Receives items from Picker, coordinates with Driver
• Driver Agent: Receives packed orders from Packer, manages delivery
        """)
        
        print("\n\n🛠️ TECHNOLOGY STACK:")
        print("-" * 50)
        print("""
• LangChain: Agent framework and tool orchestration
• OpenAI GPT: Natural language processing and decision making
• FastAPI: REST API for agent communication
• ChromaDB: Vector database for knowledge retrieval
• NetworkX: Path optimization and route planning
• MongoDB: Persistent data storage
        """)


def main():
    """Main function to generate visualizations"""
    visualizer = WMSWorkflowVisualizer()
    
    # Print workflow summary
    visualizer.print_workflow_summary()
      # Generate visualizations
    save_directory = "wms_agent_visualizations"
    visualizer.generate_all_visualizations(save_path=save_directory)
    
    # Display figures
    plt.show()
    
    print(f"\n✅ All visualizations generated and saved to '{save_directory}' directory!")
    print("\nVisualization files created:")
    print("• wms_overview.png - Agent network overview")
    print("• wms_workflow.png - Detailed workflow process")  
    print("• wms_tools_matrix.png - Agent tools access matrix")
    print("• wms_architecture.png - System architecture diagram")


if __name__ == "__main__":
    main()
