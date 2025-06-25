# WMS Agent Workflow Documentation

## Overview

This warehouse management system (WMS) implements an **agentic workflow** where specialized AI agents handle different warehouse operations. Each agent has specific roles, tools, and responsibilities that work together to create an efficient warehouse management ecosystem.

## Generated Visualizations

The following visualization files have been generated in the `wms_agent_visualizations/` directory:

1. **`wms_overview.png`** - Agent Network Overview
   - Shows how all agents are connected
   - Displays the relationship between agents and external entities

2. **`wms_workflow.png`** - Detailed Workflow Process
   - Step-by-step order fulfillment flow
   - Return processing workflow
   - Receiving process workflow

3. **`wms_tools_matrix.png`** - Agent Tools Access Matrix
   - Matrix showing which tools each agent can access
   - Visual representation of agent capabilities

4. **`wms_architecture.png`** - System Architecture Diagram
   - Technical architecture showing system layers
   - Integration between agents and external systems

## Agent Roles and Responsibilities

### 🏪 **Clerk Agent (Receiving Clerk)**
- **Color Code**: Red (#FF6B6B)
- **Primary Role**: Inventory management and returns processing
- **Key Responsibilities**:
  - Receive new inventory from suppliers
  - Process customer returns
  - Check and update inventory levels
  - Verify supplier information
  - Add new items to the warehouse system

- **Available Tools** (8):
  - `inventory_query_tool` - Query inventory information
  - `inventory_add_tool` - Add new items to inventory
  - `inventory_update_tool` - Update existing inventory
  - `locate_item_tool` - Find item locations
  - `check_order_tool` - Check order details
  - `create_sub_order_tool` - Create sub-orders when needed
  - `process_return_tool` - Handle return processing
  - `check_supplier_tool` - Verify supplier information

### 📦 **Picker Agent (Order Picker)**
- **Color Code**: Teal (#4ECDC4)
- **Primary Role**: Optimizing item picking and collection
- **Key Responsibilities**:
  - Optimize picking routes through the warehouse
  - Locate items for order fulfillment
  - Create and manage picking tasks
  - Update picking status and progress

- **Available Tools** (5):
  - `locate_item_tool` - Find items in warehouse
  - `check_order_tool` - Verify order information
  - `create_picking_task_tool` - Create new picking tasks
  - `update_picking_task_tool` - Update picking progress
  - `path_optimize_tool` - Optimize picking routes

### 📋 **Packer Agent (Order Packer)**
- **Color Code**: Blue (#45B7D1)
- **Primary Role**: Order verification and packaging
- **Key Responsibilities**:
  - Verify order completeness
  - Create packing tasks
  - Update packing status
  - Optimize packaging methods

- **Available Tools** (4):
  - `locate_item_tool` - Find items for packing
  - `check_order_tool` - Verify order details
  - `create_packing_task_tool` - Create packing tasks
  - `update_packing_task_tool` - Update packing progress

### 🚛 **Driver Agent (Delivery Driver)**
- **Color Code**: Green (#96CEB4)
- **Primary Role**: Delivery and route optimization
- **Key Responsibilities**:
  - Optimize delivery routes
  - Select appropriate vehicles
  - Manage delivery schedules
  - Update shipping status

- **Available Tools** (3):
  - `check_order_tool` - Check order for delivery
  - `calculate_route_tool` - Calculate optimal routes
  - `vehicle_select_tool` - Select appropriate vehicle

### 👔 **Manager Agent (Warehouse Manager)**
- **Color Code**: Yellow (#FFEAA7)
- **Primary Role**: Overall oversight and coordination
- **Key Responsibilities**:
  - Oversee all warehouse operations
  - Approve orders and workflows
  - Manage worker assignments
  - Generate analytics and reports
  - Monitor system performance
  - Detect and handle anomalies

- **Available Tools** (8):
  - `inventory_query_tool` - Query all inventory
  - `inventory_update_tool` - Update any inventory
  - `check_order_tool` - Access all orders
  - `approve_orders_tool` - Approve order processing
  - `worker_manage_tool` - Manage workforce
  - `check_analytics_tool` - Generate reports
  - `system_manage_tool` - System administration
  - `check_anomalies_tool` - Detect anomalies

## Workflow Processes

### 1. Order Fulfillment Flow
```
Customer Request → Manager (Approval) → Picker (Collection) → Packer (Verification) → Driver (Delivery) → Customer
```

**Detailed Steps**:
1. **Order Received**: Customer places order
2. **Manager Approval**: Manager reviews and approves order
3. **Picking Assignment**: Manager assigns picking tasks
4. **Route Optimization**: Picker optimizes collection route
5. **Item Collection**: Picker collects items from warehouse
6. **Packing Assignment**: Items sent to packer
7. **Order Verification**: Packer verifies order completeness
8. **Package Creation**: Packer creates final package
9. **Shipping Assignment**: Package assigned to driver
10. **Vehicle Selection**: Driver selects appropriate vehicle
11. **Route Planning**: Driver plans delivery route
12. **Order Delivered**: Order delivered to customer

### 2. Return Process
```
Customer Return → Clerk (Processing) → Inventory Update
```

**Detailed Steps**:
1. **Return Request**: Customer initiates return
2. **Return Processing**: Clerk processes the return
3. **Inventory Update**: Items added back to inventory

### 3. Receiving Process
```
Supplier Delivery → Clerk (Receiving) → Inventory Addition → Location Assignment
```

**Detailed Steps**:
1. **Supplier Delivery**: Supplier delivers new inventory
2. **Item Receiving**: Clerk receives and verifies items
3. **Inventory Addition**: Items added to system
4. **Location Assignment**: Items assigned warehouse locations

## Agent Interactions

### **Manager as Central Hub**
- The Manager Agent acts as the central coordination hub
- Has oversight of all other agents
- Can access most tools and approve major operations
- Monitors system performance and detects anomalies

### **Sequential Workflow**
- **Clerk** ↔ **Manager**: Inventory coordination
- **Manager** → **Picker**: Task assignment
- **Picker** → **Packer**: Item handoff
- **Packer** → **Driver**: Package handoff
- **Driver** → **Customer**: Final delivery

### **Independent Operations**
- **Clerk Agent**: Operates independently for returns and receiving
- Each agent can access shared tools like `check_order_tool` and `locate_item_tool`

## Technology Stack

### **Core Technologies**
- **LangChain**: Agent framework and tool orchestration
- **OpenAI GPT**: Natural language processing and decision making
- **FastAPI**: REST API for agent communication
- **ChromaDB**: Vector database for knowledge retrieval
- **NetworkX**: Path optimization and route planning

### **Data Storage**
- **MongoDB**: Primary database for persistent storage
- **Vector Database**: Knowledge base and document retrieval
- **In-memory Storage**: Conversation history and session data

### **Integration Layer**
- **API Client**: Handles external system integration
- **Authentication**: Role-based access control
- **Middleware**: CORS and request processing

## Key Features

### **Role-Based Access Control**
- Each agent has specific tools and permissions
- Authentication system prevents unauthorized access
- Development mode for testing and debugging

### **Intelligent Tool Usage**
- Agents automatically select appropriate tools
- Tools are shared where appropriate (e.g., `check_order_tool`)
- Tool access is restricted based on agent role

### **Knowledge Integration**
- Agents can access warehouse knowledge base
- Procedure documents and protocols
- Best practices and standard operating procedures

### **Conversation Management**
- Persistent conversation history
- User-specific conversation storage
- Context-aware responses

## Development and Deployment

### **Running the Visualization**
```bash
# Install dependencies
pip install -r visualization_requirements.txt

# Generate visualizations
python agent_workflow_visualizer.py
```

### **Starting the Chatbot API**
```bash
# Install main dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

### **Environment Configuration**
- OpenAI API key for LLM functionality
- LangSmith for tracing and monitoring
- MongoDB for data persistence
- ChromaDB for vector storage

## Future Enhancements

### **Potential Improvements**
1. **Multi-agent Collaboration**: Agents working together on complex tasks
2. **Real-time Notifications**: Push notifications for urgent tasks
3. **Performance Analytics**: Detailed agent performance metrics
4. **Machine Learning**: Predictive analytics for demand forecasting
5. **Mobile Interface**: Mobile app for warehouse workers
6. **Voice Integration**: Voice commands for hands-free operation

### **Scalability Considerations**
- Horizontal scaling of agent instances
- Load balancing for high-traffic scenarios
- Distributed task queue for complex operations
- Microservices architecture for better maintainability

---

*This documentation provides a comprehensive overview of the WMS agentic workflow. The generated visualizations complement this documentation by providing visual representations of the system architecture and agent interactions.*
