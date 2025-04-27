"""
This module provides functions to populate the knowledge base with warehouse-related documents.
"""

from ..utils.knowledge_base import knowledge_base

# Sample documents for warehouse procedures
WAREHOUSE_PROCEDURES = {
    "picking_procedure": """
# Standard Operating Procedure: Order Picking

## Overview
This document outlines the standard procedure for picking items from warehouse inventory to fulfill customer orders.

## Responsibilities
The Picker is responsible for efficiently collecting items from their storage locations while maintaining inventory accuracy.

## Equipment Needed
- Barcode scanner
- Mobile cart/trolley
- Personal protective equipment (PPE)
- Mobile device with WMS access

## Procedure

1. **Task Assignment**
   - Log into the WMS system
   - Accept picking tasks assigned to you
   - Review task details including priority and deadlines

2. **Preparation**
   - Ensure your equipment is functional
   - Check that your mobile device is showing the correct order information
   - Prepare your cart/trolley for the task

3. **Path Optimization**
   - Use the "path_optimize" tool to determine the most efficient route
   - Follow the suggested path to minimize travel distance

4. **Item Location**
   - For each item, navigate to the indicated storage location
   - Use the "locate_item" tool if you need assistance finding an item
   - Verify you are at the correct location by checking zone, aisle, shelf, and bin markers

5. **Item Selection**
   - Select the exact item specified in the order
   - Verify the SKU/barcode matches the order
   - Check for damage or quality issues
   - Scan the item barcode to confirm selection

6. **Quantity Verification**
   - Count items carefully to ensure you pick the exact quantity ordered
   - For bulk items, use scales when necessary
   - If the required quantity is not available, update the system immediately

7. **System Update**
   - Confirm each item as picked in the WMS
   - Update status for partially fulfilled items
   - Use the "update_picking_task" tool to mark progress

8. **Order Completion**
   - When all items are collected, review the order for accuracy
   - Mark the picking task as complete in the system
   - Transport items to the packing station
   - Notify the packing team that the order is ready

9. **Exception Handling**
   - If an item cannot be found, check nearby locations
   - If an item is damaged, report it through the system
   - If quantity is insufficient, pick what is available and notify management
   - For any issues that prevent completion, update the task status and add notes

10. **Documentation**
    - Record any discrepancies or issues encountered
    - Document any inventory adjustments needed
    - Complete all required fields in the picking task

## Quality Standards
- 100% accuracy in item selection
- 98% or better fill rate
- Less than 1% error rate in quantities
- Average pick rate of 60 items per hour

## Safety Guidelines
- Always wear required PPE
- Follow proper lifting techniques
- Observe speed limits for carts/trolleys
- Report any safety concerns immediately
""",

    "packing_procedure": """
# Standard Operating Procedure: Order Packing

## Overview
This document outlines the standard procedure for packing picked items for customer shipment.

## Responsibilities
The Packer is responsible for securely packaging items, verifying order completeness, and preparing shipments.

## Equipment Needed
- Packing materials (boxes, padding, tape, etc.)
- Label printer
- Barcode scanner
- Scale
- Mobile device with WMS access

## Procedure

1. **Task Assignment**
   - Log into the WMS system
   - Accept packing tasks assigned to you
   - Review task details including any special instructions

2. **Order Verification**
   - Receive picked items from the picker
   - Use the "check_order" tool to verify all items are present
   - Count items and compare against the order manifest
   - Verify item SKUs match the order details

3. **Packaging Selection**
   - Choose appropriate packaging based on:
     - Item size and weight
     - Fragility
     - Shipping method
     - Customer requirements

4. **Packing Process**
   - Scan each item before packing
   - Use appropriate padding/protection for fragile items
   - Pack items efficiently to minimize box size
   - Ensure items cannot move within the package
   - Include any required documentation (invoices, return forms)

5. **Partial Orders**
   - If order is incomplete, determine if it can be partially shipped
   - Use the "create_sub_order" tool for items that will ship later
   - Document the reason for partial fulfillment

6. **Package Sealing**
   - Securely seal the package with tape
   - Ensure the package is sturdy and will not open during transit
   - Check that the weight does not exceed shipping carrier limits

7. **Labeling**
   - Generate and print shipping label
   - Affix shipping label prominently on the package
   - Add any special handling labels (Fragile, This Side Up, etc.)
   - Apply internal tracking barcode

8. **System Update**
   - Mark the packing task as complete in the WMS
   - Update package dimensions and weight
   - Use the "update_packing_task" tool to record completion
   - Record any packaging materials used for inventory purposes

9. **Shipment Staging**
   - Place the completed package in the appropriate staging area
   - Organize by carrier and shipping priority
   - Alert shipping personnel that packages are ready

10. **Exception Handling**
    - If items are damaged, initiate replacement process
    - If packaging materials are unavailable, notify supervisor
    - For custom packaging requirements, consult guidelines
    - Document any issues encountered

## Quality Standards
- Zero mispacked items
- Less than 0.5% damage rate during shipping
- 99% accuracy in package contents
- Average packing rate of 20 orders per hour

## Safety Guidelines
- Use cutting tools safely
- Practice proper lifting techniques
- Report any repetitive strain issues
- Keep work area clear of trip hazards
""",

    "shipping_procedure": """
# Standard Operating Procedure: Order Shipping

## Overview
This document outlines the standard procedure for shipping packed orders to customers.

## Responsibilities
The Driver is responsible for selecting appropriate vehicles, loading shipments, and delivering orders.

## Equipment Needed
- Delivery vehicle
- Mobile device with WMS access
- Barcode scanner
- GPS navigation system
- Loading equipment (hand truck, pallet jack)

## Procedure

1. **Task Assignment**
   - Log into the WMS system
   - Review shipping tasks assigned to you
   - Check delivery locations and estimated volumes

2. **Vehicle Selection**
   - Use the "vehicle_select" tool to choose appropriate vehicle
   - Consider:
     - Total package volume and weight
     - Delivery distance and route
     - Special requirements (refrigeration, etc.)
     - Fuel efficiency

3. **Route Planning**
   - Use the "calculate_route" tool to optimize delivery sequence
   - Review the route for efficiency
   - Check for traffic conditions or road closures
   - Estimate delivery times for customer notification

4. **Shipment Verification**
   - Scan each package before loading
   - Verify shipping labels and destinations
   - Check for any damage before departure
   - Confirm all packages for the route are accounted for

5. **Vehicle Loading**
   - Load packages in reverse delivery order
   - Secure packages to prevent shifting during transport
   - Distribute weight evenly in the vehicle
   - Keep fragile items protected from heavier items

6. **Departure Procedures**
   - Complete vehicle safety check
   - Update shipping task status to "in transit"
   - Notify dispatch of departure
   - Begin following planned route

7. **Delivery Process**
   - Notify customer of approximate arrival time
   - Scan package at delivery
   - Obtain signature if required
   - Take delivery photo when appropriate
   - Provide any necessary instructions to the customer

8. **System Update**
   - Mark each delivery as complete in real-time
   - Use the "update_shipping_task" tool to record status
   - Document any issues or exceptions
   - Record mileage and fuel usage

9. **Return to Warehouse**
   - Return any undeliverable packages
   - Complete end-of-day vehicle inspection
   - Submit all delivery documentation
   - Prepare vehicle for next use

10. **Exception Handling**
    - If customer is not available, follow reattempt procedure
    - If address is incorrect, contact dispatch for instructions
    - For damaged packages, document and return to warehouse
    - In case of vehicle issues, notify management immediately

## Quality Standards
- 98% on-time delivery rate
- Zero package damage during transport
- 100% accuracy in package delivery
- Complete documentation for all deliveries

## Safety Guidelines
- Always follow traffic laws and regulations
- Use proper lifting techniques
- Take required rest periods
- Report any safety concerns immediately
""",

    "receiving_procedure": """
# Standard Operating Procedure: Receiving

## Overview
This document outlines the standard procedure for receiving and processing incoming inventory.

## Responsibilities
The Receiving Clerk is responsible for accepting deliveries, inspecting items, and updating inventory.

## Equipment Needed
- Barcode scanner
- Mobile device with WMS access
- Pallet jack/forklift (if certified)
- Inspection tools
- Labeling supplies

## Procedure

1. **Delivery Notification**
   - Receive advance delivery notification
   - Prepare receiving area for expected items
   - Review purchase orders for expected contents

2. **Shipment Arrival**
   - Greet delivery personnel
   - Verify delivery is for your facility
   - Direct truck to appropriate dock

3. **Documentation Review**
   - Check delivery against purchase order
   - Verify item quantities and SKUs
   - Confirm supplier information is correct
   - Note any discrepancies on delivery receipt

4. **Unloading Process**
   - Ensure safe unloading of items
   - Count packages/pallets as they are unloaded
   - Place items in designated receiving area
   - Keep supplier shipments separated

5. **Detailed Inspection**
   - Open packages to verify contents
   - Check for any damage or quality issues
   - Verify item specifications match purchase order
   - Test functionality if required

6. **System Update**
   - Use the "inventory_add" tool to add new items
   - Scan each item into inventory
   - Assign appropriate storage locations
   - Document any discrepancies

7. **Labeling**
   - Generate and apply internal barcodes if needed
   - Ensure all items have proper identification
   - Mark any items requiring special handling

8. **Storage Assignment**
   - Determine optimal storage location
   - Consider item size, weight, and frequency of access
   - Update location in the WMS system
   - Transport items to assigned locations

9. **Documentation Completion**
   - Sign and date all receiving documents
   - File paperwork according to procedure
   - Send electronic confirmation to supplier
   - Notify purchasing of any issues

10. **Exception Handling**
    - For damaged items, initiate return process
    - If quantity discrepancies exist, contact supplier
    - For unknown or unexpected items, quarantine and investigate
    - Document all exceptions with photos when possible

## Quality Standards
- 100% accuracy in received item documentation
- Less than 24-hour turnaround from receipt to storage
- 99% accuracy in quantity verification
- Zero undocumented discrepancies

## Safety Guidelines
- Use proper lifting techniques
- Operate material handling equipment safely
- Wear appropriate PPE
- Report any safety hazards immediately
""",

    "returns_procedure": """
# Standard Operating Procedure: Returns Processing

## Overview
This document outlines the standard procedure for processing returned items from customers.

## Responsibilities
The Returns Clerk is responsible for receiving, inspecting, and processing returned merchandise.

## Equipment Needed
- Barcode scanner
- Mobile device with WMS access
- Inspection tools
- Testing equipment
- Repackaging materials

## Procedure

1. **Return Receipt**
   - Accept returned packages
   - Log the return in the system
   - Verify return authorization if required
   - Match return to original order

2. **Package Inspection**
   - Check package for external damage
   - Document condition with photos
   - Open package carefully
   - Verify contents match return documentation

3. **Item Assessment**
   - Inspect item for damage or wear
   - Test functionality when applicable
   - Categorize return reason (damaged, unwanted, incorrect item, etc.)
   - Determine if item can be restocked

4. **Return Categorization**
   - Designate as:
     - Resellable (return to inventory)
     - Refurbishable (send to repair)
     - Damaged (discard or return to supplier)
     - Mis-shipped (return to correct inventory)

5. **System Update**
   - Use the "process_return" tool to document the return
   - Update inventory quantities for resellable items
   - Process customer refund if applicable
   - Document disposition of returned items

6. **Inventory Reintegration**
   - For resellable items:
     - Repackage if necessary
     - Generate new barcode if needed
     - Assign storage location
     - Transport to appropriate warehouse area

7. **Customer Communication**
   - Confirm receipt of return to customer
   - Provide refund or replacement status
   - Address any customer concerns
   - Document all communications

8. **Supplier Returns**
   - For items to be returned to supplier:
     - Prepare return documentation
     - Package according to supplier requirements
     - Schedule pickup or delivery
     - Track until supplier confirms receipt

9. **Analysis and Reporting**
   - Document return reason codes
   - Identify patterns in returns
   - Generate reports for management
   - Suggest process improvements based on findings

10. **Exception Handling**
    - For unauthorized returns, contact customer
    - If item condition is disputed, escalate to management
    - For high-value returns, follow special handling procedure
    - Document any unusual circumstances

## Quality Standards
- Process returns within 48 hours of receipt
- 100% accuracy in return categorization
- 98% accuracy in inventory updates
- Complete documentation for all returns

## Safety Guidelines
- Use caution when opening packages
- Wear gloves when handling returned items
- Follow proper disposal procedures for damaged items
- Report any potentially hazardous returns immediately
""",

    "inventory_management": """
# Standard Operating Procedure: Inventory Management

## Overview
This document outlines the procedures for maintaining accurate inventory records and optimizing stock levels.

## Responsibilities
Managers and inventory specialists are responsible for monitoring, adjusting, and optimizing inventory.

## Tools Needed
- WMS system access
- Barcode scanners
- Cycle counting equipment
- Analytics software

## Procedures

1. **Inventory Monitoring**
   - Review inventory levels daily
   - Use the "inventory_query" tool to check current stock
   - Monitor fast-moving items closely
   - Identify slow-moving inventory

2. **Stock Replenishment**
   - Set reorder points based on:
     - Historical demand
     - Lead time
     - Seasonality
     - Safety stock requirements
   - Generate purchase orders automatically or manually
   - Verify orders with suppliers

3. **Cycle Counting**
   - Implement regular cycle counting schedule
   - Count high-value items more frequently
   - Investigate and resolve discrepancies immediately
   - Document all count results and adjustments

4. **Inventory Optimization**
   - Analyze inventory turnover rates
   - Identify dead stock (no movement in 12 months)
   - Adjust stock levels based on forecasted demand
   - Balance inventory investment against service levels

5. **Location Management**
   - Optimize product placement based on:
     - Pick frequency
     - Size and weight
     - Special storage requirements
   - Use ABC classification for positioning
   - Implement slotting optimization

6. **Anomaly Detection**
   - Use the "check_anomalies" tool to identify discrepancies
   - Investigate significant variances
   - Implement corrective actions
   - Document resolution for all anomalies

7. **Inventory Accuracy**
   - Maintain 98%+ inventory accuracy
   - Perform full physical inventory annually
   - Reconcile system counts with physical counts
   - Address systemic issues causing discrepancies

8. **Seasonal Planning**
   - Forecast seasonal demand patterns
   - Adjust inventory levels proactively
   - Plan for storage of seasonal merchandise
   - Implement special handling for seasonal transitions

9. **Reporting and Analytics**
   - Generate regular inventory reports:
     - Inventory valuation
     - Stock status
     - Aging report
     - Turnover analysis
   - Share reports with relevant departments

10. **Exception Handling**
    - For significant discrepancies, conduct investigation
    - When supplier shortages occur, identify alternatives
    - For damaged inventory, process write-offs
    - During system issues, implement manual procedures

## Quality Standards
- 98% or higher inventory accuracy
- Less than 2% of total inventory as dead stock
- Inventory turnover appropriate for product category
- Stock-outs below 1% for A-class items

## Best Practices
- Review reorder points quarterly
- Validate forecasts against actual demand
- Maintain clear documentation of all adjustments
- Cross-train staff on inventory procedures
""",

    "path_optimization": """
# Warehouse Path Optimization Guidelines

## Overview
This document provides guidelines for optimizing picking paths in the warehouse to maximize efficiency and reduce travel time.

## Optimization Principles

1. **Travel Distance Minimization**
   - Reduce total walking/driving distance
   - Eliminate backtracking wherever possible
   - Group picks in same zone/aisle
   - Use the "path_optimize" tool for complex orders

2. **Wave Picking**
   - Group similar orders together
   - Pick for multiple orders simultaneously
   - Sort items into individual orders during packing
   - Optimize for shared locations across orders

3. **Zone Picking**
   - Divide warehouse into logical zones
   - Assign pickers to specific zones
   - Transfer items between zones efficiently
   - Balance workload across zones

4. **Slotting Optimization**
   - Place fast-moving items in easily accessible locations
   - Store commonly purchased together items nearby
   - Position heavy items at appropriate heights
   - Consider item dimensions and weights

5. **Path Algorithms**
   - S-pattern: Zigzag through aisles
   - Return: Up one aisle and down the next
   - Midpoint: Split aisle with two pickers
   - Largest gap: Skip aisles with few picks

6. **Equipment Utilization**
   - Match equipment to picking task
   - Consider order volume and item characteristics
   - Use mobile picking stations when appropriate
   - Maintain equipment for reliable operation

7. **Real-time Adjustments**
   - Dynamically reassign tasks as conditions change
   - Incorporate new orders into existing routes
   - Adjust for blocked aisles or equipment issues
   - Use WMS to recalculate optimal paths

8. **Performance Metrics**
   - Measure picks per hour
   - Track travel time vs. pick time
   - Calculate distance traveled per order
   - Compare actual vs. optimal paths

9. **Warehouse Layout Considerations**
   - Ensure clear aisle markings
   - Maintain optimal aisle widths
   - Design cross-aisles at strategic points
   - Position high-velocity items near shipping/packing

10. **Continuous Improvement**
    - Analyze path efficiency regularly
    - Gather picker feedback on route logic
    - Test alternative routing strategies
    - Update algorithms based on results

## Implementation Steps

1. Analyze current picking patterns
2. Map warehouse layout in WMS
3. Implement appropriate algorithm
4. Train pickers on new procedures
5. Monitor results and adjust as needed

## Technology Tools

- Use the "path_optimize" tool for complex multi-item orders
- Implement pick-to-light systems for high-density areas
- Consider voice-directed picking for hands-free operation
- Evaluate automated guided vehicles for heavy or bulk items
"""
}

def populate_knowledge_base():
    """
    Add warehouse procedure documents to the knowledge base.
    """
    print("Populating knowledge base with warehouse procedures...")
    
    for doc_name, content in WAREHOUSE_PROCEDURES.items():
        knowledge_base.add_text(
            text=content,
            metadata={
                "type": "sop",
                "description": "Standard Operating Procedure",
                "title": doc_name.replace("_", " ").title()
            }
        )
        
    print(f"Added {len(WAREHOUSE_PROCEDURES)} documents to the knowledge base.")

if __name__ == "__main__":
    populate_knowledge_base()