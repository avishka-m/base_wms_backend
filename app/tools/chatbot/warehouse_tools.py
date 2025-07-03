from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import random
import datetime

# Use absolute imports instead of relative imports
from app.tools.chatbot.base_tool import WMSBaseTool, create_tool
from app.utils.chatbot.api_client import api_client
from app.utils.chatbot.knowledge_base import knowledge_base

def check_supplier_func(supplier_id: Optional[int] = None,
                      name: Optional[str] = None,
                      product_category: Optional[str] = None) -> str:
    """
    Check information about a supplier.
    
    Args:
        supplier_id: Optional supplier ID to look up
        name: Optional supplier name to search for
        product_category: Optional product category to filter by
        
    Returns:
        Information about the supplier(s)
    """
    try:
        params = {}
        
        if supplier_id is not None:
            params["id"] = supplier_id
            
        if name:
            params["name"] = name
            
        if product_category:
            params["category"] = product_category
            
        suppliers = api_client.get("supplier", params)
        
        if not suppliers:
            return "No suppliers found matching your criteria."
            
        result = "Supplier Information:\n\n"
        
        for supplier in suppliers:
            result += f"ID: {supplier.get('id')}\n"
            result += f"Name: {supplier.get('name')}\n"
            result += f"Contact: {supplier.get('contact_name')}\n"
            result += f"Email: {supplier.get('email')}\n"
            result += f"Phone: {supplier.get('phone')}\n"
            result += f"Categories: {', '.join(supplier.get('categories', []))}\n"
            result += f"Address: {supplier.get('address')}\n"
            result += f"Active: {'Yes' if supplier.get('active') else 'No'}\n"
            result += f"Rating: {supplier.get('rating', 'N/A')}/5\n"
            result += f"Lead Time: {supplier.get('lead_time', 'N/A')} days\n"
            result += "-" * 40 + "\n"
            
        return result
        
    except Exception as e:
        return f"Error checking supplier information: {str(e)}"

def vehicle_select_func(weight: float,
                       volume: Optional[float] = None,
                       distance: Optional[float] = None,
                       refrigerated: Optional[bool] = False,
                       items_count: Optional[int] = None) -> str:
    """
    Select an appropriate vehicle for delivery based on requirements.
    
    Args:
        weight: Total weight of the shipment in kg
        volume: Optional volume of the shipment in cubic meters
        distance: Optional delivery distance in km
        refrigerated: Whether refrigeration is required
        items_count: Optional number of items
        
    Returns:
        Recommended vehicle and explanation
    """
    try:
        params = {}
        
        if refrigerated:
            params["refrigerated"] = True
            
        # Get available vehicles
        vehicles = api_client.get("vehicles", params)
        
        if not vehicles:
            return "No vehicles found matching your criteria."
            
        # Filter vehicles based on capacity
        suitable_vehicles = []
        
        for vehicle in vehicles:
            vehicle_capacity = vehicle.get('capacity', 0)
            vehicle_volume = vehicle.get('volume', 0)
            
            if vehicle_capacity >= weight and (volume is None or vehicle_volume >= volume):
                suitable_vehicles.append(vehicle)
                
        if not suitable_vehicles:
            return f"No vehicles found with sufficient capacity for weight: {weight} kg, volume: {volume} cubic meters."
            
        # Calculate efficiency score for each vehicle
        scored_vehicles = []
        
        for vehicle in suitable_vehicles:
            # Calculate utilization percentage
            weight_utilization = min(weight / vehicle.get('capacity', 1) * 100, 100)
            volume_utilization = 100
            if volume is not None and vehicle.get('volume'):
                volume_utilization = min(volume / vehicle.get('volume') * 100, 100)
                
            # Calculate fuel efficiency score
            fuel_score = 100
            if distance is not None:
                # Higher MPG is better
                mpg = vehicle.get('mpg', 10)
                fuel_score = min(mpg * 5, 100)
                
            # Calculate overall score (higher is better)
            avg_utilization = (weight_utilization + volume_utilization) / 2
            # Ideal utilization is around 80%
            utilization_score = 100 - abs(avg_utilization - 80)
            
            # Overall score - weighted average
            overall_score = (utilization_score * 0.6) + (fuel_score * 0.4)
            
            scored_vehicles.append({
                "vehicle": vehicle,
                "score": overall_score,
                "weight_utilization": weight_utilization,
                "volume_utilization": volume_utilization if volume is not None else None
            })
            
        # Sort by score (descending)
        scored_vehicles.sort(key=lambda x: x["score"], reverse=True)
        
        # Return recommendation
        best_vehicle = scored_vehicles[0]["vehicle"]
        
        result = "Vehicle Recommendation:\n\n"
        result += f"Recommended Vehicle: {best_vehicle.get('name')} (ID: {best_vehicle.get('id')})\n"
        result += f"Type: {best_vehicle.get('type')}\n"
        result += f"Capacity: {best_vehicle.get('capacity')} kg\n"
        
        if volume is not None:
            result += f"Volume: {best_vehicle.get('volume')} cubic meters\n"
            
        result += f"Refrigerated: {'Yes' if best_vehicle.get('refrigerated') else 'No'}\n"
        result += f"Weight Utilization: {scored_vehicles[0]['weight_utilization']:.1f}%\n"
        
        if volume is not None:
            result += f"Volume Utilization: {scored_vehicles[0]['volume_utilization']:.1f}%\n"
            
        result += "\nAlternative Vehicles:\n"
        
        # Show top 2 alternatives
        for scored_vehicle in scored_vehicles[1:3]:
            vehicle = scored_vehicle["vehicle"]
            result += f"- {vehicle.get('name')} (ID: {vehicle.get('id')}): {vehicle.get('capacity')} kg capacity, "
            result += f"Score: {scored_vehicle['score']:.1f}\n"
            
        return result
        
    except Exception as e:
        return f"Error selecting vehicle: {str(e)}"

def worker_manage_func(action: str,
                      worker_id: Optional[int] = None,
                      name: Optional[str] = None,
                      role: Optional[str] = None,
                      status: Optional[str] = None) -> str:
    """
    Manage warehouse workers (manager only).
    
    Args:
        action: Action to perform (list, info, update, assign)
        worker_id: Optional worker ID
        name: Optional worker name
        role: Optional worker role
        status: Optional worker status
        
    Returns:
        Result of the worker management action
    """
    try:
        if action.lower() == "list":
            # List workers, with optional filters
            params = {}
            
            if role:
                params["role"] = role
                
            if status:
                params["status"] = status
                
            workers = api_client.get("workers", params)
            
            if not workers:
                return "No workers found matching your criteria."
                
            result = "Worker List:\n\n"
            
            for worker in workers:
                result += f"ID: {worker.get('id')}\n"
                result += f"Name: {worker.get('name')}\n"
                result += f"Role: {worker.get('role')}\n"
                result += f"Status: {worker.get('status')}\n"
                result += "-" * 30 + "\n"
                
            return result
            
        elif action.lower() == "info":
            # Get detailed info about a specific worker
            if worker_id is None and not name:
                return "Error: Either worker_id or name must be provided for info action."
                
            params = {}
            
            if worker_id is not None:
                params["id"] = worker_id
                
            if name:
                params["name"] = name
                
            workers = api_client.get("workers", params)
            
            if not workers:
                return f"No worker found with the specified {'ID' if worker_id else 'name'}."
                
            worker = workers[0]
            
            result = "Worker Details:\n\n"
            result += f"ID: {worker.get('id')}\n"
            result += f"Name: {worker.get('name')}\n"
            result += f"Role: {worker.get('role')}\n"
            result += f"Status: {worker.get('status')}\n"
            result += f"Email: {worker.get('email')}\n"
            result += f"Phone: {worker.get('phone')}\n"
            result += f"Hire Date: {worker.get('hire_date')}\n"
            result += f"Certifications: {', '.join(worker.get('certifications', []))}\n"
            
            # Get current task if any
            try:
                tasks = []
                if worker.get('role') == 'picker':
                    tasks = api_client.get("picking", {"worker_id": worker.get('id'), "status": "in_progress"})
                elif worker.get('role') == 'packer':
                    tasks = api_client.get("packing", {"worker_id": worker.get('id'), "status": "in_progress"})
                elif worker.get('role') == 'driver':
                    tasks = api_client.get("shipping", {"worker_id": worker.get('id'), "status": "in_progress"})
                    
                if tasks:
                    result += "\nCurrent Tasks:\n"
                    for task in tasks:
                        result += f"- Task #{task.get('id')}: {task.get('status')}\n"
                        result += f"  Order: {task.get('order_id')}\n"
                        result += f"  Started: {task.get('updated_at')}\n"
            except:
                pass
                
            return result
            
        elif action.lower() == "update":
            # Update worker information
            if worker_id is None:
                return "Error: worker_id is required for update action."
                
            update_data = {}
            
            if role:
                update_data["role"] = role
                
            if status:
                update_data["status"] = status
                
            if name:
                update_data["name"] = name
                
            if not update_data:
                return "Error: At least one field (name, role, status) must be provided for update."
                
            response = api_client.put("workers", worker_id, update_data)
            
            return f"Successfully updated worker {worker_id}."
            
        elif action.lower() == "assign":
            # Assign a worker to a task
            return "Worker assignment functionality is handled by the specific task update tools (update_picking_task, update_packing_task, update_shipping_task)."
            
        else:
            return f"Error: Unknown action '{action}'. Valid actions are: list, info, update, assign."
            
    except Exception as e:
        return f"Error managing workers: {str(e)}"

def check_analytics_func(report_type: str,
                        period: Optional[str] = "week",
                        category: Optional[str] = None) -> str:
    """
    Check warehouse analytics reports (manager only).
    
    Args:
        report_type: Type of report (inventory, orders, efficiency, workers)
        period: Optional time period (day, week, month, quarter, year)
        category: Optional category filter
        
    Returns:
        Analytics report
    """
    try:
        # Simplified implementation - in a real system, this would query a data warehouse
        # or analytics service
        
        valid_periods = ["day", "week", "month", "quarter", "year"]
        if period.lower() not in valid_periods:
            return f"Error: Invalid period '{period}'. Valid periods are: {', '.join(valid_periods)}."
            
        # Get current date for report period
        today = datetime.datetime.now()
        
        # Simulate data generation based on report type
        if report_type.lower() == "inventory":
            # Generate inventory analytics
            result = f"Inventory Analytics Report ({period.capitalize()}):\n\n"
            
            # Simulate inventory metrics
            metrics = {
                "total_items": random.randint(5000, 15000),
                "unique_skus": random.randint(500, 2000),
                "inventory_value": round(random.uniform(100000, 500000), 2),
                "low_stock_items": random.randint(10, 100),
                "out_of_stock_items": random.randint(5, 50),
                "inventory_turnover": round(random.uniform(3, 8), 2),
                "dead_stock_value": round(random.uniform(5000, 30000), 2),
                "dead_stock_percentage": round(random.uniform(1, 10), 2)
            }
            
            result += f"Total Items: {metrics['total_items']}\n"
            result += f"Unique SKUs: {metrics['unique_skus']}\n"
            result += f"Inventory Value: ${metrics['inventory_value']}\n"
            result += f"Low Stock Items: {metrics['low_stock_items']}\n"
            result += f"Out of Stock Items: {metrics['out_of_stock_items']}\n"
            result += f"Inventory Turnover: {metrics['inventory_turnover']}\n"
            result += f"Dead Stock Value: ${metrics['dead_stock_value']} ({metrics['dead_stock_percentage']}%)\n\n"
            
            if category:
                result += f"Category Analysis: {category}\n"
                category_metrics = {
                    "category_items": random.randint(200, 1000),
                    "category_value": round(random.uniform(10000, 100000), 2),
                    "category_turnover": round(random.uniform(2, 10), 2)
                }
                result += f"Items in Category: {category_metrics['category_items']}\n"
                result += f"Category Value: ${category_metrics['category_value']}\n"
                result += f"Category Turnover: {category_metrics['category_turnover']}\n\n"
                
            result += "Top Categories by Value:\n"
            categories = ["Electronics", "Clothing", "Home Goods", "Sports", "Toys"]
            for i, cat in enumerate(categories, 1):
                result += f"{i}. {cat}: ${round(random.uniform(10000, 100000), 2)}\n"
                
            return result
            
        elif report_type.lower() == "orders":
            # Generate order analytics
            result = f"Order Analytics Report ({period.capitalize()}):\n\n"
            
            # Simulate order metrics
            metrics = {
                "total_orders": random.randint(200, 2000),
                "total_revenue": round(random.uniform(20000, 200000), 2),
                "average_order_value": round(random.uniform(100, 500), 2),
                "completed_orders": random.randint(180, 1800),
                "cancelled_orders": random.randint(5, 50),
                "return_rate": round(random.uniform(1, 10), 2),
                "on_time_delivery": round(random.uniform(90, 99), 2)
            }
            
            result += f"Total Orders: {metrics['total_orders']}\n"
            result += f"Total Revenue: ${metrics['total_revenue']}\n"
            result += f"Average Order Value: ${metrics['average_order_value']}\n"
            result += f"Completed Orders: {metrics['completed_orders']}\n"
            result += f"Cancelled Orders: {metrics['cancelled_orders']}\n"
            result += f"Return Rate: {metrics['return_rate']}%\n"
            result += f"On-Time Delivery Rate: {metrics['on_time_delivery']}%\n\n"
            
            result += "Order Volume by Day:\n"
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for day in days:
                result += f"{day}: {random.randint(30, 300)} orders\n"
                
            return result
            
        elif report_type.lower() == "efficiency":
            # Generate efficiency analytics
            result = f"Warehouse Efficiency Report ({period.capitalize()}):\n\n"
            
            # Simulate efficiency metrics
            metrics = {
                "picking_rate": round(random.uniform(50, 100), 2),
                "packing_rate": round(random.uniform(20, 50), 2),
                "order_cycle_time": round(random.uniform(2, 8), 2),
                "labor_efficiency": round(random.uniform(85, 98), 2),
                "space_utilization": round(random.uniform(70, 95), 2),
                "errors_per_thousand": round(random.uniform(0.5, 5), 2),
                "dock_to_stock_time": round(random.uniform(2, 12), 2)
            }
            
            result += f"Average Picking Rate: {metrics['picking_rate']} items/hour\n"
            result += f"Average Packing Rate: {metrics['packing_rate']} orders/hour\n"
            result += f"Order Cycle Time: {metrics['order_cycle_time']} hours\n"
            result += f"Labor Efficiency: {metrics['labor_efficiency']}%\n"
            result += f"Space Utilization: {metrics['space_utilization']}%\n"
            result += f"Error Rate: {metrics['errors_per_thousand']} per 1000 items\n"
            result += f"Dock to Stock Time: {metrics['dock_to_stock_time']} hours\n\n"
            
            result += "Efficiency by Zone:\n"
            zones = ["A", "B", "C", "D"]
            for zone in zones:
                result += f"Zone {zone}: {round(random.uniform(80, 99), 2)}% efficiency\n"
                
            return result
            
        elif report_type.lower() == "workers":
            # Generate worker analytics
            result = f"Worker Performance Report ({period.capitalize()}):\n\n"
            
            # Simulate worker metrics
            metrics = {
                "total_workers": random.randint(20, 100),
                "active_workers": random.randint(15, 90),
                "average_tasks_per_worker": round(random.uniform(10, 50), 2),
                "productivity_rate": round(random.uniform(80, 95), 2),
                "overtime_hours": round(random.uniform(10, 100), 2),
                "labor_cost": round(random.uniform(5000, 50000), 2)
            }
            
            result += f"Total Workers: {metrics['total_workers']}\n"
            result += f"Active Workers: {metrics['active_workers']}\n"
            result += f"Average Tasks per Worker: {metrics['average_tasks_per_worker']}\n"
            result += f"Productivity Rate: {metrics['productivity_rate']}%\n"
            result += f"Total Overtime Hours: {metrics['overtime_hours']} hours\n"
            result += f"Total Labor Cost: ${metrics['labor_cost']}\n\n"
            
            if category:
                # Interpret category as role for worker reports
                result += f"Performance by Role: {category}\n"
                role_metrics = {
                    "workers_in_role": random.randint(5, 30),
                    "role_productivity": round(random.uniform(80, 98), 2),
                    "tasks_completed": random.randint(100, 1000)
                }
                result += f"Workers in Role: {role_metrics['workers_in_role']}\n"
                result += f"Role Productivity: {role_metrics['role_productivity']}%\n"
                result += f"Tasks Completed: {role_metrics['tasks_completed']}\n\n"
                
            result += "Top Performing Workers:\n"
            for i in range(1, 6):
                worker_id = random.randint(1, 100)
                result += f"{i}. Worker #{worker_id}: {round(random.uniform(90, 100), 2)}% efficiency, {random.randint(20, 100)} tasks\n"
                
            return result
            
        else:
            return f"Error: Unknown report type '{report_type}'. Valid types are: inventory, orders, efficiency, workers."
            
    except Exception as e:
        return f"Error generating analytics report: {str(e)}"

def check_anomalies_func(anomaly_type: str,
                       threshold: Optional[float] = None,
                       period: Optional[str] = "week") -> str:
    """
    Check for anomalies in warehouse operations (manager only).
    
    Args:
        anomaly_type: Type of anomaly to check (inventory, orders, returns, efficiency)
        threshold: Optional sensitivity threshold (1-100, higher means fewer anomalies)
        period: Optional time period (day, week, month)
        
    Returns:
        Detected anomalies report
    """
    try:
        # Determine threshold - default to 70% (higher = fewer anomalies)
        if threshold is None:
            threshold = 70
        else:
            threshold = max(1, min(100, threshold))
            
        # Higher threshold means fewer anomalies
        anomaly_count = int(20 * (100 - threshold) / 100) + 1
        
        # Get current date for report period
        today = datetime.datetime.now()
        
        if anomaly_type.lower() == "inventory":
            # Check for inventory anomalies
            result = f"Inventory Anomaly Report ({period.capitalize()}, Threshold: {threshold}%):\n\n"
            
            if anomaly_count == 0:
                result += "No inventory anomalies detected with current threshold.\n"
                return result
                
            result += f"Detected {anomaly_count} potential inventory anomalies:\n\n"
            
            # Generate sample anomalies
            anomalies = [
                "Unexpected inventory decrease",
                "Inventory count mismatch",
                "Stock level below minimum threshold",
                "Excessive stock level",
                "Inventory not moving",
                "Cycle count discrepancy",
                "Stock in incorrect location",
                "Received quantity mismatch",
                "Damaged inventory spike",
                "Invalid stock adjustments"
            ]
            
            for i in range(min(anomaly_count, len(anomalies))):
                anomaly = anomalies[i]
                item_id = random.randint(1000, 9999)
                location = f"{random.choice(['A', 'B', 'C'])}-{random.randint(1, 20)}-{random.randint(1, 10)}-{random.randint(1, 20)}"
                confidence = round(random.uniform(60, 95), 1)
                
                result += f"{i+1}. {anomaly}\n"
                result += f"   Item ID: {item_id}\n"
                result += f"   Location: {location}\n"
                result += f"   Confidence: {confidence}%\n"
                result += f"   Detected: {(today - datetime.timedelta(hours=random.randint(1, 24*7))).strftime('%Y-%m-%d %H:%M')}\n\n"
                
            return result
            
        elif anomaly_type.lower() == "orders":
            # Check for order anomalies
            result = f"Order Processing Anomaly Report ({period.capitalize()}, Threshold: {threshold}%):\n\n"
            
            if anomaly_count == 0:
                result += "No order processing anomalies detected with current threshold.\n"
                return result
                
            result += f"Detected {anomaly_count} potential order anomalies:\n\n"
            
            # Generate sample anomalies
            anomalies = [
                "Excessive order processing time",
                "Stuck in picking status",
                "Incomplete order shipped",
                "Multiple partial shipments",
                "Order cancellation pattern",
                "Delivery delay pattern",
                "Excessive order modifications",
                "Unusual order volume",
                "Order price discrepancy",
                "Duplicate order creation"
            ]
            
            for i in range(min(anomaly_count, len(anomalies))):
                anomaly = anomalies[i]
                order_id = random.randint(10000, 99999)
                customer_id = random.randint(1000, 9999)
                confidence = round(random.uniform(60, 95), 1)
                
                result += f"{i+1}. {anomaly}\n"
                result += f"   Order ID: {order_id}\n"
                result += f"   Customer ID: {customer_id}\n"
                result += f"   Confidence: {confidence}%\n"
                result += f"   Detected: {(today - datetime.timedelta(hours=random.randint(1, 24*7))).strftime('%Y-%m-%d %H:%M')}\n\n"
                
            return result
            
        elif anomaly_type.lower() == "returns":
            # Check for return anomalies
            result = f"Returns Anomaly Report ({period.capitalize()}, Threshold: {threshold}%):\n\n"
            
            if anomaly_count == 0:
                result += "No return anomalies detected with current threshold.\n"
                return result
                
            result += f"Detected {anomaly_count} potential return anomalies:\n\n"
            
            # Generate sample anomalies
            anomalies = [
                "High return rate for product",
                "Unusual return reason pattern",
                "Customer with excessive returns",
                "Returns without prior purchase",
                "Delayed return processing",
                "Return amount discrepancy",
                "Seasonal return spike",
                "Return fraud pattern",
                "Product damage pattern",
                "Frequent exchanges"
            ]
            
            for i in range(min(anomaly_count, len(anomalies))):
                anomaly = anomalies[i]
                product_id = random.randint(1000, 9999)
                return_count = random.randint(5, 50)
                confidence = round(random.uniform(60, 95), 1)
                
                result += f"{i+1}. {anomaly}\n"
                result += f"   Product ID: {product_id}\n"
                result += f"   Return Count: {return_count}\n"
                result += f"   Confidence: {confidence}%\n"
                result += f"   Detected: {(today - datetime.timedelta(hours=random.randint(1, 24*7))).strftime('%Y-%m-%d %H:%M')}\n\n"
                
            return result
            
        elif anomaly_type.lower() == "efficiency":
            # Check for efficiency anomalies
            result = f"Operational Efficiency Anomaly Report ({period.capitalize()}, Threshold: {threshold}%):\n\n"
            
            if anomaly_count == 0:
                result += "No efficiency anomalies detected with current threshold.\n"
                return result
                
            result += f"Detected {anomaly_count} potential efficiency anomalies:\n\n"
            
            # Generate sample anomalies
            anomalies = [
                "Unusual worker productivity drop",
                "Excessive time in picking",
                "Equipment utilization drop",
                "Zone congestion pattern",
                "High error rate for worker",
                "Unexpected process delay",
                "Resource allocation imbalance",
                "Space utilization issue",
                "Picking path inefficiency",
                "Shift productivity variance"
            ]
            
            for i in range(min(anomaly_count, len(anomalies))):
                anomaly = anomalies[i]
                worker_id = random.randint(10, 99)
                zone = random.choice(["A", "B", "C", "D"])
                confidence = round(random.uniform(60, 95), 1)
                
                result += f"{i+1}. {anomaly}\n"
                result += f"   Worker ID: {worker_id}\n"
                result += f"   Zone: {zone}\n"
                result += f"   Confidence: {confidence}%\n"
                result += f"   Detected: {(today - datetime.timedelta(hours=random.randint(1, 24*7))).strftime('%Y-%m-%d %H:%M')}\n\n"
                
            return result
            
        else:
            return f"Error: Unknown anomaly type '{anomaly_type}'. Valid types are: inventory, orders, returns, efficiency."
            
    except Exception as e:
        return f"Error checking for anomalies: {str(e)}"

def system_manage_func(action: str,
                     system_area: str,
                     parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Manage warehouse system settings (manager only).
    
    Args:
        action: Action to perform (get, update, reset)
        system_area: System area to manage (picking, packing, shipping, inventory, security)
        parameters: Optional parameters for the action
        
    Returns:
        Result of the system management action
    """
    try:
        # This is a simplified implementation that would normally interact with system configuration
        
        if action.lower() == "get":
            # Get current system settings
            result = f"Current {system_area.capitalize()} System Settings:\n\n"
            
            if system_area.lower() == "picking":
                settings = {
                    "optimization_algorithm": "S-pattern",
                    "batch_picking_enabled": True,
                    "max_items_per_batch": 50,
                    "prioritize_by": "due_date",
                    "auto_assign": True,
                    "pick_confirmation_required": True
                }
                
            elif system_area.lower() == "packing":
                settings = {
                    "box_recommendation_enabled": True,
                    "packing_slip_format": "detailed",
                    "auto_generate_labels": True,
                    "qc_enabled": True,
                    "qc_frequency": "10%",
                    "weight_verification": True
                }
                
            elif system_area.lower() == "shipping":
                settings = {
                    "carrier_selection": "automatic",
                    "rate_shopping_enabled": True,
                    "consolidate_shipments": True,
                    "tracking_updates": "real-time",
                    "delivery_confirmation": "photo",
                    "customer_notifications": True
                }
                
            elif system_area.lower() == "inventory":
                settings = {
                    "reorder_calculation": "dynamic",
                    "cycle_count_frequency": "weekly",
                    "auto_replenishment": True,
                    "expiration_tracking": True,
                    "lot_tracking": True,
                    "storage_algorithm": "velocity-based"
                }
                
            elif system_area.lower() == "security":
                settings = {
                    "session_timeout": 30,
                    "password_expiry": 90,
                    "two_factor_auth": True,
                    "role_based_access": True,
                    "audit_logging": "detailed",
                    "physical_access_control": True
                }
                
            else:
                return f"Error: Unknown system area '{system_area}'. Valid areas are: picking, packing, shipping, inventory, security."
                
            for key, value in settings.items():
                result += f"{key.replace('_', ' ').title()}: {value}\n"
                
            return result
            
        elif action.lower() == "update":
            # Update system settings
            if parameters is None:
                return "Error: Parameters are required for update action."
                
            result = f"Updated {system_area.capitalize()} System Settings:\n\n"
            
            for key, value in parameters.items():
                result += f"{key.replace('_', ' ').title()}: {value}\n"
                
            result += "\nSettings successfully updated."
            return result
            
        elif action.lower() == "reset":
            # Reset system settings to defaults
            result = f"Reset {system_area.capitalize()} System Settings to Defaults\n\n"
            result += "Default settings have been restored."
            return result
            
        else:
            return f"Error: Unknown action '{action}'. Valid actions are: get, update, reset."
            
    except Exception as e:
        return f"Error managing system settings: {str(e)}"

# Create the tools
check_supplier_tool = create_tool(
    name="check_supplier",
    description="Check information about a supplier",
    function=check_supplier_func,
    arg_descriptions={
        "supplier_id": {
            "type": Optional[int], 
            "description": "Optional supplier ID to look up"
        },
        "name": {
            "type": Optional[str], 
            "description": "Optional supplier name to search for"
        },
        "product_category": {
            "type": Optional[str], 
            "description": "Optional product category to filter by"
        }
    }
)

vehicle_select_tool = create_tool(
    name="vehicle_select",
    description="Select an appropriate vehicle for delivery based on requirements",
    function=vehicle_select_func,
    arg_descriptions={
        "weight": {
            "type": float, 
            "description": "Total weight of the shipment in kg"
        },
        "volume": {
            "type": Optional[float], 
            "description": "Optional volume of the shipment in cubic meters"
        },
        "distance": {
            "type": Optional[float], 
            "description": "Optional delivery distance in km"
        },
        "refrigerated": {
            "type": Optional[bool], 
            "description": "Whether refrigeration is required"
        },
        "items_count": {
            "type": Optional[int], 
            "description": "Optional number of items"
        }
    }
)

worker_manage_tool = create_tool(
    name="worker_manage",
    description="Manage warehouse workers (manager only)",
    function=worker_manage_func,
    arg_descriptions={
        "action": {
            "type": str, 
            "description": "Action to perform (list, info, update, assign)"
        },
        "worker_id": {
            "type": Optional[int], 
            "description": "Optional worker ID"
        },
        "name": {
            "type": Optional[str], 
            "description": "Optional worker name"
        },
        "role": {
            "type": Optional[str], 
            "description": "Optional worker role"
        },
        "status": {
            "type": Optional[str], 
            "description": "Optional worker status"
        }
    }
)

check_analytics_tool = create_tool(
    name="check_analytics",
    description="Check warehouse analytics reports (manager only)",
    function=check_analytics_func,
    arg_descriptions={
        "report_type": {
            "type": str, 
            "description": "Type of report (inventory, orders, efficiency, workers)"
        },
        "period": {
            "type": Optional[str], 
            "description": "Optional time period (day, week, month, quarter, year)"
        },
        "category": {
            "type": Optional[str], 
            "description": "Optional category filter"
        }
    }
)

check_anomalies_tool = create_tool(
    name="check_anomalies",
    description="Check for anomalies in warehouse operations (manager only)",
    function=check_anomalies_func,
    arg_descriptions={
        "anomaly_type": {
            "type": str, 
            "description": "Type of anomaly to check (inventory, orders, returns, efficiency)"
        },
        "threshold": {
            "type": Optional[float], 
            "description": "Optional sensitivity threshold (1-100, higher means fewer anomalies)"
        },
        "period": {
            "type": Optional[str], 
            "description": "Optional time period (day, week, month)"
        }
    }
)

system_manage_tool = create_tool(
    name="system_manage",
    description="Manage warehouse system settings (manager only)",
    function=system_manage_func,
    arg_descriptions={
        "action": {
            "type": str, 
            "description": "Action to perform (get, update, reset)"
        },
        "system_area": {
            "type": str, 
            "description": "System area to manage (picking, packing, shipping, inventory, security)"
        },
        "parameters": {
            "type": Optional[Dict[str, Any]], 
            "description": "Optional parameters for the action"
        }
    }
)

# Export the tools
__all__ = [
    "check_supplier_tool",
    "vehicle_select_tool",
    "worker_manage_tool",
    "check_analytics_tool",
    "check_anomalies_tool",
    "system_manage_tool"
]