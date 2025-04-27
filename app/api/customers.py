from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.customer import CustomerCreate, CustomerUpdate, CustomerResponse
from ..utils.database import get_collection

router = APIRouter()

# Get all customers
@router.get("/", response_model=List[CustomerResponse])
async def get_customers(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"])),
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all customers with optional filtering.
    
    Only managers can view all customers.
    """
    customer_collection = get_collection("customers")
    
    # Build query
    query = {}
    if name:
        query["name"] = {"$regex": name, "$options": "i"}  # Case-insensitive search
    
    # Execute query
    customers = list(customer_collection.find(query).skip(skip).limit(limit))
    return customers

# Get customer by ID
@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific customer by ID.
    
    Customers can only view their own profile. Staff can view any customer.
    """
    customer_collection = get_collection("customers")
    customer = customer_collection.find_one({"customerID": customer_id})
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )
    
    # Check if user has permission to view this customer
    if current_user.get("role") == "Customer" and current_user.get("customerID") != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this customer's information"
        )
    
    return customer

# Create new customer
@router.post("/", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(
    customer: CustomerCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new customer.
    
    Only managers can create customers.
    """
    customer_collection = get_collection("customers")
    
    # Check if email already exists
    if customer_collection.find_one({"email": customer.email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Customer with email {customer.email} already exists"
        )
    
    # Find the next available customerID
    last_customer = customer_collection.find_one(
        sort=[("customerID", -1)]
    )
    next_id = 1
    if last_customer:
        next_id = last_customer.get("customerID", 0) + 1
    
    # Prepare customer document
    customer_data = customer.model_dump()
    customer_data.update({
        "customerID": next_id,
        "created_at": customer_data.get("created_at", None),
        "updated_at": customer_data.get("updated_at", None)
    })
    
    # Insert customer to database
    result = customer_collection.insert_one(customer_data)
    
    # Return the created customer
    created_customer = customer_collection.find_one({"_id": result.inserted_id})
    return created_customer

# Update customer
@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: int,
    customer_update: CustomerUpdate,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Update a customer.
    
    Customers can only update their own profile. Staff can update any customer.
    """
    customer_collection = get_collection("customers")
    
    # Check if customer exists
    customer = customer_collection.find_one({"customerID": customer_id})
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )
    
    # Check if user has permission to update this customer
    if current_user.get("role") == "Customer" and current_user.get("customerID") != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this customer's information"
        )
    
    # Check if email is being updated and already exists
    if customer_update.email and customer_update.email != customer.get("email"):
        if customer_collection.find_one({"email": customer_update.email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Customer with email {customer_update.email} already exists"
            )
    
    # Prepare update data
    update_data = customer_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = customer_update.model_dump().get("updated_at")
    
    # Update customer
    customer_collection.update_one(
        {"customerID": customer_id},
        {"$set": update_data}
    )
    
    # Return updated customer
    updated_customer = customer_collection.find_one({"customerID": customer_id})
    return updated_customer

# Delete customer
@router.delete("/{customer_id}", response_model=Dict[str, Any])
async def delete_customer(
    customer_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Delete a customer.
    
    Only managers can delete customers.
    """
    customer_collection = get_collection("customers")
    orders_collection = get_collection("orders")
    
    # Check if customer exists
    customer = customer_collection.find_one({"customerID": customer_id})
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )
    
    # Check if customer has any orders
    orders = list(orders_collection.find({"customerID": customer_id}))
    if orders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete customer with ID {customer_id} because they have {len(orders)} orders"
        )
    
    # Delete customer
    customer_collection.delete_one({"customerID": customer_id})
    
    return {"message": f"Customer with ID {customer_id} has been deleted"}

# Get customer orders
@router.get("/{customer_id}/orders", response_model=List[Dict[str, Any]])
async def get_customer_orders(
    customer_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all orders for a specific customer.
    
    Customers can only view their own orders. Staff can view any customer's orders.
    """
    customer_collection = get_collection("customers")
    orders_collection = get_collection("orders")
    
    # Check if customer exists
    customer = customer_collection.find_one({"customerID": customer_id})
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found"
        )
    
    # Check if user has permission to view this customer's orders
    if current_user.get("role") == "Customer" and current_user.get("customerID") != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this customer's orders"
        )
    
    # Build query
    query = {"customerID": customer_id}
    if status:
        query["order_status"] = status
    
    # Execute query
    orders = list(orders_collection.find(query).sort([("order_date", -1)]).skip(skip).limit(limit))
    return orders