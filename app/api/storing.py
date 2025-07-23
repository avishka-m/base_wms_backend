from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional

from ..auth.dependencies import get_current_active_user, has_role
from ..models.storing import StoringCreate, StoringUpdate, StoringResponse
from ..utils.database import get_collection

router = APIRouter()

# Get all storing jobs
@router.get("/", response_model=List[StoringResponse])
async def get_storing_jobs(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    assigned_worker_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all storing jobs with optional filtering.
    
    You can filter by status, task type, and assigned worker ID.
    """
    storing_collection = get_collection("storing_jobs")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if task_type:
        query["task_type"] = task_type
    if assigned_worker_id:
        query["assignedWorkerID"] = assigned_worker_id
    
    # Restrict pickers to only see their own storing jobs
    if current_user.get("role") == "Picker":
        query["assignedWorkerID"] = current_user.get("workerID")
    
    # Execute query with sorting by priority and creation date
    storing_jobs = list(storing_collection.find(query).sort([("priority", 1), ("created_date", 1)]).skip(skip).limit(limit))
    
    # Convert ObjectId to string and add computed properties for each job
    from datetime import datetime
    for job in storing_jobs:
        if job.get("_id"):
            job["_id"] = str(job["_id"])
        
        # Fix null datetime fields
        now = datetime.utcnow()
        if not job.get("created_date"):
            job["created_date"] = now
        if not job.get("created_at"):
            job["created_at"] = now
        if not job.get("updated_at"):
            job["updated_at"] = now
        
        # Add computed properties
        items = job.get("items", [])
        job["total_items"] = len(items)
        job["total_quantity"] = sum(item.get("quantity", 0) for item in items)
        job["stored_items"] = sum(1 for item in items if item.get("stored", False))
        
        # Calculate progress percentage
        if len(items) > 0:
            job["progress_percentage"] = (job["stored_items"] / len(items)) * 100
        else:
            job["progress_percentage"] = 0.0
    
    return storing_jobs

# Get storing job by ID
@router.get("/{storing_id}", response_model=StoringResponse)
async def get_storing_job(
    storing_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific storing job by ID.
    """
    storing_collection = get_collection("storing_jobs")
    
    storing_job = storing_collection.find_one({"storingID": storing_id})
    
    if not storing_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storing job with ID {storing_id} not found"
        )
    
    # Check if picker can only see their own jobs
    if current_user.get("role") == "Picker" and storing_job.get("assignedWorkerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own storing jobs"
        )
    
    # Convert ObjectId to string and add computed properties
    from datetime import datetime
    if storing_job.get("_id"):
        storing_job["_id"] = str(storing_job["_id"])
    
    # Fix null datetime fields
    now = datetime.utcnow()
    if not storing_job.get("created_date"):
        storing_job["created_date"] = now
    if not storing_job.get("created_at"):
        storing_job["created_at"] = now
    if not storing_job.get("updated_at"):
        storing_job["updated_at"] = now
    
    # Add computed properties
    items = storing_job.get("items", [])
    storing_job["total_items"] = len(items)
    storing_job["total_quantity"] = sum(item.get("quantity", 0) for item in items)
    storing_job["stored_items"] = sum(1 for item in items if item.get("stored", False))
    
    # Calculate progress percentage
    if len(items) > 0:
        storing_job["progress_percentage"] = (storing_job["stored_items"] / len(items)) * 100
    else:
        storing_job["progress_percentage"] = 0.0
    
    return storing_job

# Create new storing job
@router.post("/", response_model=StoringResponse, status_code=status.HTTP_201_CREATED)
async def create_storing_job(
    storing_data: StoringCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk"]))
) -> Dict[str, Any]:
    """
    Create a new storing job.
    
    Only managers and receiving clerks can create storing jobs.
    """
    from datetime import datetime
    storing_collection = get_collection("storing_jobs")
    
    # Get next storing ID
    last_storing = storing_collection.find_one(sort=[("storingID", -1)])
    next_storing_id = 1 if not last_storing else last_storing["storingID"] + 1
    
    # Prepare storing job data
    storing_job_data = storing_data.model_dump()
    storing_job_data["storingID"] = next_storing_id
    
    # Set proper datetime fields
    now = datetime.utcnow()
    storing_job_data["created_date"] = now
    storing_job_data["created_at"] = now
    storing_job_data["updated_at"] = now
    
    # Insert storing job
    result = storing_collection.insert_one(storing_job_data)
    
    # Return created storing job with proper formatting
    created_storing_job = storing_collection.find_one({"_id": result.inserted_id})
    
    # Add computed properties for response validation
    if created_storing_job:
        # Convert ObjectId to string for the _id field
        created_storing_job["_id"] = str(created_storing_job["_id"])
        
        # Add computed properties
        items = created_storing_job.get("items", [])
        created_storing_job["total_items"] = len(items)
        created_storing_job["total_quantity"] = sum(item.get("quantity", 0) for item in items)
        created_storing_job["stored_items"] = sum(1 for item in items if item.get("stored", False))
        
        # Calculate progress percentage
        if len(items) > 0:
            created_storing_job["progress_percentage"] = (created_storing_job["stored_items"] / len(items)) * 100
        else:
            created_storing_job["progress_percentage"] = 0.0
    return created_storing_job

# Update storing job
@router.put("/{storing_id}", response_model=StoringResponse)
async def update_storing_job(
    storing_id: int,
    storing_update: StoringUpdate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager", "ReceivingClerk", "Picker"]))
) -> Dict[str, Any]:
    """
    Update a storing job.
    
    Managers and receiving clerks can update any storing job.
    Pickers can only update their own storing jobs.
    """
    storing_collection = get_collection("storing_jobs")
    
    # Check if storing job exists
    storing_job = storing_collection.find_one({"storingID": storing_id})
    if not storing_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storing job with ID {storing_id} not found"
        )
    
    # Check permissions for pickers
    if current_user.get("role") == "Picker" and storing_job.get("assignedWorkerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own storing jobs"
        )
    
    # Check if already completed
    if storing_job.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update completed storing job"
        )
    
    # Prepare update data
    update_data = storing_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = storing_update.model_dump().get("updated_at")
    
    # Update storing job
    storing_collection.update_one(
        {"storingID": storing_id},
        {"$set": update_data}
    )
    
    # Return updated storing job
    updated_storing_job = storing_collection.find_one({"storingID": storing_id})
    return updated_storing_job

# Start storing job
@router.post("/{storing_id}/start", response_model=StoringResponse)
async def start_storing_job(
    storing_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Picker"]))
) -> Dict[str, Any]:
    """
    Start a storing job (for pickers).
    """
    storing_collection = get_collection("storing_jobs")
    
    # Check if storing job exists
    storing_job = storing_collection.find_one({"storingID": storing_id})
    if not storing_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storing job with ID {storing_id} not found"
        )
    
    # Check if picker is assigned to this job
    if storing_job.get("assignedWorkerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only start storing jobs assigned to you"
        )
    
    # Check if already started or completed
    if storing_job.get("status") in ["in_progress", "completed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Storing job is already {storing_job.get('status')}"
        )
    
    # Update status to in_progress and set start time
    from datetime import datetime
    storing_collection.update_one(
        {"storingID": storing_id},
        {"$set": {
            "status": "in_progress",
            "start_time": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }}
    )
    
    # Return updated storing job
    updated_storing_job = storing_collection.find_one({"storingID": storing_id})
    return updated_storing_job

# Complete storing job
@router.post("/{storing_id}/complete", response_model=StoringResponse)
async def complete_storing_job(
    storing_id: int,
    completion_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(has_role(["Picker"]))
) -> Dict[str, Any]:
    """
    Complete a storing job (for pickers).
    
    This will update inventory levels based on the stored items.
    """
    storing_collection = get_collection("storing_jobs")
    inventory_collection = get_collection("inventory")
    
    # Check if storing job exists
    storing_job = storing_collection.find_one({"storingID": storing_id})
    if not storing_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Storing job with ID {storing_id} not found"
        )
    
    # Check if picker is assigned to this job
    if storing_job.get("assignedWorkerID") != current_user.get("workerID"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only complete storing jobs assigned to you"
        )
    
    # Check if already completed
    if storing_job.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Storing job is already completed"
        )
    
    # Update inventory for each stored item
    from datetime import datetime
    for item in storing_job.get("items", []):
        # Update inventory stock level
        inventory_collection.update_one(
            {"itemID": item["itemID"]},
            {"$inc": {"stock_level": item["quantity"]}}
        )
        
        # Mark item as stored
        storing_collection.update_one(
            {"storingID": storing_id, "items.itemID": item["itemID"]},
            {"$set": {
                "items.$.stored": True,
                "items.$.actual_quantity": item["quantity"],
                "items.$.store_time": datetime.utcnow(),
                "items.$.notes": completion_data.get("notes", "")
            }}
        )
    
    # Update storing job status to completed
    storing_collection.update_one(
        {"storingID": storing_id},
        {"$set": {
            "status": "completed",
            "complete_time": datetime.utcnow(),
            "notes": completion_data.get("job_notes", ""),
            "updated_at": datetime.utcnow()
        }}
    )
    
    # Return updated storing job
    updated_storing_job = storing_collection.find_one({"storingID": storing_id})
    return updated_storing_job

# Get storing job statistics
@router.get("/stats/summary")
async def get_storing_stats(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get storing job statistics.
    """
    storing_collection = get_collection("storing_jobs")
    
    # Build base query for role restrictions
    base_query = {}
    if current_user.get("role") == "Picker":
        base_query["assignedWorkerID"] = current_user.get("workerID")
    
    # Get statistics
    total_jobs = storing_collection.count_documents(base_query)
    pending_jobs = storing_collection.count_documents({**base_query, "status": "pending"})
    in_progress_jobs = storing_collection.count_documents({**base_query, "status": "in_progress"})
    completed_jobs = storing_collection.count_documents({**base_query, "status": "completed"})
    
    return {
        "total_jobs": total_jobs,
        "pending": pending_jobs,
        "in_progress": in_progress_jobs,
        "completed": completed_jobs
    }
