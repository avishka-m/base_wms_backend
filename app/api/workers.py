from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from bson import ObjectId

from ..auth.dependencies import get_current_active_user, has_role, get_password_hash
from ..auth.utils import create_new_user
from ..models.worker import WorkerCreate, WorkerUpdate, WorkerResponse
from ..utils.database import get_collection

router = APIRouter()

# Get all workers
@router.get("/", response_model=List[WorkerResponse])
async def get_workers(
    current_user: Dict[str, Any] = Depends(has_role(["Manager"])),
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all workers with optional role filtering.
    
    Only managers can access this endpoint.
    """
    workers_collection = get_collection("workers")
    
    # Build query
    query = {}
    if role:
        query["role"] = role
    
    # Execute query
    workers = list(workers_collection.find(query).skip(skip).limit(limit))
    return workers

# Get worker by ID
@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(
    worker_id: int,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get a specific worker by ID.
    
    Workers can view their own profile, Managers can view any profile.
    """
    # Check if user has permission to view this worker
    if current_user.get("role") != "Manager" and current_user.get("workerID") != worker_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this worker's information"
        )
    
    workers_collection = get_collection("workers")
    worker = workers_collection.find_one({"workerID": worker_id})
    
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker with ID {worker_id} not found"
        )
    
    return worker

# Create new worker
@router.post("/", response_model=WorkerResponse, status_code=status.HTTP_201_CREATED)
async def create_worker(
    worker: WorkerCreate,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Create a new worker.
    
    Only managers can create new workers.
    """
    workers_collection = get_collection("workers")
    
    # Check if username already exists
    if workers_collection.find_one({"username": worker.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username {worker.username} already registered"
        )
    
    # Find the next available workerID
    last_worker = workers_collection.find_one(
        sort=[("workerID", -1)]
    )
    next_id = 1
    if last_worker:
        next_id = last_worker.get("workerID", 0) + 1
    
    # Prepare worker document
    worker_data = worker.model_dump()
    worker_data.update({
        "workerID": next_id,
        "disabled": False
    })
    
    # Create new user
    user_doc = create_new_user(worker_data)
    
    # Add workerID to the user document
    user_doc["workerID"] = next_id
    
    # Insert worker to database
    result = workers_collection.insert_one(user_doc)
    
    # Return the created worker
    created_worker = workers_collection.find_one({"_id": result.inserted_id})
    return created_worker

# Update worker
@router.put("/{worker_id}", response_model=WorkerResponse)
async def update_worker(
    worker_id: int,
    worker_update: WorkerUpdate,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Update worker information.
    
    Workers can update their own profile, Managers can update any profile.
    Managers can change roles, normal workers cannot change their role.
    """
    # Check if user has permission to update this worker
    if current_user.get("role") != "Manager" and current_user.get("workerID") != worker_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this worker's information"
        )
    
    # Non-managers cannot change their own role
    if current_user.get("role") != "Manager" and worker_update.role is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to change roles"
        )
    
    workers_collection = get_collection("workers")
    worker = workers_collection.find_one({"workerID": worker_id})
    
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker with ID {worker_id} not found"
        )
    
    # Prepare update data
    update_data = {}
    for field, value in worker_update.model_dump(exclude_unset=True).items():
        if value is not None:
            if field == "password":
                update_data["hashed_password"] = get_password_hash(value)
            else:
                update_data[field] = value
    
    # Add updated_at timestamp
    update_data["updated_at"] = worker_update.model_dump().get("updated_at")
    
    # Update worker
    if update_data:
        workers_collection.update_one(
            {"workerID": worker_id},
            {"$set": update_data}
        )
    
    # Return updated worker
    updated_worker = workers_collection.find_one({"workerID": worker_id})
    return updated_worker

# Delete worker (deactivate)
@router.delete("/{worker_id}", response_model=Dict[str, Any])
async def delete_worker(
    worker_id: int,
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Deactivate a worker.
    
    Only managers can deactivate workers. This doesn't actually delete the worker
    but sets the disabled flag to True.
    """
    workers_collection = get_collection("workers")
    worker = workers_collection.find_one({"workerID": worker_id})
    
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker with ID {worker_id} not found"
        )
    
    # Deactivate worker
    workers_collection.update_one(
        {"workerID": worker_id},
        {"$set": {"disabled": True, "updated_at": worker.get("updated_at")}}
    )
    
    return {"message": f"Worker with ID {worker_id} has been deactivated"}