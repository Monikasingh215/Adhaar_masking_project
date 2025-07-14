"""
API routes for the Image Processing API - Simplified for core requirements
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Dict, Any

from app.models.schemas import (
    ProcessRequest, 
    ProcessResponse, 
    BatchStatus,
    FileValidationResult
)
from app.services.image_processor import ImageProcessor
from app.core.logging import setup_logging
from app.core.config import settings
from app.services.batch_manager import BatchManager
from app.utils.file_utils import FileUtils

router = APIRouter()
logger = setup_logging()
batch_manager = BatchManager()
image_processor = ImageProcessor()

# Global storage for processing status (in production, use Redis or database)
processing_status: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = Path("data/input")


@router.post("/process", response_model=ProcessResponse)
async def process_images(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process images in batches with format conversion and AI masking
    Core endpoint for your requirements
    """
    try:
        # Validate input
        if not request.input_directory:
            raise HTTPException(
                status_code=400, 
                detail="input_directory is required"
            )
        
        input_dir = Path(request.input_directory)
        if not input_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Input directory does not exist: {request.input_directory}"
            )
        
        # Create batches from directory grouped by upload date
        batches = await batch_manager.create_batches_from_directory(
            input_dir, 
            batch_size=request.batch_size or 20
        )
        
        if not batches:
            raise HTTPException(
                status_code=400,
                detail="No valid files found in the input directory"
            )
        
        batch_ids = [batch.batch_id for batch in batches]
        
        # Start processing for each batch in background with original file paths
        for batch in batches:
            original_file_paths = batch_manager.get_batch_file_paths(batch.batch_id)
            background_tasks.add_task(
                image_processor.process_batch, 
                batch, 
                original_file_paths
            )
            
        return ProcessResponse(
            success=True,
            message=f"Started processing {len(batches)} batches with {sum(len(batch.images) for batch in batches)} total files.",
            batch_ids=batch_ids
        )
        
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-status/{batch_id}", response_model=BatchStatus)
async def get_batch_status(batch_id: str):
    """Get the status of a specific batch"""
    status = await batch_manager.get_batch_status(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")
    return status


@router.get("/list-batches")
async def list_batches():
    """List all batches and their status"""
    return await batch_manager.list_batches()


@router.get("/validate-directory")
async def validate_directory(directory_path: str):
    """
    Validate a directory and return file information
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Directory does not exist: {directory_path}"
            )
        
        if not path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {directory_path}"
            )
        
        # Get all supported files
        files = []
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in settings.supported_formats:
                files.append(FileValidationResult(
                    filename=file_path.name,
                    is_valid=True,
                    file_size_bytes=file_path.stat().st_size,
                    format_detected=FileUtils._get_format_enum(file_path.suffix.lower())
                ))
        
        return {
            "directory": directory_path,
            "total_files": len(files),
            "supported_files": len(files),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"Error validating directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-info")
async def get_system_info():
    """
    Get system information and configuration
    """
    return {
        "api_version": "1.0.0",
        "batch_size": settings.batch_size,
        "max_workers": settings.max_workers,
        "supported_formats": settings.supported_formats,
        "jpeg_quality": settings.jpeg_quality,
        "ai_model_name": settings.ai_model_name
    }
