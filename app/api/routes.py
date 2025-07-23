"""
API routes for the Image Processing API - Simplified for core requirements
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pathlib import Path
import asyncio
from app.models.schemas import ProcessFilesRequest

from uuid import uuid4
import shutil
from typing import List

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
from app.utils.file_utils import FileUtils, save_uploaded_files
from app.utils.config_utils import (
    is_cold_stop,
    get_max_retries,
    get_retry_delay,
    get_batch_size,      
)

router = APIRouter()
logger = setup_logging()
batch_manager = BatchManager()
image_processor = ImageProcessor()


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Image Processing API"
    }

@router.post("/process-upload")
async def process_upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    batch_size: int = Form(None)
):
    """
    Upload a folder of files and automatically create batches for processing.
    Handles folder structure and creates batches of specified size.
    """
    # Cold stop check
    if is_cold_stop():
        return JSONResponse(
            status_code=503,
            content={"error": "Processing is currently paused by admin (cold stop)."}
        )

    # Use batch size from config if not provided
    if batch_size is None:
        batch_size = get_batch_size()

    try:
        # Create upload directory
        input_dir = Path("data/input/web_upload")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files, file_paths_by_date = save_uploaded_files(files, input_dir)
        
        logger.info(f"Uploaded {len(saved_files)} files")
        
        # Create batches from the uploaded files
        all_batches = []
        total_files_processed = 0
        
        for date_key, file_paths in file_paths_by_date.items():
            logger.info(f"Processing {len(file_paths)} files from date {date_key}")
            # Create batches for this date group
            batches = await batch_manager.create_batches_from_files(file_paths, batch_size=batch_size)
            all_batches.extend(batches)
            total_files_processed += len(file_paths)
            
            logger.info(f"Created {len(batches)} batches for date {date_key}")
        
        # Prepare all batch processing coroutines
        batch_tasks = [
            batch_manager.process_single_batch_with_retry(batch, image_processor)
            for batch in all_batches
        ]

        batch_ids = [batch.batch_id for batch in all_batches]

        # Run all batch processing coroutines in parallel
        await asyncio.gather(*batch_tasks)

        logger.info(f"Started processing {len(all_batches)} batches with {total_files_processed} total files")
        
        return {
            "message": f"Successfully uploaded {len(saved_files)} files and started processing {len(all_batches)} batches",
            "total_files": len(saved_files),
            "total_batches": len(all_batches),
            "batch_size": batch_size,
            "batch_ids": batch_ids,
            "status": "processing_started"
        }
        
    except Exception as e:
        logger.error(f"Error in process_upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


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
                    format_detected=FileUtils.get_format_enum(file_path.suffix.lower())
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

from datetime import datetime

def get_files_by_date(date: str, directory: str):
    from pathlib import Path

    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []

    try:
        # Parse MM/DD/YYYY to datetime.date
        target_date = datetime.strptime(date, "%m/%d/%Y").date()
    except ValueError:
        return []

    matching_files = []
    for file_path in path.iterdir():
        if file_path.is_file():
            modified_date = datetime.fromtimestamp(file_path.stat().st_mtime).date()
            if modified_date == target_date:
                matching_files.append(str(file_path))

    return matching_files

"""
@router.post("/process-files/")
async def process_files_by_date_post(request: ProcessFilesRequest = Body(...)):
    
    #Trigger server-side processing for all files in the specified directory matching the given date (POST version).
    
    from pathlib import Path
    str_paths = get_files_by_date(request.date, request.directory)
    file_paths = [Path(p) for p in str_paths]  # ✅ Convert to Path

    if not file_paths:
        raise HTTPException(status_code=404, detail="No files found for the given date and directory.")
    batch_manager = BatchManager()
    image_processor = ImageProcessor()
    all_batches = await batch_manager.create_batches_from_files(file_paths, batch_size=request.batch_size)
    batch_ids = [batch.batch_id for batch in all_batches]
    # Prepare all batch processing coroutines
    batch_tasks = [
    batch_manager.process_single_batch_with_retry(batch, image_processor)
    for batch in all_batches
    ]

    # Run all batch processing coroutines in parallel
    await asyncio.gather(*batch_tasks)
    return {
        "message": f"Processed {len(file_paths)} files in {len(all_batches)} batches for date {request.date}",
        "files_processed": len(file_paths),
        "batches": len(all_batches),
        "batch_ids": batch_ids
    }
"""

@router.post("/process-files/")
async def process_files_by_date_post(request: ProcessFilesRequest = Body(...)):
    """
    Trigger processing of files saved on the specified date within the given directory (recursively).
    Files are grouped by filesystem-modified date. Batches are created and processed asynchronously.
    """
    try:
        input_dir = Path(request.directory)
        if not input_dir.exists():
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")

        try:
            target_date = datetime.strptime(request.date, "%m/%d/%Y").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use MM/DD/YYYY.")

        matching_files = []
        for file_path in input_dir.rglob("*"):
            if file_path.is_file():
                modified_date = datetime.fromtimestamp(file_path.stat().st_mtime).date()
                if modified_date == target_date:
                    matching_files.append(file_path)

        if not matching_files:
            raise HTTPException(status_code=404, detail="No files found for the given date and directory.")

        logger.info(f"[process-files] Found {len(matching_files)} files for {request.date}")
        for f in matching_files:
            logger.info(f"[process-files] File: {f}")

        # Optional: Copy to staging area `data/input/` for temporary processing (as per your design)
        staging_dir = Path("data/input")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_files = []

        for src_file in matching_files:
            dest_file = staging_dir / src_file.name
            shutil.copy2(src_file, dest_file)
            staged_files.append(dest_file)

        # Use provided or default batch size
        batch_size = request.batch_size or get_batch_size()

        # Create batches from the staged files
        all_batches = await batch_manager.create_batches_from_files(staged_files, batch_size=batch_size)
        if not all_batches:
            raise HTTPException(status_code=500, detail="Failed to create batches from selected files.")

        logger.info(f"[process-files] Created {len(all_batches)} batches with batch size {batch_size}")

        # Process all batches asynchronously
        batch_tasks = [
            batch_manager.process_single_batch_with_retry(batch, image_processor)
            for batch in all_batches
        ]
        await asyncio.gather(*batch_tasks)

        return {
            "message": f"Processed {len(staged_files)} files in {len(all_batches)} batches for date {request.date}",
            "files_processed": len(staged_files),
            "batches": len(all_batches),
            "batch_ids": [batch.batch_id for batch in all_batches],
        }

    except Exception as e:
        logger.error(f"[process-files] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
