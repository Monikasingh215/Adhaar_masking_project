"""
API routes for the Image Processing API - Simplified for core requirements
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pathlib import Path
import asyncio
from app.models.schemas import ProcessFilesRequest


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
from app.utils.config_utils import is_cold_stop  

router = APIRouter()
logger = setup_logging()
batch_manager = BatchManager()
image_processor = ImageProcessor()

async def process_single_batch_with_retry(batch, batch_manager, image_processor, max_retries=3):
    """
    Process a single batch with retry logic.
    """
    for attempt in range(max_retries):
        try:
            original_file_paths = batch_manager.get_batch_file_paths(batch.batch_id)
            await image_processor.process_batch(batch, original_file_paths)
            return True
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id} (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2)  # wait before retrying

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
            process_single_batch_with_retry(batch, batch_manager, image_processor)
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

def get_files_by_date(date: str, directory: str):
    """
    Get all files in the specified directory that match the given date.
    Assumes files are organized or named in a way that allows filtering by date.
    """
    from pathlib import Path
    import re

    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []

    # Example: filter files by date in filename (customize as needed)
    date_pattern = re.compile(rf"{date}")
    return [
        str(file_path)
        for file_path in path.iterdir()
        if file_path.is_file() and date_pattern.search(file_path.name)
    ]

@router.post("/process-files/")
async def process_files_by_date_post(request: ProcessFilesRequest = Body(...)):
    """
    Trigger server-side processing for all files in the specified directory matching the given date (POST version).
    """
    file_paths = get_files_by_date(request.date, request.directory)
    if not file_paths:
        raise HTTPException(status_code=404, detail="No files found for the given date and directory.")
    batch_manager = BatchManager()
    image_processor = ImageProcessor()
    all_batches = await batch_manager.create_batches_from_files(file_paths, batch_size=request.batch_size)
    batch_ids = [batch.batch_id for batch in all_batches]
    # Prepare all batch processing coroutines
    batch_tasks = [
        process_single_batch_with_retry(batch, batch_manager, image_processor)
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
