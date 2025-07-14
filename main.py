"""
FastAPI application for image processing with batch management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
import os
import shutil
from datetime import datetime
from pydantic import BaseModel
import asyncio

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router
from app.services.batch_manager import BatchManager
from app.services.image_processor import ImageProcessor
from app.utils.file_utils import get_files_by_date
from utils.config_utils import read_json_config

# Setup logging
logger = setup_logging()

def is_cold_stop():
    config = read_json_config("aadhaar_stop.json")
    return config.get("cold_stop", 0) == 1

def get_batch_size():
    config = read_json_config("aadhaar_start.json")
    return config.get("batch_size", 20)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Image Processing API...")
    
    # Create necessary directories
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Image Processing API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Image Processing API...")

# Create FastAPI app
app = FastAPI(
    title="Image Processing API",
    description="API for batch image processing with format conversion and AI masking",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for general errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Image Processing API"
    }

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Image Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Serve static files (the UI)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse("app/static/index.html")

@app.post("/process-upload")
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
        
        # Save uploaded files maintaining folder structure
        saved_files = []
        file_paths_by_date = {}  # Group files by date for batch creation
        
        for file in files:
            # Handle folder structure from webkitRelativePath
            if hasattr(file, 'webkitRelativePath') and file.webkitRelativePath:
                # This is a folder upload - maintain structure
                relative_path = file.webkitRelativePath
                dest_path = input_dir / relative_path
            else:
                # Single file upload
                dest_path = input_dir / file.filename
            
            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            saved_files.append(dest_path)
            
            # Get file creation date for grouping
            try:
                file_info = dest_path.stat()
                # Use creation time for grouping
                date_key = file_info.st_ctime
                if date_key not in file_paths_by_date:
                    file_paths_by_date[date_key] = []
                file_paths_by_date[date_key].append(dest_path)
            except Exception as e:
                logger.warning(f"Could not get file date for {dest_path}: {e}")
                # Fallback: use current time
                import time
                current_time = time.time()
                if current_time not in file_paths_by_date:
                    file_paths_by_date[current_time] = []
                file_paths_by_date[current_time].append(dest_path)
        
        logger.info(f"Uploaded {len(saved_files)} files")
        
        # Initialize batch manager and image processor
        batch_manager = BatchManager()
        image_processor = ImageProcessor()
        
        # Create batches from the uploaded files
        all_batches = []
        total_files_processed = 0
        
        for date_key, file_paths in file_paths_by_date.items():
            logger.info(f"Processing {len(file_paths)} files from date {date_key}")
            
            # Create batches for this date group
            batches = await batch_manager.create_batches_from_files(
                file_paths, 
                batch_size=batch_size
            )
            
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

class ProcessFilesRequest(BaseModel):
    date: str
    directory: str
    batch_size: int = 20

# Deprecated: GET /process-files/ (use POST instead)
# @app.get("/process-files/")
# async def process_files_by_date(
#     date: str = Query(..., example="2025-06-30"),
#     directory: str = Query(..., example="C:/your/folder/path"),
#     batch_size: int = Query(20, description="Files per batch")
# ):
#     """
#     Trigger server-side processing for all files in the specified directory matching the given date.
#     """
#     file_paths = get_files_by_date(date, directory)
#     if not file_paths:
#         raise HTTPException(status_code=404, detail="No files found for the given date and directory.")
#     batch_manager = BatchManager()
#     image_processor = ImageProcessor()
#     all_batches = await batch_manager.create_batches_from_files(file_paths, batch_size=batch_size)
#     batch_ids = []
#     for batch in all_batches:
#         original_file_paths = batch_manager.get_batch_file_paths(batch.batch_id)
#         await image_processor.process_batch(batch, original_file_paths)
#         batch_ids.append(batch.batch_id)
#     return {
#         "message": f"Processed {len(file_paths)} files in {len(all_batches)} batches for date {date}",
#         "files_processed": len(file_paths),
#         "batches": len(all_batches),
#         "batch_ids": batch_ids
#     }

@app.post("/process-files/")
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

async def process_single_batch_with_retry(batch_metadata, batch_manager, image_processor):
    while batch_metadata.retry_count < batch_metadata.max_retries:
        result = await image_processor.process_batch(batch_metadata)
        if result and result.processing_status == "COMPLETED":
            print(f"Batch {batch_metadata.batch_id} processed successfully.")
            break
        else:
            batch_metadata.retry_count += 1
            print(f"Batch {batch_metadata.batch_id} failed. Retry {batch_metadata.retry_count}/{batch_metadata.max_retries}")
            # Optionally, update metadata in storage here
            await asyncio.sleep(2)  # Optional: wait before retrying
    else:
        print(f"Batch {batch_metadata.batch_id} failed after {batch_metadata.max_retries} retries.")
        # Optionally, log/report this batch as permanently failed

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )