"""
FastAPI application for image processing with batch management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import threading
import time


from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router
from app.utils.startup_utils import ensure_directories
from app.utils.file_utils import delete_old_folders

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Image Processing API...")
    ensure_directories(settings)
    logger.info("Image Processing API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Image Processing API...")

# Initialize FastAPI app
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

# Include API routes
app.include_router(router, prefix="/api/v1")

# Serve static files (the UI)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def start_cleanup_background_job():
    def job():
        while True:
            # Example: Clean up folders older than 3 days in manual_qc and 2 days in data
            delete_old_folders('C:/python/aadhaar_masking_clone/data/manual_qc', days_old=3)
            delete_old_folders('C:/python/aadhaar_masking_clone/data', days_old=2)
            time.sleep(24 * 60 * 60)  # Run once a day

    t = threading.Thread(target=job, daemon=True)
    t.start()

# Call this function when your app starts
start_cleanup_background_job()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )