from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ImageFormat(str, Enum):
    TIFF = "tiff"
    PDF = "pdf"
    PNG = "png"
    BMP = "bmp"
    JPEG = "jpeg"

class CompressionType(str, Enum):
    NONE = "none"
    TIFF_LZW = "tiff_lzw"
    TIFF_JPEG = "tiff_jpeg"
    TIFF_PACKBITS = "tiff_packbits"
    PNG_COMPRESSION = "png_compression"

class ColorMode(str, Enum):
    RGB = "RGB"
    RGBA = "RGBA"
    L = "L"  # Grayscale
    LA = "LA"  # Grayscale with alpha
    P = "P"  # Palette
    CMYK = "CMYK"

class FileSource(str, Enum):
    LOCAL = "local"
    DATABASE = "database"
    EXTERNAL = "external"

class ProcessRequest(BaseModel):
    """Request model for processing images - simplified for core requirements."""
    input_directory: str = Field(..., description="Path to input directory containing images")
    batch_size: Optional[int] = Field(20, ge=1, le=100, description="Number of images per batch")
    
    @validator('input_directory')
    def validate_input_directory(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Input directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Input path is not a directory: {v}")
        return str(path.absolute())

class ImageMetadata(BaseModel):
    """Metadata for a single image - focused on format restoration."""
    original_filename: str
    original_format: ImageFormat
    original_size: Tuple[int, int]  # (width, height)
    original_dpi: Tuple[int, int]   # (x_dpi, y_dpi)
    color_mode: ColorMode
    compression_type: Optional[CompressionType]
    file_size_bytes: int
    upload_date: datetime
    
    # Processing metadata
    jpeg_filenames: Optional[List[str]] = None  # Multi-page support

    processed_filename: Optional[str] = None
    
    # Essential metadata for format restoration
    original_bit_depth: Optional[int] = None
    original_alpha_channel: bool = False
    original_transparency: bool = False

        # --- NEW fields needed by pipeline ---
    source_path: Optional[str] = None
    source_type: Optional[FileSource] = None
    original_palette: Optional[list] = None
    original_icc_profile: Optional[str] = None
    jpeg_path: Optional[str] = None
    processed_path: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BatchMetadata(BaseModel):
    """Metadata for a batch of images - simplified."""
    batch_id: str
    batch_number: int
    total_files: int
    created_at: datetime
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    images: List[ImageMetadata] = []
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    errors: List[str] = []
    input_source: FileSource = FileSource.LOCAL
    # retry bookkeeping (used by BatchManager)
    retry_count: int = 0
    max_retries: int = 3

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessingStats(BaseModel):
    """Statistics for the processing operation."""
    total_files: int
    total_batches: int
    successful_conversions: int
    failed_conversions: int
    processing_time_seconds: float
    errors: List[str] = []

class ProcessResponse(BaseModel):
    """Response model for processing operation."""
    success: bool
    message: str
    batch_ids: List[str] = []
    stats: Optional[ProcessingStats] = None
    metadata_files: List[str] = []

class BatchStatus(BaseModel):
    """Status of a specific batch."""
    batch_id: str
    status: ProcessingStatus
    progress_percentage: float
    current_step: str
    files_processed: int
    total_files: int
    errors: List[str] = []

class AIProcessingResult(BaseModel):
    """Result from AI model processing (placeholder)."""
    image_filename: str
    processing_successful: bool
    confidence_score: Optional[float] = None
    detected_objects: List[str] = []
    processing_time_ms: float
    model_version: str
    
class FileValidationResult(BaseModel):
    """Result of file validation."""
    filename: str
    is_valid: bool
    format_detected: Optional[ImageFormat] = None
    error_message: Optional[str] = None
    file_size_bytes: int
    
class ConversionResult(BaseModel):
    """Result of image conversion."""
    original_file: str
    converted_file: str
    success: bool
    error_message: Optional[str] = None
    processing_time_ms: float
    size_change_bytes: int
    page_count: Optional[int] = 1  # Number of pages/frames converted

class FileUploadRequest(BaseModel):
    """Request model for uploading files from external sources."""
    file_paths: List[str] = Field(..., description="List of file paths to upload and process")
    source_type: FileSource = FileSource.EXTERNAL
    preserve_original: bool = Field(True, description="Whether to preserve original files")
    batch_size: int = Field(20, ge=1, le=100, description="Number of files per batch")

class ProcessFilesRequest(BaseModel):
    date: str
    directory: str
    batch_size: int = 20

