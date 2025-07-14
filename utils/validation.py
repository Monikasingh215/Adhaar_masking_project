from pathlib import Path
from typing import Optional
from ..models.schemas import FileValidationResult, ImageFormat
import imghdr
import os

def validate_image_file(file_path: Path, max_size_bytes: int = 50 * 1024 * 1024) -> FileValidationResult:
    """Validate if the file is a supported image and within size limits."""
    if not file_path.exists():
        return FileValidationResult(
            filename=str(file_path),
            is_valid=False,
            error_message="File does not exist.",
            file_size_bytes=0
        )
    if not file_path.is_file():
        return FileValidationResult(
            filename=str(file_path),
            is_valid=False,
            error_message="Path is not a file.",
            file_size_bytes=0
        )
    file_size = file_path.stat().st_size
    if file_size > max_size_bytes:
        return FileValidationResult(
            filename=str(file_path),
            is_valid=False,
            error_message=f"File size exceeds limit ({max_size_bytes} bytes).",
            file_size_bytes=file_size
        )
    # Check format
    ext = file_path.suffix.lower().replace('.', '')
    try:
        fmt = ImageFormat(ext)
    except ValueError:
        fmt = None
    if fmt is None or fmt not in ImageFormat:
        return FileValidationResult(
            filename=str(file_path),
            is_valid=False,
            error_message=f"Unsupported file format: {ext}",
            file_size_bytes=file_size
        )
    return FileValidationResult(
        filename=str(file_path),
        is_valid=True,
        format_detected=fmt,
        file_size_bytes=file_size
    )

def validate_directory(directory_path: Path) -> bool:
    """Check if a directory exists and is readable."""
    return directory_path.exists() and directory_path.is_dir() and os.access(directory_path, os.R_OK)
