# app/utils/file_utils.py
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import mimetypes
from PIL import Image, TiffTags
import fitz  # PyMuPDF
from datetime import datetime, timedelta
import logging
import re


from app.core.config import settings
from app.core.logging import setup_logging
from app.models.schemas import FileValidationResult, ImageFormat, ColorMode, CompressionType, FileSource

logger = setup_logging()

class FileUtils:
    """Enhanced utility class for file operations with detailed metadata extraction."""
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            stat = file_path.stat()
            return {
                'filename': file_path.name,
                'full_path': str(file_path),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'extension': file_path.suffix.lower(),
                'mime_type': mimetypes.guess_type(str(file_path))[0]
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}
    
    @staticmethod
    def get_file_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract detailed metadata for format preservation."""
        try:
            if not file_path.exists():
                return None
            
            file_info = FileUtils.get_file_info(file_path)
            extension = file_info.get('extension', '').lower()
            
            # Basic metadata
            metadata = {
                'original_filename': file_path.name,
                'source_path': str(file_path.absolute()),
                'source_type': FileSource.LOCAL,
                'file_size_bytes': file_info.get('size', 0),
                'upload_date': file_info.get('created', datetime.now()),
                'original_format': FileUtils.get_format_enum(extension),
                'original_size': (0, 0),
                'original_dpi': settings.default_dpi,
                'color_mode': ColorMode.RGB,
                'compression_type': None,
                'original_bit_depth': None,
                'original_alpha_channel': False,
                'original_transparency': False,
                'original_palette': None,
                'original_icc_profile': None
            }
            
            # Extract detailed metadata based on format
            if extension == '.pdf':
                metadata.update(FileUtils._extract_pdf_metadata(file_path))
            elif extension in ['.tiff', '.tif', '.png', '.bmp', '.jpg', '.jpeg']:
                metadata.update(FileUtils._extract_image_metadata(file_path))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
            return None
    
    @staticmethod
    def _extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF files."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]  # Get first page
            
            # Get page dimensions and DPI
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Estimate DPI (PDFs don't have explicit DPI, use default)
            dpi = settings.default_dpi
            
            metadata = {
                'original_size': (int(width), int(height)),
                'original_dpi': dpi,
                'color_mode': ColorMode.RGB,  # PDFs are typically RGB
                'compression_type': None,
                'original_bit_depth': 8,
                'original_alpha_channel': False,
                'original_transparency': False
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {e}")
            return {
                'original_size': (0, 0),
                'original_dpi': settings.default_dpi,
                'color_mode': ColorMode.RGB,
                'compression_type': None
            }
    
    @staticmethod
    def _extract_image_metadata(image_path: Path) -> Dict[str, Any]:
        """Extract detailed metadata from image files."""
        try:
            with Image.open(image_path) as img:
                # Basic image info
                width, height = img.size
                mode = img.mode
                
                # Get DPI information
                dpi = img.info.get('dpi', settings.default_dpi)
                if isinstance(dpi, tuple) and len(dpi) == 2:
                    x_dpi, y_dpi = dpi
                else:
                    x_dpi = y_dpi = settings.default_dpi[0]
                
                # Determine color mode
                color_mode = FileUtils._get_color_mode(mode)
                
                # Get compression info
                compression = img.info.get('compression', None)
                compression_type = FileUtils._get_compression_type(compression, image_path.suffix.lower())
                
                # Get bit depth
                bit_depth = FileUtils._get_bit_depth(img)
                
                # Check for transparency/alpha
                has_alpha = 'A' in mode or mode in ['RGBA', 'LA', 'PA']
                has_transparency = has_alpha or (mode == 'P' and img.palette and img.palette.mode == 'RGBA')
                
                # Get palette if present
                palette = None
                if mode == 'P' and img.palette:
                    try:
                        palette = list(img.palette.getdata())
                    except:
                        palette = None
                
                # Get ICC profile
                icc_profile = None
                if 'icc_profile' in img.info:
                    icc_profile = img.info['icc_profile']
                
                metadata = {
                    'original_size': (width, height),
                    'original_dpi': (x_dpi, y_dpi),
                    'color_mode': color_mode,
                    'compression_type': compression_type,
                    'original_bit_depth': bit_depth,
                    'original_alpha_channel': has_alpha,
                    'original_transparency': has_transparency,
                    'original_palette': palette,
                    'original_icc_profile': icc_profile
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
            return {
                'original_size': (0, 0),
                'original_dpi': settings.default_dpi,
                'color_mode': ColorMode.RGB,
                'compression_type': None
            }
    
    @staticmethod
    def get_format_enum(extension: str) -> ImageFormat:
        """Convert file extension to ImageFormat enum."""
        format_map = {
            '.tiff': ImageFormat.TIFF,
            '.tif': ImageFormat.TIFF,
            '.pdf': ImageFormat.PDF,
            '.png': ImageFormat.PNG,
            '.bmp': ImageFormat.BMP,
            '.jpg': ImageFormat.JPEG,
            '.jpeg': ImageFormat.JPEG
        }
        return format_map.get(extension, ImageFormat.JPEG)
    
    @staticmethod
    def _get_color_mode(mode: str) -> ColorMode:
        """Convert PIL mode to ColorMode enum."""
        mode_map = {
            'RGB': ColorMode.RGB,
            'RGBA': ColorMode.RGBA,
            'L': ColorMode.L,
            'LA': ColorMode.LA,
            'P': ColorMode.P,
            'CMYK': ColorMode.CMYK
        }
        return mode_map.get(mode, ColorMode.RGB)
    
    @staticmethod
    def _get_compression_type(compression: Any, extension: str) -> Optional[CompressionType]:
        """Determine compression type."""
        if compression is None:
            return None
        
        compression_str = str(compression).lower()
        if 'lzw' in compression_str:
            return CompressionType.TIFF_LZW
        elif 'jpeg' in compression_str:
            return CompressionType.TIFF_JPEG
        elif 'packbits' in compression_str:
            return CompressionType.TIFF_PACKBITS
        elif extension == '.png':
            return CompressionType.PNG_COMPRESSION
        
        return None
    
    @staticmethod
    def _get_bit_depth(img: Image.Image) -> int:
        """Get image bit depth."""
        try:
            if img.mode in ['L', 'LA']:
                return 8
            elif img.mode in ['RGB', 'RGBA']:
                return 8
            elif img.mode == 'P':
                return 8
            elif img.mode == 'CMYK':
                return 8
            else:
                return 8
        except:
            return 8
    
    @staticmethod
    def validate_file(file_path: Path) -> FileValidationResult:
        """Validate if file is a supported image format."""
        try:
            if not file_path.exists():
                return FileValidationResult(
                    filename=file_path.name,
                    is_valid=False,
                    file_size_bytes=0,
                    error_message="File does not exist"
                )
            
            file_info = FileUtils.get_file_info(file_path)
            extension = file_info.get('extension', '').lower()
            
            if extension not in settings.supported_formats:
                return FileValidationResult(
                    filename=file_path.name,
                    is_valid=False,
                    file_size_bytes=file_info.get('size', 0),
                    error_message=f"Unsupported format: {extension}"
                )
            
            # Additional validation for specific formats
            if extension == '.pdf':
                is_valid = FileUtils._validate_pdf(file_path)
            elif extension in ['.tiff', '.tif', '.png', '.bmp', '.jpg', '.jpeg']:
                is_valid = FileUtils._validate_image(file_path)
            else:
                is_valid = True
            
            return FileValidationResult(
                filename=file_path.name,
                is_valid=is_valid,
                file_size_bytes=file_info.get('size', 0),
                format_detected=FileUtils.get_format_enum(extension),
                error_message=None if is_valid else "Invalid file format"
            )
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return FileValidationResult(
                filename=file_path.name,
                is_valid=False,
                file_size_bytes=0,
                error_message=str(e)
            )
    
    @staticmethod
    def _validate_pdf(file_path: Path) -> bool:
        """Validate PDF file."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception:
            return False
    
    @staticmethod
    def _validate_image(file_path: Path) -> bool:
        """Validate image file using PIL."""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_files_by_date(directory: Path, extensions: List[str] = None) -> Dict[str, List[Path]]:
        """Group files by their creation date."""
        if extensions is None:
            extensions = settings.supported_formats
        
        files_by_date = {}
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    file_info = FileUtils.get_file_info(file_path)
                    date_key = file_info.get('created', datetime.now()).strftime('%Y-%m-%d')
                    
                    if date_key not in files_by_date:
                        files_by_date[date_key] = []
                    files_by_date[date_key].append(file_path)
            
            # Sort files within each date group by creation time
            for date_key in files_by_date:
                files_by_date[date_key].sort(key=lambda x: FileUtils.get_file_info(x).get('created', datetime.now()))
                
        except Exception as e:
            logger.error(f"Error grouping files by date: {e}")
        
        return files_by_date
    
    @staticmethod
    def get_files_by_upload_date(directory: Path, extensions: List[str] = None) -> Dict[str, List[Path]]:
        """Group files by their upload (creation) date as YYYY-MM-DD."""
        return FileUtils.get_files_by_date(directory, extensions)
    
    @staticmethod
    def create_batches(files: List[Path], batch_size: int) -> List[List[Path]]:
        """Create batches of files."""
        batches = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batches.append(batch)
        return batches
    
    @staticmethod
    def copy_file_to_storage(source_path: Path, destination_dir: Path, preserve_original: bool = True) -> Path:
        """Copy file to storage location."""
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / source_path.name
            
            if preserve_original:
                shutil.copy2(source_path, destination_path)
            else:
                shutil.move(str(source_path), str(destination_path))
            
            return destination_path
            
        except Exception as e:
            logger.error(f"Error copying file {source_path}: {e}")
            raise
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    @staticmethod
    def safe_copy(source: Path, destination: Path) -> bool:
        """Safely copy file with error handling."""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            logger.error(f"Error copying {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def safe_move(source: Path, destination: Path) -> bool:
        """Safely move file with error handling."""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            return True
        except Exception as e:
            logger.error(f"Error moving {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def cleanup_directory(directory: Path, pattern: str = "*") -> bool:
        """Clean up directory by removing files matching pattern."""
        try:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            return True
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory}: {e}")
            return False
    
    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """Get total size of directory in bytes."""
        try:
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
            return 0

def get_files_by_date(date: str, search_dir: str) -> List[Path]:
    """
    Find all files in search_dir (recursively) with creation date matching the given date (YYYY-MM-DD).
    """
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    files = []
    for file in Path(search_dir).rglob("*"):
        if file.is_file():
            file_date = datetime.fromtimestamp(file.stat().st_ctime).date()
            if file_date == target_date:
                files.append(file)
    return files

def save_uploaded_files(files, input_dir):
    saved_files = []
    file_paths_by_date = {}
    for file in files:
        if hasattr(file, 'webkitRelativePath') and file.webkitRelativePath:
            relative_path = file.webkitRelativePath
            dest_path = input_dir / relative_path
        else:
            dest_path = input_dir / file.filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(dest_path)
        try:
            file_info = dest_path.stat()
            date_key = file_info.st_ctime
        except Exception:
            import time
            date_key = time.time()
        file_paths_by_date.setdefault(date_key, []).append(dest_path)
    return saved_files, file_paths_by_date


def delete_old_folders(base_dir, days_old):
    """
    Delete folders in base_dir older than days_old.
    """
    import re
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    now = datetime.now().date()
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            if date_pattern.fullmatch(folder):
                try:
                    folder_date = datetime.strptime(folder, "%Y-%m-%d").date()
                    if (now - folder_date).days > days_old:
                        shutil.rmtree(folder_path)
                        logging.info(f"Deleted old folder: {folder_path}")
                except Exception as e:
                    logging.warning(f"Skipping {folder_path}: {e}")
            else:
                # Not a date-formatted folder, skip without warning
                continue

def cleanup_files_in_batch(dirs, filenames):
    """
    Remove files with given filenames from each directory in dirs.
    """
    for dir_path in dirs:
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.warning(f"Could not delete {file_path}: {e}")