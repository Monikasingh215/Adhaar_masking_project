import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import BatchMetadata, ImageMetadata, ProcessingStatus, FileSource
from ..utils.file_utils import FileUtils
from app.services.converter import ImageConverter
from app.services.ai_simulator import simulate_ai_processing

logger = get_logger(__name__)

class MetadataManager:
    """Utility class for managing batch and image metadata."""
    
    @staticmethod
    def create_batch_metadata(batch_id: str, batch_number: int, image_files: List[Path]) -> BatchMetadata:
        """Create metadata for a batch of images with enhanced metadata preservation."""
        images_metadata = []
        
        for file_path in image_files:
            try:
                # Extract detailed metadata using enhanced FileUtils
                file_metadata = FileUtils.get_file_metadata(file_path)
                
                if file_metadata:
                    image_metadata = ImageMetadata(
                        original_filename=file_metadata['original_filename'],
                        original_format=file_metadata['original_format'],
                        original_size=file_metadata['original_size'],
                        original_dpi=file_metadata['original_dpi'],
                        color_mode=file_metadata['color_mode'],
                        compression_type=file_metadata['compression_type'],
                        file_size_bytes=file_metadata['file_size_bytes'],
                        upload_date=file_metadata['upload_date'],
                        source_path=file_metadata['source_path'],
                        source_type=file_metadata['source_type'],
                        original_bit_depth=file_metadata['original_bit_depth'],
                        original_alpha_channel=file_metadata['original_alpha_channel'],
                        original_transparency=file_metadata['original_transparency'],
                        original_palette=file_metadata['original_palette'],
                        original_icc_profile=file_metadata['original_icc_profile']
                    )
                    images_metadata.append(image_metadata)
                    
            except Exception as e:
                logger.error(f"Error creating metadata for {file_path}: {e}")
                continue
        
        batch_metadata = BatchMetadata(
            batch_id=batch_id,
            batch_number=batch_number,
            total_files=len(images_metadata),
            created_at=datetime.now(),
            processing_status=ProcessingStatus.PENDING,
            images=images_metadata,
            input_source=FileSource.LOCAL
        )
        
        return batch_metadata
    
    @staticmethod
    async def save_batch_metadata(batch_metadata: BatchMetadata) -> Path:
        """Save batch metadata to JSON file."""
        try:
            metadata_file = settings.metadata_dir / f"batch_{batch_metadata.batch_id}.json"
            
            # Convert to dict and handle datetime serialization
            metadata_dict = batch_metadata.dict()
            
            # Custom JSON encoder for datetime objects
            def json_encoder(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, default=json_encoder, ensure_ascii=False)
            
            logger.info(f"Saved batch metadata to {metadata_file}")
            return metadata_file
            
        except Exception as e:
            logger.error(f"Error saving batch metadata: {e}")
            raise
    
    @staticmethod
    async def load_batch_metadata(batch_id: str) -> Optional[BatchMetadata]:
        """Load batch metadata from JSON file."""
        try:
            metadata_file = settings.metadata_dir / f"batch_{batch_id}.json"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Convert string dates back to datetime objects
            def parse_datetime(date_str):
                if isinstance(date_str, str):
                    return datetime.fromisoformat(date_str)
                return date_str
            
            # Parse main batch dates
            if 'created_at' in metadata_dict:
                metadata_dict['created_at'] = parse_datetime(metadata_dict['created_at'])
            if 'processing_start_time' in metadata_dict and metadata_dict['processing_start_time']:
                metadata_dict['processing_start_time'] = parse_datetime(metadata_dict['processing_start_time'])
            if 'processing_end_time' in metadata_dict and metadata_dict['processing_end_time']:
                metadata_dict['processing_end_time'] = parse_datetime(metadata_dict['processing_end_time'])
            
            # Parse image dates
            if 'images' in metadata_dict:
                for image in metadata_dict['images']:
                    if 'upload_date' in image:
                        image['upload_date'] = parse_datetime(image['upload_date'])
            
            batch_metadata = BatchMetadata(**metadata_dict)
            return batch_metadata
            
        except Exception as e:
            logger.error(f"Error loading batch metadata for {batch_id}: {e}")
            return None
    
    @staticmethod
    async def update_batch_status(batch_id: str, status: ProcessingStatus, 
                                 error_message: Optional[str] = None) -> bool:
        """Update batch processing status."""
        try:
            batch_metadata = await MetadataManager.load_batch_metadata(batch_id)
            if not batch_metadata:
                logger.error(f"Could not load metadata for batch {batch_id}")
                return False
            
            batch_metadata.processing_status = status
            
            if status == ProcessingStatus.PROCESSING and not batch_metadata.processing_start_time:
                batch_metadata.processing_start_time = datetime.now()
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                batch_metadata.processing_end_time = datetime.now()
            
            if error_message:
                batch_metadata.errors.append(error_message)
            
            await MetadataManager.save_batch_metadata(batch_metadata)
            return True
            
        except Exception as e:
            logger.error(f"Error updating batch status for {batch_id}: {e}")
            return False
    
    @staticmethod
    async def update_image_metadata(batch_id: str, original_filename: str, 
                                   jpeg_filename: Optional[str] = None,
                                   jpeg_path: Optional[str] = None,
                                   processed_filename: Optional[str] = None,
                                   processed_path: Optional[str] = None) -> bool:
        """Update metadata for a specific image in a batch."""
        try:
            batch_metadata = await MetadataManager.load_batch_metadata(batch_id)
            if not batch_metadata:
                return False
            
            # Find the image metadata
            for image in batch_metadata.images:
                if image.original_filename == original_filename:
                    if jpeg_filename:
                        image.jpeg_filename = jpeg_filename
                    if jpeg_path:
                        image.jpeg_path = jpeg_path
                    if processed_filename:
                        image.processed_filename = processed_filename
                    if processed_path:
                        image.processed_path = processed_path
                    break
            
            await MetadataManager.save_batch_metadata(batch_metadata)
            return True
            
        except Exception as e:
            logger.error(f"Error updating image metadata for {batch_id}: {e}")
            return False
    
    @staticmethod
    async def get_image_metadata(batch_id: str, original_filename: str) -> Optional[ImageMetadata]:
        """Get metadata for a specific image."""
        try:
            batch_metadata = await MetadataManager.load_batch_metadata(batch_id)
            if not batch_metadata:
                return None
            
            for image in batch_metadata.images:
                if image.original_filename == original_filename:
                    return image
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting image metadata for {batch_id}: {e}")
            return None
    
    @staticmethod
    async def export_batch_metadata(batch_id: str, export_path: Path) -> bool:
        """Export batch metadata to a specific location."""
        try:
            batch_metadata = await MetadataManager.load_batch_metadata(batch_id)
            if not batch_metadata:
                return False
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict and handle datetime serialization
            metadata_dict = batch_metadata.dict()
            
            def json_encoder(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, default=json_encoder, ensure_ascii=False)
            
            logger.info(f"Exported batch metadata to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting batch metadata: {e}")
            return False

