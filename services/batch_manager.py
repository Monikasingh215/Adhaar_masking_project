import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import BatchMetadata, ProcessingStatus, FileSource
from ..utils.file_utils import FileUtils
from ..utils.metadata_utils import MetadataManager
import json

logger = get_logger(__name__)

class BatchManager:
    """Manages batch creation and processing workflow."""
    
    def __init__(self):
        self.active_batches: Dict[str, BatchMetadata] = {}
        self.batch_file_paths: Dict[str, List[Path]] = {}  # Store original file paths for each batch
    
    async def create_batches_from_directory(self, input_dir: Path, 
                                          batch_size: int = 20) -> List[BatchMetadata]:
        """Create batches from images in a directory, grouped by upload date."""
        logger.info(f"Creating batches from directory: {input_dir}")
        
        try:
            # Get files grouped by upload date
            files_by_date = FileUtils.get_files_by_upload_date(input_dir)
            
            all_batches = []
            batch_counter = 1
            unsupported_files = []  # Collect unsupported files for logging and manifest
            
            for date, files in files_by_date.items():
                logger.info(f"Processing {len(files)} files from {date}")
                
                # Filter supported files
                supported_files = []
                for file_path in files:
                    validation_result = FileUtils.validate_file(file_path)
                    if validation_result.is_valid:
                        supported_files.append(file_path)
                    else:
                        unsupported_files.append({
                            "file": str(file_path),
                            "reason": validation_result.error_message
                        })
                        logger.warning(f"Skipping unsupported file {file_path}: {validation_result.error_message}")
                
                # Create batches for this date
                file_batches = FileUtils.create_batches(supported_files, batch_size)
                
                for batch_files in file_batches:
                    batch_id = f"batch_{datetime.now().strftime('%Y%m%d')}_{batch_counter:03d}"
                    
                    # Create batch metadata
                    batch_metadata = MetadataManager.create_batch_metadata(
                        batch_id, batch_counter, batch_files
                    )
                    
                    # Store original file paths for this batch
                    self.batch_file_paths[batch_id] = batch_files
                    
                    # Save metadata
                    await MetadataManager.save_batch_metadata(batch_metadata)
                    
                    # Add to active batches
                    self.active_batches[batch_id] = batch_metadata
                    all_batches.append(batch_metadata)
                    
                    batch_counter += 1
                    
                    logger.info(f"Created batch {batch_id} with {len(batch_files)} files")
            
            if unsupported_files:
                logger.info(f"Total unsupported files skipped: {len(unsupported_files)}")
                for entry in unsupported_files:
                    logger.info(f"Unsupported file: {entry['file']} - Reason: {entry['reason']}")
                # Write manifest to logs directory
                manifest_path = settings.logs_dir / f"unsupported_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(unsupported_files, f, indent=2)
                logger.info(f"Unsupported files manifest written to: {manifest_path}")
            
            logger.info(f"Created {len(all_batches)} batches total")
            return all_batches
            
        except Exception as e:
            logger.error(f"Error creating batches: {e}")
            raise
    
    def get_batch_file_paths(self, batch_id: str) -> Optional[List[Path]]:
        """Get the original file paths for a batch."""
        return self.batch_file_paths.get(batch_id)
    
    async def create_batches_from_files(self, file_paths: List[Path], 
                                       batch_size: int = 20,
                                       preserve_metadata: bool = True) -> List[BatchMetadata]:
        """Create batches from specific file paths."""
        logger.info(f"Creating batches from {len(file_paths)} specific files")
        
        try:
            # Validate files
            valid_files = []
            for file_path in file_paths:
                validation_result = FileUtils.validate_file(file_path)
                if validation_result.is_valid:
                    valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping invalid file {file_path}: {validation_result.error_message}")
            
            if not valid_files:
                raise ValueError("No valid files found for processing")
            
            # Create batches
            file_batches = FileUtils.create_batches(valid_files, batch_size)
            
            all_batches = []
            batch_counter = 1
            
            for batch_files in file_batches:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d')}_{batch_counter:03d}"
                
                # Create batch metadata with enhanced metadata preservation
                batch_metadata = MetadataManager.create_batch_metadata(
                    batch_id, batch_counter, batch_files
                )
                
                # Set source type
                batch_metadata.input_source = FileSource.EXTERNAL
                
                # Store original file paths for this batch
                self.batch_file_paths[batch_id] = batch_files
                
                # Save metadata
                await MetadataManager.save_batch_metadata(batch_metadata)
                
                # Add to active batches
                self.active_batches[batch_id] = batch_metadata
                all_batches.append(batch_metadata)
                
                batch_counter += 1
                
                logger.info(f"Created batch {batch_id} with {len(batch_files)} files")
            
            logger.info(f"Created {len(all_batches)} batches from {len(valid_files)} files")
            return all_batches
            
        except Exception as e:
            logger.error(f"Error creating batches from files: {e}")
            raise
    
    async def create_batches_from_database(self, database_ids: List[str], 
                                          batch_size: int = 20) -> List[BatchMetadata]:
        """Create batches from database IDs (placeholder for future implementation)."""
        logger.info(f"Creating batches from {len(database_ids)} database IDs")
        
        try:
            # TODO: Implement database integration
            # For now, return empty list
            logger.warning("Database integration not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error creating batches from database: {e}")
            raise
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get the current status of a batch."""
        try:
            if batch_id in self.active_batches:
                batch_metadata = self.active_batches[batch_id]
            else:
                batch_metadata = await MetadataManager.load_batch_metadata(batch_id)
            
            if not batch_metadata:
                return None
            
            # Calculate progress
            total_files = batch_metadata.total_files
            
            # Count processed files (those with processed_filename)
            processed_files = sum(1 for img in batch_metadata.images if img.processed_filename)
            
            progress_percentage = (processed_files / total_files * 100) if total_files > 0 else 0
            
            # Determine current step
            current_step = "pending"
            if batch_metadata.processing_status == ProcessingStatus.PROCESSING:
                # Check how many files have been through each step
                jpeg_converted = sum(1 for img in batch_metadata.images if img.jpeg_filename)
                
                if jpeg_converted == 0:
                    current_step = "converting_to_jpeg"
                elif jpeg_converted == total_files and processed_files == 0:
                    current_step = "ai_processing"
                elif processed_files > 0:
                    current_step = "converting_back_to_original"
                else:
                    current_step = "processing"
            elif batch_metadata.processing_status == ProcessingStatus.COMPLETED:
                current_step = "completed"
            elif batch_metadata.processing_status == ProcessingStatus.FAILED:
                current_step = "failed"
            else:
                current_step = str(batch_metadata.processing_status).lower()

            return {
                "batch_id": batch_metadata.batch_id,
                "status": batch_metadata.processing_status.value if hasattr(batch_metadata.processing_status, 'value') else str(batch_metadata.processing_status),
                "current_step": current_step,
                "progress": progress_percentage,
                "total_files": total_files,
                "processed_files": processed_files,
                "input_source": batch_metadata.input_source,
                "created_at": batch_metadata.created_at.isoformat() if batch_metadata.created_at else None
            }
        except Exception as e:
            logger.error(f"Error getting batch status for {batch_id}: {e}")
            return None
    
    async def list_batches(self) -> Dict:
        """List all batches and their status."""
        try:
            batches = []
            
            # Get active batches
            for batch_id, batch_metadata in self.active_batches.items():
                status = await self.get_batch_status(batch_id)
                if status:
                    batches.append(status)
            
            # Get completed batches from metadata directory
            metadata_files = list(settings.metadata_dir.glob("batch_*.json"))
            for metadata_file in metadata_files:
                batch_id = metadata_file.stem.replace("batch_", "")
                if batch_id not in self.active_batches:
                    status = await self.get_batch_status(batch_id)
                    if status:
                        batches.append(status)
            
            return {
                "batches": batches,
                "total_batches": len(batches),
                "active_batches": len([b for b in batches if b.get("status") in ["pending", "processing"]])
            }
            
        except Exception as e:
            logger.error(f"Error listing batches: {e}")
            return {"batches": [], "total_batches": 0, "active_batches": 0}
    
    async def get_batch_metadata(self, batch_id: str) -> Optional[BatchMetadata]:
        """Get full batch metadata."""
        try:
            if batch_id in self.active_batches:
                return self.active_batches[batch_id]
            else:
                return await MetadataManager.load_batch_metadata(batch_id)
        except Exception as e:
            logger.error(f"Error getting batch metadata for {batch_id}: {e}")
            return None
    
    async def cleanup_completed_batches(self, older_than_days: int = 7) -> int:
        """Clean up completed batches older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cleaned_count = 0
            
            # Clean up active batches
            for batch_id in list(self.active_batches.keys()):
                batch_metadata = self.active_batches[batch_id]
                if (batch_metadata.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED] and
                    batch_metadata.created_at and batch_metadata.created_at < cutoff_date):
                    del self.active_batches[batch_id]
                    if batch_id in self.batch_file_paths:
                        del self.batch_file_paths[batch_id]
                    cleaned_count += 1
            
            # Clean up metadata files
            metadata_files = list(settings.metadata_dir.glob("batch_*.json"))
            for metadata_file in metadata_files:
                try:
                    batch_metadata = await MetadataManager.load_batch_metadata(metadata_file.stem.replace("batch_", ""))
                    if (batch_metadata and batch_metadata.created_at and 
                        batch_metadata.created_at < cutoff_date):
                        metadata_file.unlink()
                        cleaned_count += 1
                except Exception:
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} old batches")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up batches: {e}")
            return 0