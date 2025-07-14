# app/services/image_processor.py

import asyncio
from pathlib import Path
from typing import List, Optional
from PIL import Image

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    BatchMetadata, ImageMetadata, ConversionResult, ProcessingStatus,
    ImageFormat, ColorMode
)

from ..utils.metadata_utils import MetadataManager
from .converter import ImageConverter
from .ai_simulator import AIModelSimulator

logger = get_logger(__name__)


class ImageProcessor:
    """Service to orchestrate image batch processing with focus on core requirements."""

    def __init__(self):
        self.converter = ImageConverter()
        self.ai_simulator = AIModelSimulator()

    async def process_batch(self, batch_metadata: BatchMetadata, original_file_paths: List[Path] = None) -> Optional[BatchMetadata]:
        """Main entry point to process a batch of images with the complete workflow."""
        try:
            batch_id = batch_metadata.batch_id
            logger.info(f"Starting processing for batch {batch_id} with {len(batch_metadata.images)} files")
            
            # Step 1: Update status to PROCESSING
            await MetadataManager.update_batch_status(batch_id, ProcessingStatus.PROCESSING)
            
            # Step 2: Convert to JPEG concurrently
            logger.info(f"Converting {len(batch_metadata.images)} files to JPEG format")
            
            # Fix: Use original file paths if provided, otherwise try to find files
            image_files = []
            if original_file_paths:
                # Use the original file paths from batch creation
                image_files = original_file_paths
            else:
                # Fallback: try to find files in input directory
                for img in batch_metadata.images:
                    file_path = settings.input_dir / img.original_filename
                    if file_path.exists():
                        image_files.append(file_path)
                    else:
                        logger.error(f"File not found: {file_path}")
            
            if not image_files:
                logger.error(f"No valid files found for batch {batch_id}")
                await MetadataManager.update_batch_status(batch_id, ProcessingStatus.FAILED)
                return None
            
            # Separate JPEG/JPG files from others
            jpeg_files = []
            non_jpeg_files = []
            for file_path in image_files:
                ext = file_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    jpeg_files.append(file_path)
                    logger.info(f"Skipping conversion for JPEG file: {file_path.name}")
                else:
                    non_jpeg_files.append(file_path)
            
            # Convert only non-JPEG files to JPEG
            conversion_results = await self.converter.convert_to_jpeg_batch(
                non_jpeg_files,
                batch_id,
                [img for img in batch_metadata.images if Path(img.original_filename).suffix.lower() not in ['.jpg', '.jpeg']]
            )
            
            # Step 3: Update metadata with JPEG filenames and collect all JPEG files
            all_jpeg_files = []
            # Add already-JPEG files directly
            all_jpeg_files.extend(jpeg_files)
            for file_path in jpeg_files:
                await MetadataManager.update_image_metadata(batch_id, file_path.name, jpeg_filename=file_path.name)
            for result in conversion_results:
                if result.success:
                    original_name = Path(result.original_file).name
                    if "," in result.converted_file:
                        jpeg_file_list = result.converted_file.split(",")
                        jpeg_filenames = [Path(jf).name for jf in jpeg_file_list]
                        jpeg_filename_str = ",".join(jpeg_filenames)
                        all_jpeg_files.extend([Path(jf.strip()) for jf in jpeg_file_list])
                        logger.info(f"Multi-page file {original_name} converted to {len(jpeg_file_list)} JPEG files")
                    else:
                        jpeg_filename = Path(result.converted_file).name
                        jpeg_filename_str = jpeg_filename
                        all_jpeg_files.append(Path(result.converted_file))
                    await MetadataManager.update_image_metadata(batch_id, original_name, jpeg_filename=jpeg_filename_str)
                else:
                    logger.error(f"JPEG conversion failed for {result.original_file}: {result.error_message}")
            
            # Step 4: AI Processing (Masking) - Process all JPEG files
            logger.info(f"Starting AI processing for batch {batch_id} with {len(all_jpeg_files)} JPEG files")
            
            if all_jpeg_files:
                ai_results = await self.ai_simulator.process_images_batch(all_jpeg_files)
                logger.info(f"AI processing completed for {len(ai_results)} files")
                
                # Log AI processing results
                successful_ai = sum(1 for r in ai_results if r.processing_successful)
                logger.info(f"AI processing: {successful_ai}/{len(ai_results)} successful")
            else:
                logger.warning(f"No JPEG files found for AI processing in batch {batch_id}")
            
            # Step 5: Convert back to original format
            logger.info(f"Converting processed files back to original format")
            output_dir = settings.output_dir / batch_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Group JPEG files by original file for restoration
            restoration_results = await self._restore_original_formats(
                all_jpeg_files, 
                batch_metadata.images, 
                output_dir
            )
            
            # Step 6: Update metadata with processed filenames
            for result in restoration_results:
                if result.success:
                    original_name = Path(result.original_file).name
                    processed_name = Path(result.converted_file).name
                    await MetadataManager.update_image_metadata(batch_id, original_name, processed_filename=processed_name)
                else:
                    logger.error(f"Format restoration failed for {result.original_file}: {result.error_message}")
            
            # Step 7: Update final status
            failed_count = len([r for r in restoration_results if not r.success])
            final_status = ProcessingStatus.COMPLETED if failed_count == 0 else ProcessingStatus.FAILED
            await MetadataManager.update_batch_status(batch_id, final_status)
            
            logger.info(f"Batch {batch_id} processing completed with status: {final_status}")
            return await MetadataManager.load_batch_metadata(batch_id)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_metadata.batch_id}: {e}")
            await MetadataManager.update_batch_status(batch_metadata.batch_id, ProcessingStatus.FAILED, error_message=str(e))
            return None

    async def _restore_original_formats(self, jpeg_files: List[Path], 
                                      metadata_list: List[ImageMetadata], 
                                      output_dir: Path) -> List[ConversionResult]:
        """Restore original formats from JPEG files, handling multi-page files."""
        restoration_results = []
        
        # Group JPEG files by original file
        jpeg_groups = {}
        for jpeg_file in jpeg_files:
            # Extract original file info from JPEG filename
            # Format: originalname_index_p01.jpg, originalname_index_p02.jpg, etc.
            filename_parts = jpeg_file.stem.split('_')
            
            # Handle different naming patterns
            if len(filename_parts) >= 3 and filename_parts[-1].startswith('p'):
                # Multi-page file: originalname_index_p01.jpg
                try:
                    page_num = int(filename_parts[-1][1:])  # Extract number after 'p'
                    original_stem = '_'.join(filename_parts[:-2])  # Remove index and page number
                    if original_stem not in jpeg_groups:
                        jpeg_groups[original_stem] = []
                    jpeg_groups[original_stem].append((jpeg_file, page_num))
                except ValueError:
                    # Fallback: treat as single page
                    original_stem = '_'.join(filename_parts[:-1])
                    if original_stem not in jpeg_groups:
                        jpeg_groups[original_stem] = []
                    jpeg_groups[original_stem].append((jpeg_file, 1))
            else:
                # Single-page file: originalname_index.jpg
                original_stem = '_'.join(filename_parts[:-1])  # Remove index
                if original_stem not in jpeg_groups:
                    jpeg_groups[original_stem] = []
                jpeg_groups[original_stem].append((jpeg_file, 1))
        
        # Restore each original file
        for original_stem, group_jpeg_files in jpeg_groups.items():
            # Find corresponding metadata
            metadata = None
            for img_metadata in metadata_list:
                if img_metadata.original_filename.startswith(original_stem):
                    metadata = img_metadata
                    break
            
            if metadata is None:
                logger.error(f"No metadata found for {original_stem}")
                continue
            
            # Sort JPEG files by page number for multi-page files
            group_jpeg_files.sort(key=lambda x: x[1])  # Sort by page number
            sorted_jpeg_files = [f[0] for f in group_jpeg_files]  # Extract just the file paths
            
            # Restore to original format
            if metadata.original_format in [ImageFormat.PDF, ImageFormat.TIFF] and len(sorted_jpeg_files) > 1:
                # Multi-page restoration
                result = await self._restore_multi_page_format(sorted_jpeg_files, metadata, output_dir)
            else:
                # Single-page restoration
                result = await self.converter._convert_single_from_jpeg(sorted_jpeg_files[0], metadata, output_dir)
            
            restoration_results.append(result)
        
        return restoration_results

    async def _restore_multi_page_format(self, jpeg_files: List[Path], 
                                        metadata: ImageMetadata, 
                                        output_dir: Path) -> ConversionResult:
        """Restore multi-page PDF or TIFF from multiple JPEG files."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            original_name = metadata.original_filename
            original_stem = Path(original_name).stem
            output_filename = f"{original_stem}_processed{Path(original_name).suffix}"
            output_path = output_dir / output_filename
            
            if metadata.original_format == ImageFormat.PDF:
                success = await self._restore_multi_page_pdf(jpeg_files, output_path, metadata)
            elif metadata.original_format == ImageFormat.TIFF:
                success = await self._restore_multi_page_tiff(jpeg_files, output_path, metadata)
            else:
                # Fallback to single page
                result = await self.converter._convert_single_from_jpeg(jpeg_files[0], metadata, output_dir)
                return result
            
            if success and output_path.exists():
                return ConversionResult(
                    original_file=",".join(str(f) for f in jpeg_files),
                    converted_file=str(output_path),
                    success=True,
                    processing_time_ms=0,
                    size_change_bytes=0,
                    page_count=len(jpeg_files)
                )
            else:
                return ConversionResult(
                    original_file=",".join(str(f) for f in jpeg_files),
                    converted_file="",
                    success=False,
                    error_message="Multi-page format restoration failed",
                    processing_time_ms=0,
                    size_change_bytes=0
                )
                
        except Exception as e:
            logger.error(f"Error restoring multi-page format: {e}")
            return ConversionResult(
                original_file=",".join(str(f) for f in jpeg_files),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=0,
                size_change_bytes=0
            )

    async def _restore_multi_page_pdf(self, jpeg_files: List[Path], 
                                     output_path: Path, 
                                     metadata: ImageMetadata) -> bool:
        """Restore multi-page PDF from JPEG files."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            # Create PDF with all pages
            c = canvas.Canvas(str(output_path), pagesize=letter)
            
            for jpeg_file in jpeg_files:
                # Add JPEG as page to PDF
                c.drawImage(str(jpeg_file), 0, 0, width=letter[0], height=letter[1])
                c.showPage()
            
            c.save()
            return True
            
        except Exception as e:
            logger.error(f"Error creating multi-page PDF: {e}")
            return False

    async def _restore_multi_page_tiff(self, jpeg_files: List[Path], 
                                      output_path: Path, 
                                      metadata: ImageMetadata) -> bool:
        """Restore multi-page TIFF from JPEG files."""
        try:
            images = []
            for jpeg_file in jpeg_files:
                with Image.open(jpeg_file) as img:
                    # Convert to appropriate mode based on metadata
                    if metadata.color_mode == ColorMode.L:
                        img = img.convert('L')
                    elif metadata.color_mode == ColorMode.RGBA:
                        img = img.convert('RGBA')
                    else:
                        img = img.convert('RGB')
                    images.append(img.copy())
            
            if images:
                # Save as multi-page TIFF
                images[0].save(
                    output_path,
                    'TIFF',
                    save_all=True,
                    append_images=images[1:],
                    compression='tiff_lzw',
                    dpi=metadata.original_dpi
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating multi-page TIFF: {e}")
            return False
