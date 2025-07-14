import asyncio
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from pdf2image import convert_from_path
from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import ImageFormat, ImageMetadata, ConversionResult, ColorMode, CompressionType


logger = get_logger(__name__)

class ImageConverter:
    """Handles image format conversions with enhanced metadata preservation."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
    
    async def convert_to_jpeg_batch(self, image_files: List[Path], 
                                   batch_id: str, metadata_list: List[ImageMetadata]) -> List[ConversionResult]:
        """Convert a batch of images to JPEG format concurrently."""
        tasks = []
        
        for i, (file_path, metadata) in enumerate(zip(image_files, metadata_list)):
            task = self._convert_single_to_jpeg(file_path, batch_id, metadata, i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        conversion_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Conversion failed with exception: {result}")
                conversion_results.append(ConversionResult(
                    original_file="unknown",
                    converted_file="",
                    success=False,
                    error_message=str(result),
                    processing_time_ms=0,
                    size_change_bytes=0
                ))
            else:
                conversion_results.append(result)
        
        return conversion_results
    
    async def _convert_single_to_jpeg(self, file_path: Path, batch_id: str, 
                                     metadata: ImageMetadata, index: int) -> ConversionResult:
        """Convert a single image to JPEG format with metadata preservation."""
        start_time = time.time()
        
        try:
            output_dir = settings.temp_dir / "jpeg_converted" / batch_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert based on format
            if metadata.original_format == ImageFormat.PDF:
                return await self._convert_pdf_to_jpeg(file_path, output_dir, metadata, index)
            elif metadata.original_format == ImageFormat.TIFF:
                return await self._convert_tiff_to_jpeg(file_path, output_dir, metadata, index)
            else:
                return await self._convert_image_to_jpeg(file_path, output_dir, metadata, index)
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error converting {file_path} to JPEG: {e}")
            return ConversionResult(
                original_file=str(file_path),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                size_change_bytes=0
            )
    
    async def _convert_pdf_to_jpeg(self, pdf_path: Path, output_dir: Path, 
                                  metadata: ImageMetadata, index: int) -> ConversionResult:
        """Convert PDF to JPEG using pdf2image - handles all pages."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get total page count first
            def get_page_count():
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
                return page_count
            
            page_count = await loop.run_in_executor(self.executor, get_page_count)
            
            if page_count == 0:
                return ConversionResult(
                    original_file=str(pdf_path),
                    converted_file="",
                    success=False,
                    error_message="PDF has no pages",
                    processing_time_ms=0,
                    size_change_bytes=0
                )
            
            # Convert all pages to JPEG
            def convert_pdf():
                images = convert_from_path(
                    pdf_path,
                    dpi=metadata.original_dpi[0],
                    first_page=1,
                    last_page=page_count,  # Convert all pages
                    fmt='jpeg'
                )
                return images
            
            images = await loop.run_in_executor(self.executor, convert_pdf)
            
            if not images:
                return ConversionResult(
                    original_file=str(pdf_path),
                    converted_file="",
                    success=False,
                    error_message="PDF conversion failed",
                    processing_time_ms=0,
                    size_change_bytes=0
                )
            
            # Save each page as a separate JPEG
            jpeg_files = []
            original_size = pdf_path.stat().st_size
            total_converted_size = 0
            
            for page_num, image in enumerate(images, 1):
                jpeg_filename = f"{pdf_path.stem}_{index:03d}_p{page_num:02d}.jpg"
                jpeg_path = output_dir / jpeg_filename
                
                image.save(
                    jpeg_path,
                    'JPEG',
                    quality=settings.jpeg_quality,
                    optimize=True
                )
                
                jpeg_files.append(str(jpeg_path))
                total_converted_size += jpeg_path.stat().st_size
            
            processing_time = (time.time() - start_time) * 1000
            size_change = total_converted_size - original_size
            
            # Store the list of JPEG files in the converted_file field (comma-separated)
            return ConversionResult(
                original_file=str(pdf_path),
                converted_file=",".join(jpeg_files),  # Multiple files
                success=True,
                processing_time_ms=processing_time,
                size_change_bytes=size_change,
                page_count=page_count
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error converting PDF {pdf_path} to JPEG: {e}")
            return ConversionResult(
                original_file=str(pdf_path),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                size_change_bytes=0
            )
    
    async def _convert_tiff_to_jpeg(self, tiff_path: Path, output_dir: Path, 
                                   metadata: ImageMetadata, index: int) -> ConversionResult:
        """Convert TIFF to JPEG - handles all pages/frames."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            
            def convert_tiff():
                jpeg_files = []
                original_size = tiff_path.stat().st_size
                total_converted_size = 0
                
                with Image.open(tiff_path) as img:
                    # Check if it's a multi-page TIFF
                    try:
                        page_count = img.n_frames
                    except AttributeError:
                        page_count = 1
                    
                    for page_num in range(page_count):
                        # Seek to the specific page/frame
                        if page_count > 1:
                            img.seek(page_num)
                        
                        # Convert to RGB if necessary
                        current_img = img.copy()
                        if current_img.mode in ['RGBA', 'LA', 'P']:
                            # Create white background for transparent images
                            background = Image.new('RGB', current_img.size, (255, 255, 255))
                            if current_img.mode == 'P':
                                current_img = current_img.convert('RGBA')
                            background.paste(current_img, mask=current_img.split()[-1] if current_img.mode in ['RGBA', 'LA'] else None)
                            current_img = background
                        elif current_img.mode not in ['RGB', 'L']:
                            current_img = current_img.convert('RGB')
                        
                        # Save as JPEG
                        jpeg_filename = f"{tiff_path.stem}_{index:03d}_p{page_num+1:02d}.jpg"
                        jpeg_path = output_dir / jpeg_filename
                        
                        current_img.save(
                            jpeg_path,
                            'JPEG',
                            quality=settings.jpeg_quality,
                            optimize=True,
                            dpi=metadata.original_dpi
                        )
                        
                        jpeg_files.append(str(jpeg_path))
                        total_converted_size += jpeg_path.stat().st_size
                
                return jpeg_files, total_converted_size, original_size
            
            jpeg_files, total_converted_size, original_size = await loop.run_in_executor(self.executor, convert_tiff)
            
            processing_time = (time.time() - start_time) * 1000
            size_change = total_converted_size - original_size
            
            return ConversionResult(
                original_file=str(tiff_path),
                converted_file=",".join(jpeg_files),  # Multiple files
                success=True,
                processing_time_ms=processing_time,
                size_change_bytes=size_change,
                page_count=len(jpeg_files)
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error converting TIFF {tiff_path} to JPEG: {e}")
            return ConversionResult(
                original_file=str(tiff_path),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                size_change_bytes=0
            )
    
    async def _convert_image_to_jpeg(self, image_path: Path, output_dir: Path, 
                                    metadata: ImageMetadata, index: int) -> ConversionResult:
        """Convert single-page image formats to JPEG with metadata preservation."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            
            def convert_image():
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ['RGBA', 'LA', 'P']:
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ['RGBA', 'LA'] else None)
                        img = background
                    elif img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                    
                    # Save as JPEG
                    jpeg_filename = f"{image_path.stem}_{index:03d}.jpg"
                    jpeg_path = output_dir / jpeg_filename
                    
                    img.save(
                        jpeg_path,
                        'JPEG',
                        quality=settings.jpeg_quality,
                        optimize=True,
                        dpi=metadata.original_dpi
                    )
                    
                    return str(jpeg_path)
            
            jpeg_path_str = await loop.run_in_executor(self.executor, convert_image)
            
            processing_time = (time.time() - start_time) * 1000
            original_size = image_path.stat().st_size
            converted_size = Path(jpeg_path_str).stat().st_size
            size_change = converted_size - original_size
            
            return ConversionResult(
                original_file=str(image_path),
                converted_file=jpeg_path_str,
                success=True,
                processing_time_ms=processing_time,
                size_change_bytes=size_change,
                page_count=1
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error converting image {image_path} to JPEG: {e}")
            return ConversionResult(
                original_file=str(image_path),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                size_change_bytes=0
            )
    
    async def convert_from_jpeg_batch(self, jpeg_files: List[Path], 
                                     metadata_list: List[ImageMetadata],
                                     output_dir: Path) -> List[ConversionResult]:
        """Convert JPEG files back to their original formats using enhanced metadata."""
        tasks = []
        
        for jpeg_file, metadata in zip(jpeg_files, metadata_list):
            task = self._convert_single_from_jpeg(jpeg_file, metadata, output_dir)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        conversion_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Conversion from JPEG failed with exception: {result}")
                conversion_results.append(ConversionResult(
                    original_file="unknown",
                    converted_file="",
                    success=False,
                    error_message=str(result),
                    processing_time_ms=0,
                    size_change_bytes=0
                ))
            else:
                conversion_results.append(result)
        
        return conversion_results
    
    async def _convert_single_from_jpeg(self, jpeg_path: Path, 
                                       metadata: ImageMetadata, 
                                       output_dir: Path) -> ConversionResult:
        """Convert a single JPEG file back to original format using enhanced metadata."""
        start_time = time.time()
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            original_name = metadata.original_filename
            original_stem = Path(original_name).stem
            output_filename = f"{original_stem}_processed{Path(original_name).suffix}"
            output_path = output_dir / output_filename
            
            # Convert based on original format
            success = False
            if metadata.original_format == ImageFormat.TIFF:
                success = await self._convert_jpeg_to_tiff(jpeg_path, output_path, metadata)
            elif metadata.original_format == ImageFormat.PDF:
                success = await self._convert_jpeg_to_pdf(jpeg_path, output_path, metadata)
            elif metadata.original_format == ImageFormat.PNG:
                success = await self._convert_jpeg_to_png(jpeg_path, output_path, metadata)
            elif metadata.original_format == ImageFormat.BMP:
                success = await self._convert_jpeg_to_bmp(jpeg_path, output_path, metadata)
            else:
                # Default to JPEG
                success = await self._convert_jpeg_to_jpeg(jpeg_path, output_path, metadata)
            
            processing_time = (time.time() - start_time) * 1000
            
            if success and output_path.exists():
                jpeg_size = jpeg_path.stat().st_size
                output_size = output_path.stat().st_size
                size_change = output_size - jpeg_size
                
                return ConversionResult(
                    original_file=str(jpeg_path),
                    converted_file=str(output_path),
                    success=True,
                    processing_time_ms=processing_time,
                    size_change_bytes=size_change
                )
            else:
                return ConversionResult(
                    original_file=str(jpeg_path),
                    converted_file="",
                    success=False,
                    error_message="Format restoration failed",
                    processing_time_ms=processing_time,
                    size_change_bytes=0
                )
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error converting {jpeg_path} from JPEG: {e}")
            return ConversionResult(
                original_file=str(jpeg_path),
                converted_file="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                size_change_bytes=0
            )
    
    async def _convert_jpeg_to_tiff(self, jpeg_path: Path, tiff_path: Path, 
                                   metadata: ImageMetadata) -> bool:
        """Convert JPEG to TIFF with enhanced metadata preservation."""
        try:
            loop = asyncio.get_event_loop()
            
            def convert_to_tiff():
                with Image.open(jpeg_path) as img:
                    # Restore original color mode if possible
                    if metadata.original_alpha_channel and metadata.color_mode == ColorMode.RGBA:
                        img = img.convert('RGBA')
                    elif metadata.color_mode == ColorMode.L:
                        img = img.convert('L')
                    elif metadata.color_mode == ColorMode.LA:
                        img = img.convert('LA')
                    elif metadata.color_mode == ColorMode.P and metadata.original_palette:
                        img = img.convert('P', palette=Image.Palette.ADAPTIVE)
                    else:
                        img = img.convert('RGB')
                    
                    # Determine compression
                    compression = 'tiff_lzw'
                    if metadata.compression_type == CompressionType.TIFF_JPEG:
                        compression = 'tiff_jpeg'
                    elif metadata.compression_type == CompressionType.TIFF_PACKBITS:
                        compression = 'tiff_packbits'
                    
                    # Save with preserved metadata
                    img.save(
                        tiff_path,
                        'TIFF',
                        compression=compression,
                        dpi=metadata.original_dpi,
                        description=f"Processed from {metadata.original_filename}"
                    )
                return True
            
            return await loop.run_in_executor(self.executor, convert_to_tiff)
            
        except Exception as e:
            logger.error(f"Error converting JPEG to TIFF: {e}")
            return False
    
    async def _convert_jpeg_to_pdf(self, jpeg_path: Path, pdf_path: Path, 
                                  metadata: ImageMetadata) -> bool:
        """Convert JPEG to PDF with metadata preservation."""
        try:
            loop = asyncio.get_event_loop()
            
            def convert_to_pdf():
                with Image.open(jpeg_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as PDF
                    img.save(
                        pdf_path,
                        'PDF',
                        resolution=metadata.original_dpi[0]
                    )
                return True
            
            return await loop.run_in_executor(self.executor, convert_to_pdf)
            
        except Exception as e:
            logger.error(f"Error converting JPEG to PDF: {e}")
            return False
    
    async def _convert_jpeg_to_png(self, jpeg_path: Path, png_path: Path, 
                                  metadata: ImageMetadata) -> bool:
        """Convert JPEG to PNG with metadata preservation."""
        try:
            loop = asyncio.get_event_loop()
            
            def convert_to_png():
                with Image.open(jpeg_path) as img:
                    # Restore transparency if original had it
                    if metadata.original_transparency:
                        # Create transparent background
                        img = img.convert('RGBA')
                        # Make white pixels transparent
                        data = img.getdata()
                        new_data = []
                        for item in data:
                            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                                new_data.append((255, 255, 255, 0))
                            else:
                                new_data.append(item)
                        img.putdata(new_data)
                    
                    # Save with preserved metadata
                    img.save(
                        png_path,
                        'PNG',
                        optimize=True,
                        dpi=metadata.original_dpi
                    )
                return True
            
            return await loop.run_in_executor(self.executor, convert_to_png)
            
        except Exception as e:
            logger.error(f"Error converting JPEG to PNG: {e}")
            return False
    
    async def _convert_jpeg_to_bmp(self, jpeg_path: Path, bmp_path: Path, 
                                  metadata: ImageMetadata) -> bool:
        """Convert JPEG to BMP with metadata preservation."""
        try:
            loop = asyncio.get_event_loop()
            
            def convert_to_bmp():
                with Image.open(jpeg_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as BMP
                    img.save(
                        bmp_path,
                        'BMP',
                        dpi=metadata.original_dpi
                    )
                return True
            
            return await loop.run_in_executor(self.executor, convert_to_bmp)
            
        except Exception as e:
            logger.error(f"Error converting JPEG to BMP: {e}")
            return False
    
    async def _convert_jpeg_to_jpeg(self, jpeg_path: Path, output_path: Path, 
                                   metadata: ImageMetadata) -> bool:
        """Copy JPEG with metadata preservation."""
        try:
            loop = asyncio.get_event_loop()
            
            def copy_jpeg():
                with Image.open(jpeg_path) as img:
                    img.save(
                        output_path,
                        'JPEG',
                        quality=settings.jpeg_quality,
                        optimize=True,
                        dpi=metadata.original_dpi
                    )
                return True
            
            return await loop.run_in_executor(self.executor, copy_jpeg)
            
        except Exception as e:
            logger.error(f"Error copying JPEG: {e}")
            return False
    
    def __del__(self):
        """Cleanup executor on object destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)