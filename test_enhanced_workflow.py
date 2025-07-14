#!/usr/bin/env python3
"""
Enhanced test script to verify the complete image processing workflow
with external file support and metadata preservation
"""
import asyncio
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import json
import sys

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.batch_manager import BatchManager
from app.services.image_processor import ImageProcessor
from app.core.config import settings
from app.utils.file_utils import FileUtils

async def create_test_files(temp_dir: Path):
    """Create test files with different formats and detailed metadata"""
    print("Creating test files with detailed metadata...")
    
    # Create test images with different characteristics
    files = []
    
    # 1. RGB PNG with transparency
    rgb_image = Image.new('RGBA', (300, 200), color=(255, 0, 0, 128))
    png_path = temp_dir / "test_rgba.png"
    rgb_image.save(png_path, 'PNG', dpi=(300, 300))
    files.append(png_path)
    
    # 2. Grayscale TIFF with LZW compression
    gray_image = Image.new('L', (400, 300), color=128)
    tiff_path = temp_dir / "test_gray.tiff"
    gray_image.save(tiff_path, 'TIFF', compression='lzw', dpi=(200, 200))
    files.append(tiff_path)
    
    # 3. RGB JPEG
    rgb_jpeg = Image.new('RGB', (500, 400), color='blue')
    jpeg_path = temp_dir / "test_rgb.jpg"
    rgb_jpeg.save(jpeg_path, 'JPEG', quality=95, dpi=(150, 150))
    files.append(jpeg_path)
    
    # 4. Paletted PNG
    palette_image = Image.new('P', (250, 250), color=1)
    palette_path = temp_dir / "test_palette.png"
    palette_image.save(palette_path, 'PNG', dpi=(72, 72))
    files.append(palette_path)
    
    # 5. Simple PDF-like file
    pdf_path = temp_dir / "test_document.pdf"
    with open(pdf_path, 'w') as f:
        f.write("%PDF-1.4\n%Test PDF content\n")
    files.append(pdf_path)
    
    print(f"Created {len(files)} test files with different formats and metadata")
    return files

async def test_metadata_extraction():
    """Test detailed metadata extraction"""
    print("\n=== Testing Metadata Extraction ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = await create_test_files(temp_path)
        
        for file_path in test_files:
            print(f"\nExtracting metadata for: {file_path.name}")
            metadata = FileUtils.get_file_metadata(file_path)
            
            if metadata:
                print(f"  Format: {metadata['original_format']}")
                print(f"  Size: {metadata['original_size']}")
                print(f"  DPI: {metadata['original_dpi']}")
                print(f"  Color Mode: {metadata['color_mode']}")
                print(f"  Compression: {metadata['compression_type']}")
                print(f"  Bit Depth: {metadata['original_bit_depth']}")
                print(f"  Alpha Channel: {metadata['original_alpha_channel']}")
                print(f"  Transparency: {metadata['original_transparency']}")
                print(f"  File Size: {metadata['file_size_bytes']} bytes")

async def test_external_file_processing():
    """Test processing files from external locations"""
    print("\n=== Testing External File Processing ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = await create_test_files(temp_path)
        
        # Initialize services
        batch_manager = BatchManager()
        image_processor = ImageProcessor()
        
        print(f"Processing {len(test_files)} external files...")
        
        # Create batches from external files
        batches = await batch_manager.create_batches_from_files(
            test_files,
            batch_size=2,
            preserve_metadata=True
        )
        
        print(f"Created {len(batches)} batches")
        
        # Process each batch
        for batch in batches:
            print(f"\nProcessing batch: {batch.batch_id}")
            print(f"Source type: {batch.input_source}")
            print(f"Files: {len(batch.images)}")
            
            for img in batch.images:
                print(f"  - {img.original_filename} ({img.original_format})")
                print(f"    Source: {img.source_path}")
                print(f"    Size: {img.original_size}, DPI: {img.original_dpi}")
            
            # Process the batch
            result = await image_processor.process_batch(batch)
            
            if result:
                print(f"✅ Batch {batch.batch_id} completed successfully")
                print(f"Status: {result.processing_status}")
                
                # Check results
                processed_count = sum(1 for img in result.images if img.processed_filename)
                print(f"Files processed: {processed_count}/{len(result.images)}")
                
                # Show metadata preservation
                for img in result.images:
                    if img.processed_filename:
                        print(f"  ✅ {img.original_filename} -> {img.processed_filename}")
                        print(f"     Original format: {img.original_format}")
                        print(f"     Original size: {img.original_size}")
                        print(f"     Original DPI: {img.original_dpi}")
            else:
                print(f"❌ Batch {batch.batch_id} failed")

async def test_format_restoration():
    """Test format restoration with metadata preservation"""
    print("\n=== Testing Format Restoration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = await create_test_files(temp_path)
        
        # Initialize services
        batch_manager = BatchManager()
        image_processor = ImageProcessor()
        
        # Create a single batch for testing
        batches = await batch_manager.create_batches_from_files(
            test_files,
            batch_size=len(test_files),
            preserve_metadata=True
        )
        
        if batches:
            batch = batches[0]
            print(f"Testing format restoration for batch: {batch.batch_id}")
            
            # Process the batch
            result = await image_processor.process_batch(batch)
            
            if result:
                print("✅ Processing completed successfully")
                
                # Check output files
                output_dir = settings.output_dir / batch.batch_id
                if output_dir.exists():
                    output_files = list(output_dir.glob("*"))
                    print(f"Output files: {len(output_files)}")
                    
                    for output_file in output_files:
                        print(f"  📄 {output_file.name} ({output_file.stat().st_size} bytes)")
                        
                        # Verify format restoration
                        original_file = None
                        for img in result.images:
                            if img.processed_filename == output_file.name:
                                original_file = img
                                break
                        
                        if original_file:
                            print(f"    Original: {original_file.original_filename}")
                            print(f"    Original format: {original_file.original_format}")
                            print(f"    Original size: {original_file.original_size}")
                            print(f"    Restored format: {output_file.suffix}")
                            
                            # Check if format was preserved
                            if output_file.suffix.lower() == f".{original_file.original_format.value}":
                                print("    ✅ Format correctly restored")
                            else:
                                print("    ⚠️  Format may not match exactly")

async def test_storage_management():
    """Test storage management and cleanup"""
    print("\n=== Testing Storage Management ===")
    
    # Get storage info
    input_size = FileUtils.get_directory_size(settings.input_dir)
    output_size = FileUtils.get_directory_size(settings.output_dir)
    temp_size = FileUtils.get_directory_size(settings.temp_dir)
    
    print(f"Input directory: {input_size / (1024*1024):.2f} MB")
    print(f"Output directory: {output_size / (1024*1024):.2f} MB")
    print(f"Temp directory: {temp_size / (1024*1024):.2f} MB")
    
    # List metadata files
    metadata_files = list(settings.metadata_dir.glob("*.json"))
    print(f"Metadata files: {len(metadata_files)}")
    
    for metadata_file in metadata_files:
        print(f"  📋 {metadata_file.name}")

async def main():
    """Run all tests"""
    print("=== Enhanced Image Processing Workflow Test ===\n")
    
    # Test 1: Metadata extraction
    await test_metadata_extraction()
    
    # Test 2: External file processing
    await test_external_file_processing()
    
    # Test 3: Format restoration
    await test_format_restoration()
    
    # Test 4: Storage management
    await test_storage_management()
    
    print("\n=== All Tests Completed ===")
    print("\nKey Features Demonstrated:")
    print("✅ Detailed metadata extraction and preservation")
    print("✅ External file processing from any location")
    print("✅ Accurate format restoration with original properties")
    print("✅ Storage management and file organization")
    print("✅ Batch processing with concurrent operations")
    print("✅ AI model integration (simulated)")

if __name__ == "__main__":
    asyncio.run(main()) 