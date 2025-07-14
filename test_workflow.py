#!/usr/bin/env python3
"""
Test script to verify the complete image processing workflow
"""
import asyncio
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import json

# Add the app directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.batch_manager import BatchManager
from app.services.image_processor import ImageProcessor
from app.core.config import settings

async def create_test_files(temp_dir: Path):
    """Create test files with different formats and dates"""
    print("Creating test files...")
    
    # Create test images
    test_image = Image.new('RGB', (200, 200), color='blue')
    
    # Create files with different formats
    files = [
        temp_dir / "test1.pdf",  # We'll create a simple PDF-like file
        temp_dir / "test2.png",
        temp_dir / "test3.tiff",
        temp_dir / "test4.jpg"
    ]
    
    # Save images
    test_image.save(files[1], 'PNG')
    test_image.save(files[2], 'TIFF', compression='lzw')
    test_image.save(files[3], 'JPEG', quality=95)
    
    # Create a simple text file that looks like a PDF
    with open(files[0], 'w') as f:
        f.write("%PDF-1.4\n%Test PDF content\n")
    
    print(f"Created {len(files)} test files")
    return files

async def test_complete_workflow():
    """Test the complete workflow from batch creation to processing"""
    print("=== Testing Complete Image Processing Workflow ===\n")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        test_files = await create_test_files(temp_path)
        
        # Initialize services
        batch_manager = BatchManager()
        image_processor = ImageProcessor()
        
        print("1. Creating batches from directory...")
        batches = await batch_manager.create_batches_from_directory(temp_path, batch_size=2)
        print(f"   Created {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            print(f"   Batch {i+1}: {len(batch.images)} files")
            for img in batch.images:
                print(f"     - {img.original_filename} ({img.original_format})")
        
        print("\n2. Processing batches...")
        for batch in batches:
            print(f"   Processing batch: {batch.batch_id}")
            result = await image_processor.process_batch(batch)
            
            if result:
                print(f"   ✅ Batch {batch.batch_id} completed successfully")
                print(f"   Status: {result.processing_status}")
                
                # Check results
                processed_count = sum(1 for img in result.images if img.processed_filename)
                print(f"   Files processed: {processed_count}/{len(result.images)}")
            else:
                print(f"   ❌ Batch {batch.batch_id} failed")
        
        print("\n3. Checking output directory...")
        output_files = list(settings.output_dir.rglob("*"))
        print(f"   Output files: {len(output_files)}")
        for file in output_files:
            if file.is_file():
                print(f"     - {file.name} ({file.stat().st_size} bytes)")
        
        print("\n=== Workflow Test Completed ===")

if __name__ == "__main__":
    asyncio.run(test_complete_workflow()) 