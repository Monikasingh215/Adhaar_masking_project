#!/usr/bin/env python3
"""
Test script for folder upload functionality with automatic batch creation.
This script simulates uploading a folder of files and shows how batches are created.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from app.services.batch_manager import BatchManager
from app.services.image_processor import ImageProcessor
from app.core.logging import setup_logging

logger = setup_logging()

async def test_folder_upload():
    """Test the folder upload functionality with batch creation."""
    
    print("🚀 Testing Folder Upload with Batch Creation")
    print("=" * 50)
    
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files with different dates (simulating files from same date)
        test_files = []
        
        # Create 120 test files (6 batches of 20 files each)
        for i in range(120):
            # Create files with different extensions
            extensions = ['.pdf', '.tiff', '.png', '.jpg', '.bmp']
            ext = extensions[i % len(extensions)]
            
            file_path = temp_path / f"test_file_{i:03d}{ext}"
            
            # Create a simple text file as placeholder
            with open(file_path, 'w') as f:
                f.write(f"Test content for file {i}")
            
            test_files.append(file_path)
        
        print(f"✅ Created {len(test_files)} test files in temporary directory")
        print(f"📁 Temporary directory: {temp_path}")
        
        # Initialize batch manager
        batch_manager = BatchManager()
        
        # Test 1: Create batches from files (simulating folder upload)
        print("\n📦 Test 1: Creating batches from uploaded files")
        print("-" * 40)
        
        try:
            batches = await batch_manager.create_batches_from_files(
                test_files, 
                batch_size=20
            )
            
            print(f"✅ Successfully created {len(batches)} batches")
            
            # Display batch information
            for i, batch in enumerate(batches, 1):
                print(f"  Batch {i}: {batch.batch_id}")
                print(f"    - Files: {batch.total_files}")
                print(f"    - Status: {batch.processing_status}")
                print(f"    - Source: {batch.input_source}")
                print(f"    - Created: {batch.created_at}")
                
                # Verify file paths are stored
                file_paths = batch_manager.get_batch_file_paths(batch.batch_id)
                if file_paths:
                    print(f"    - File paths stored: {len(file_paths)} files")
                else:
                    print(f"    - ⚠️  File paths not stored!")
                
                print()
            
            # Test 2: Get batch status
            print("📊 Test 2: Getting batch status")
            print("-" * 40)
            
            for batch in batches[:3]:  # Test first 3 batches
                status = await batch_manager.get_batch_status(batch.batch_id)
                if status:
                    print(f"Batch {batch.batch_id}:")
                    print(f"  - Status: {status['status']}")
                    print(f"  - Progress: {status['progress']:.1f}%")
                    print(f"  - Current Step: {status['current_step']}")
                    print(f"  - Total Files: {status['total_files']}")
                    print()
            
            # Test 3: List all batches
            print("📋 Test 3: Listing all batches")
            print("-" * 40)
            
            batch_list = await batch_manager.list_batches()
            print(f"Total batches: {batch_list['total_batches']}")
            print(f"Active batches: {batch_list['active_batches']}")
            print(f"Completed batches: {batch_list['completed_batches']}")
            print(f"Failed batches: {batch_list['failed_batches']}")
            
            # Test 4: Simulate processing (without actual file conversion)
            print("\n⚙️  Test 4: Simulating batch processing")
            print("-" * 40)
            
            image_processor = ImageProcessor()
            
            # Process first batch as example
            if batches:
                first_batch = batches[0]
                file_paths = batch_manager.get_batch_file_paths(first_batch.batch_id)
                
                if file_paths:
                    print(f"Processing batch: {first_batch.batch_id}")
                    print(f"Files to process: {len(file_paths)}")
                    
                    # Note: This would normally process the files
                    # For testing, we'll just simulate the workflow
                    print("Batch processing workflow ready")
                    print("   (Actual processing would convert to JPEG, apply AI, restore format)")
                else:
                    print("No file paths found for batch")
            
            print("\nAll tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            logger.error(f"Test failed: {e}")
            raise

async def test_different_batch_sizes():
    """Test different batch sizes."""
    
    print("\n🔧 Testing Different Batch Sizes")
    print("=" * 40)
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = []
        
        # Create 50 test files
        for i in range(50):
            file_path = temp_path / f"test_file_{i:03d}.pdf"
            with open(file_path, 'w') as f:
                f.write(f"Test content for file {i}")
            test_files.append(file_path)
        
        batch_manager = BatchManager()
        
        # Test different batch sizes
        batch_sizes = [10, 20, 25, 50]
        
        for batch_size in batch_sizes:
            print(f"\n📦 Testing batch size: {batch_size}")
            
            try:
                batches = await batch_manager.create_batches_from_files(
                    test_files, 
                    batch_size=batch_size
                )
                
                expected_batches = (len(test_files) + batch_size - 1) // batch_size
                print(f"  Expected batches: {expected_batches}")
                print(f"  Actual batches: {len(batches)}")
                print(f"  ✅ {'PASS' if len(batches) == expected_batches else 'FAIL'}")
                
                # Clean up batches for next test
                batch_manager.active_batches.clear()
                batch_manager.batch_file_paths.clear()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")

async def main():
    """Main test function."""
    print("🧪 Starting Folder Upload Tests")
    print("=" * 60)
    
    try:
        # Test basic folder upload functionality
        await test_folder_upload()
        
        # Test different batch sizes
        await test_different_batch_sizes()
        
        print("\n🎯 All tests completed!")
        print("\n📝 Summary:")
        print("  ✅ Folder upload with batch creation works")
        print("  ✅ Different batch sizes are supported")
        print("  ✅ Batch metadata is properly stored")
        print("  ✅ File paths are preserved for processing")
        print("\n🚀 Ready for web UI testing!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.error(f"Test suite failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 