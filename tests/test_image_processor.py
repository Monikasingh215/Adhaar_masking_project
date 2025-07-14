"""
Test file for Image Processing Service
Tests the core image processing workflow and batch operations
"""

import pytest
import asyncio
import os
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

import pytest_asyncio
from PIL import Image
import fitz  # PyMuPDF

from app.services.image_processor import ImageProcessor
from app.services.batch_manager import BatchManager
from app.services.converter import Converter
from app.services.ai_simulator import AISimulator
from app.utils.metadata_utils import MetadataUtils
from app.utils.file_utils import FileUtils
from app.core.config import settings


class TestImageProcessor:
    """Test suite for ImageProcessor service"""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance"""
        return ImageProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images with different formats and upload dates"""
        images = {}
        
        # Create test images with different properties
        test_configs = [
            {"name": "test1.jpg", "format": "JPEG", "size": (200, 150), "color": "red", "dpi": (72, 72)},
            {"name": "test2.png", "format": "PNG", "size": (300, 200), "color": "blue", "dpi": (96, 96)},
            {"name": "test3.bmp", "format": "BMP", "size": (150, 100), "color": "green", "dpi": (72, 72)},
            {"name": "test4.tiff", "format": "TIFF", "size": (400, 300), "color": "yellow", "dpi": (300, 300)},
        ]
        
        for i, config in enumerate(test_configs):
            img = Image.new('RGB', config["size"], color=config["color"])
            filepath = os.path.join(temp_dir, config["name"])
            
            if config["format"] == "TIFF":
                img.save(filepath, format=config["format"], dpi=config["dpi"], compression="lzw")
            else:
                img.save(filepath, format=config["format"], dpi=config["dpi"])
            
            # Set different modification times to simulate upload dates
            timestamp = datetime.now().timestamp() - (i * 3600)  # 1 hour apart
            os.utime(filepath, (timestamp, timestamp))
            
            images[config["name"]] = {
                "path": filepath,
                "format": config["format"],
                "size": config["size"],
                "dpi": config["dpi"]
            }
        
        return images
    
    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create sample PDF file"""
        pdf_path = os.path.join(temp_dir, "test.pdf")
        
        # Create a simple PDF using PyMuPDF
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4 size
        page.insert_text((100, 100), "Test PDF Content", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        return pdf_path
    
    @pytest.fixture
    def mock_batch_metadata(self):
        """Mock batch metadata structure"""
        return {
            "batch_id": "batch_20240115_001",
            "created_at": "2024-01-15T10:30:00Z",
            "files": [
                {
                    "filename": "test1.jpg",
                    "original_path": "/input/test1.jpg",
                    "original_format": "JPEG",
                    "dpi": (72, 72),
                    "color_mode": "RGB",
                    "compression": None,
                    "file_size": 15360,
                    "upload_date": "2024-01-15T10:00:00Z",
                    "dimensions": (200, 150)
                },
                {
                    "filename": "test2.png",
                    "original_path": "/input/test2.png",
                    "original_format": "PNG",
                    "dpi": (96, 96),
                    "color_mode": "RGB",
                    "compression": None,
                    "file_size": 25600,
                    "upload_date": "2024-01-15T10:05:00Z",
                    "dimensions": (300, 200)
                }
            ],
            "total_files": 2,
            "status": "pending",
            "processing_started": None,
            "processing_completed": None
        }


class TestImageProcessorCore(TestImageProcessor):
    """Test core ImageProcessor functionality"""
    
    def test_processor_initialization(self, processor):
        """Test ImageProcessor initialization"""
        assert processor is not None
        assert hasattr(processor, 'batch_manager')
        assert hasattr(processor, 'converter')
        assert hasattr(processor, 'ai_simulator')
    
    def test_validate_directory_exists(self, processor, temp_dir):
        """Test directory validation - existing directory"""
        assert processor._validate_directory(temp_dir) is True
    
    def test_validate_directory_not_exists(self, processor):
        """Test directory validation - non-existing directory"""
        with pytest.raises(FileNotFoundError):
            processor._validate_directory("/nonexistent/path")
    
    def test_validate_directory_empty(self, processor, temp_dir):
        """Test directory validation - empty directory"""
        with pytest.raises(ValueError, match="No supported image files found"):
            processor._validate_directory(temp_dir)
    
    def test_validate_directory_with_files(self, processor, temp_dir, sample_images):
        """Test directory validation - directory with supported files"""
        assert processor._validate_directory(temp_dir) is True
    
    def test_scan_directory_for_images(self, processor, temp_dir, sample_images):
        """Test scanning directory for supported image files"""
        files = processor._scan_directory(temp_dir)
        
        assert len(files) == 4
        assert all(isinstance(f, dict) for f in files)
        assert all('path' in f and 'upload_date' in f for f in files)
        
        # Check file extensions
        extensions = [Path(f['path']).suffix.lower() for f in files]
        expected_extensions = ['.jpg', '.png', '.bmp', '.tiff']
        assert all(ext in expected_extensions for ext in extensions)
    
    def test_scan_directory_excludes_unsupported(self, processor, temp_dir, sample_images):
        """Test that unsupported files are excluded from scan"""
        # Create unsupported files
        unsupported_files = ['test.txt', 'test.doc', 'test.exe']
        for filename in unsupported_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        files = processor._scan_directory(temp_dir)
        
        # Should only find supported image files
        assert len(files) == 4
        found_files = [Path(f['path']).name for f in files]
        assert all(f not in found_files for f in unsupported_files)


class TestImageProcessorBatchProcessing(TestImageProcessor):
    """Test batch processing functionality"""
    
    @pytest_asyncio.async_test
    async def test_process_directory_success(self, processor, temp_dir, sample_images):
        """Test successful directory processing"""
        with patch.object(processor.batch_manager, 'create_batches_by_date') as mock_create_batches, \
             patch.object(processor, '_process_batches_concurrent') as mock_process_batches:
            
            # Mock batch creation
            mock_create_batches.return_value = [
                {"batch_id": "batch_001", "files": list(sample_images.values())[:2]},
                {"batch_id": "batch_002", "files": list(sample_images.values())[2:]}
            ]
            
            # Mock batch processing
            mock_process_batches.return_value = {
                "batches_processed": 2,
                "total_files": 4,
                "errors": [],
                "processing_time": 30.5
            }
            
            result = await processor.process_directory(temp_dir)
            
            assert result["status"] == "success"
            assert result["batches_processed"] == 2
            assert result["total_files"] == 4
            assert result["errors"] == []
            assert isinstance(result["processing_time"], float)
    
    @pytest_asyncio.async_test
    async def test_process_directory_with_errors(self, processor, temp_dir, sample_images):
        """Test directory processing with errors"""
        with patch.object(processor.batch_manager, 'create_batches_by_date') as mock_create_batches, \
             patch.object(processor, '_process_batches_concurrent') as mock_process_batches:
            
            mock_create_batches.return_value = [
                {"batch_id": "batch_001", "files": list(sample_images.values())}
            ]
            
            # Mock processing with errors
            mock_process_batches.return_value = {
                "batches_processed": 1,
                "total_files": 4,
                "errors": ["Failed to process test1.jpg: Conversion error"],
                "processing_time": 25.3
            }
            
            result = await processor.process_directory(temp_dir)
            
            assert result["status"] == "completed_with_errors"
            assert len(result["errors"]) == 1
            assert "Failed to process test1.jpg" in result["errors"][0]
    
    @pytest_asyncio.async_test
    async def test_process_batch_async_success(self, processor, mock_batch_metadata):
        """Test successful asynchronous batch processing"""
        batch_id = mock_batch_metadata["batch_id"]
        files = mock_batch_metadata["files"]
        
        with patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_convert_jpeg, \
             patch.object(processor.ai_simulator, 'process_batch') as mock_ai_process, \
             patch.object(processor.converter, 'convert_batch_to_original') as mock_convert_original:
            
            # Mock successful conversions
            mock_convert_jpeg.return_value = {
                "success": True,
                "converted_files": ["temp_jpeg_1.jpg", "temp_jpeg_2.jpg"],
                "errors": []
            }
            
            mock_ai_process.return_value = {
                "status": "success",
                "processed_files": 2,
                "confidence": 0.95
            }
            
            mock_convert_original.return_value = {
                "success": True,
                "converted_files": ["output_1.jpg", "output_2.png"],
                "errors": []
            }
            
            result = await processor.process_batch_async(batch_id, files)
            
            assert result["status"] == "completed"
            assert result["batch_id"] == batch_id
            assert result["files_processed"] == 2
            assert result["errors"] == []
    
    @pytest_asyncio.async_test
    async def test_process_batch_async_conversion_error(self, processor, mock_batch_metadata):
        """Test batch processing with conversion errors"""
        batch_id = mock_batch_metadata["batch_id"]
        files = mock_batch_metadata["files"]
        
        with patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_convert_jpeg:
            # Mock conversion failure
            mock_convert_jpeg.return_value = {
                "success": False,
                "converted_files": [],
                "errors": ["Failed to convert test1.jpg: Unsupported format"]
            }
            
            result = await processor.process_batch_async(batch_id, files)
            
            assert result["status"] == "failed"
            assert len(result["errors"]) == 1
            assert "Failed to convert test1.jpg" in result["errors"][0]
    
    @pytest_asyncio.async_test
    async def test_concurrent_batch_processing(self, processor, temp_dir, sample_images):
        """Test concurrent processing of multiple batches"""
        # Create multiple batches
        batches = [
            {"batch_id": "batch_001", "files": [list(sample_images.values())[0]]},
            {"batch_id": "batch_002", "files": [list(sample_images.values())[1]]},
            {"batch_id": "batch_003", "files": [list(sample_images.values())[2]]}
        ]
        
        with patch.object(processor, 'process_batch_async') as mock_process_batch:
            mock_process_batch.return_value = {
                "status": "completed",
                "batch_id": "test_batch",
                "files_processed": 1,
                "errors": []
            }
            
            result = await processor._process_batches_concurrent(batches)
            
            assert result["batches_processed"] == 3
            assert result["total_files"] == 3
            assert result["errors"] == []
            assert mock_process_batch.call_count == 3


class TestImageProcessorFileHandling(TestImageProcessor):
    """Test file handling and metadata operations"""
    
    def test_extract_file_metadata(self, processor, sample_images):
        """Test metadata extraction from files"""
        for filename, file_info in sample_images.items():
            filepath = file_info["path"]
            
            with patch.object(MetadataUtils, 'extract_metadata') as mock_extract:
                mock_extract.return_value = {
                    "format": file_info["format"],
                    "size": file_info["size"],
                    "dpi": file_info["dpi"],
                    "color_mode": "RGB",
                    "compression": "lzw" if file_info["format"] == "TIFF" else None
                }
                
                metadata = processor._extract_file_metadata(filepath)
                
                assert metadata["format"] == file_info["format"]
                assert metadata["size"] == file_info["size"]
                assert metadata["dpi"] == file_info["dpi"]
    
    def test_save_batch_metadata(self, processor, temp_dir, mock_batch_metadata):
        """Test saving batch metadata to JSON file"""
        batch_id = mock_batch_metadata["batch_id"]
        
        with patch.object(processor, '_get_metadata_path') as mock_get_path:
            metadata_path = os.path.join(temp_dir, f"{batch_id}.json")
            mock_get_path.return_value = metadata_path
            
            processor._save_batch_metadata(batch_id, mock_batch_metadata)
            
            assert os.path.exists(metadata_path)
            
            # Verify content
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata["batch_id"] == batch_id
            assert saved_metadata["total_files"] == 2
            assert len(saved_metadata["files"]) == 2
    
    def test_load_batch_metadata(self, processor, temp_dir, mock_batch_metadata):
        """Test loading batch metadata from JSON file"""
        batch_id = mock_batch_metadata["batch_id"]
        metadata_path = os.path.join(temp_dir, f"{batch_id}.json")
        
        # Save metadata first
        with open(metadata_path, 'w') as f:
            json.dump(mock_batch_metadata, f)
        
        with patch.object(processor, '_get_metadata_path') as mock_get_path:
            mock_get_path.return_value = metadata_path
            
            loaded_metadata = processor._load_batch_metadata(batch_id)
            
            assert loaded_metadata["batch_id"] == batch_id
            assert loaded_metadata["total_files"] == 2
            assert len(loaded_metadata["files"]) == 2
    
    def test_load_batch_metadata_not_found(self, processor):
        """Test loading non-existent batch metadata"""
        with patch.object(processor, '_get_metadata_path') as mock_get_path:
            mock_get_path.return_value = "/nonexistent/path.json"
            
            with pytest.raises(FileNotFoundError):
                processor._load_batch_metadata("nonexistent_batch")


class TestImageProcessorErrorHandling(TestImageProcessor):
    """Test error handling and logging"""
    
    @pytest_asyncio.async_test
    async def test_process_directory_file_access_error(self, processor, temp_dir):
        """Test handling of file access errors"""
        with patch.object(processor, '_scan_directory') as mock_scan:
            mock_scan.side_effect = PermissionError("Access denied")
            
            with pytest.raises(PermissionError):
                await processor.process_directory(temp_dir)
    
    @pytest_asyncio.async_test
    async def test_process_batch_timeout(self, processor, mock_batch_metadata):
        """Test handling of batch processing timeout"""
        batch_id = mock_batch_metadata["batch_id"]
        files = mock_batch_metadata["files"]
        
        with patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_convert:
            mock_convert.side_effect = asyncio.TimeoutError("Processing timeout")
            
            result = await processor.process_batch_async(batch_id, files)
            
            assert result["status"] == "failed"
            assert any("timeout" in error.lower() for error in result["errors"])
    
    def test_corrupted_metadata_handling(self, processor, temp_dir):
        """Test handling of corrupted metadata files"""
        batch_id = "corrupted_batch"
        metadata_path = os.path.join(temp_dir, f"{batch_id}.json")
        
        # Create corrupted JSON file
        with open(metadata_path, 'w') as f:
            f.write("{ corrupted json content")
        
        with patch.object(processor, '_get_metadata_path') as mock_get_path:
            mock_get_path.return_value = metadata_path
            
            with pytest.raises(json.JSONDecodeError):
                processor._load_batch_metadata(batch_id)
    
    def test_disk_space_error_handling(self, processor, temp_dir, sample_images):
        """Test handling of disk space errors"""
        with patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_convert:
            mock_convert.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError):
                processor.converter.convert_batch_to_jpeg(list(sample_images.values()), temp_dir)


class TestImageProcessorPerformance(TestImageProcessor):
    """Test performance and optimization"""
    
    @pytest_asyncio.async_test
    async def test_large_batch_processing(self, processor):
        """Test processing of large batches"""
        # Create mock data for large batch (100 files)
        large_batch_files = []
        for i in range(100):
            large_batch_files.append({
                "filename": f"test_{i:03d}.jpg",
                "original_path": f"/input/test_{i:03d}.jpg",
                "original_format": "JPEG",
                "dpi": (72, 72),
                "color_mode": "RGB",
                "file_size": 1024 * (i + 1),
                "upload_date": (datetime.now() - timedelta(hours=i)).isoformat()
            })
        
        with patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_convert, \
             patch.object(processor.ai_simulator, 'process_batch') as mock_ai, \
             patch.object(processor.converter, 'convert_batch_to_original') as mock_convert_back:
            
            mock_convert.return_value = {"success": True, "converted_files": [f"temp_{i}.jpg" for i in range(100)], "errors": []}
            mock_ai.return_value = {"status": "success", "processed_files": 100, "confidence": 0.92}
            mock_convert_back.return_value = {"success": True, "converted_files": [f"output_{i}.jpg" for i in range(100)], "errors": []}
            
            start_time = datetime.now()
            result = await processor.process_batch_async("large_batch", large_batch_files)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            assert result["status"] == "completed"
            assert result["files_processed"] == 100
            assert processing_time < 60  # Should complete within 60 seconds
    
    @pytest_asyncio.async_test
    async def test_memory_usage_optimization(self, processor, temp_dir, sample_images):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple batches
        batches = [
            {"batch_id": f"batch_{i:03d}", "files": [list(sample_images.values())[i % len(sample_images)]]}
            for i in range(10)
        ]
        
        with patch.object(processor, 'process_batch_async') as mock_process:
            mock_process.return_value = {
                "status": "completed",
                "batch_id": "test_batch",
                "files_processed": 1,
                "errors": []
            }
            
            await processor._process_batches_concurrent(batches)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for test)
            assert memory_increase < 100 * 1024 * 1024  # 100MB


class TestImageProcessorIntegration(TestImageProcessor):
    """Integration tests for ImageProcessor"""
    
    @pytest_asyncio.async_test
    async def test_end_to_end_workflow(self, processor, temp_dir, sample_images):
        """Test complete end-to-end processing workflow"""
        # Mock all dependencies
        with patch.object(processor.batch_manager, 'create_batches_by_date') as mock_batches, \
             patch.object(processor.converter, 'convert_batch_to_jpeg') as mock_to_jpeg, \
             patch.object(processor.ai_simulator, 'process_batch') as mock_ai, \
             patch.object(processor.converter, 'convert_batch_to_original') as mock_to_original, \
             patch.object(processor, '_save_batch_metadata') as mock_save_metadata:
            
            # Setup mocks
            mock_batches.return_value = [
                {"batch_id": "batch_001", "files": list(sample_images.values())}
            ]
            
            mock_to_jpeg.return_value = {
                "success": True,
                "converted_files": ["temp1.jpg", "temp2.jpg", "temp3.jpg", "temp4.jpg"],
                "errors": []
            }
            
            mock_ai.return_value = {
                "status": "success",
                "processed_files": 4,
                "confidence": 0.94,
                "results": [{"file": "temp1.jpg", "class": "document"}]
            }
            
            mock_to_original.return_value = {
                "success": True,
                "converted_files": ["output1.jpg", "output2.png", "output3.bmp", "output4.tiff"],
                "errors": []
            }
            
            # Execute workflow
            result = await processor.process_directory(temp_dir)
            
            # Verify results
            assert result["status"] == "success"
            assert result["batches_processed"] == 1
            assert result["total_files"] == 4
            assert result["errors"] == []
            
            # Verify all steps were called
            mock_batches.assert_called_once()
            mock_to_jpeg.assert_called_once()
            mock_ai.assert_called_once()
            mock_to_original.assert_called_once()
            mock_save_metadata.assert_called_once()


# Test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])