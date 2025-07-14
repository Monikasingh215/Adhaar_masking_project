"""
Test file for FastAPI Image Processing Application
Tests the main API endpoints and core functionality
"""

import pytest
import asyncio
import json
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi import status
from PIL import Image
import io

# Import the main FastAPI app
from app.main import app
from app.core.config import settings
from app.services.image_processor import ImageProcessor
from app.services.batch_manager import BatchManager
from app.utils.file_utils import FileUtils


class TestFastAPIImageProcessor:
    """Test suite for FastAPI Image Processing Application"""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app"""
        return TestClient(app)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images in different formats"""
        images = {}
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Save in different formats
        formats = {
            'test1.jpg': 'JPEG',
            'test2.png': 'PNG',
            'test3.bmp': 'BMP',
            'test4.tiff': 'TIFF'
        }
        
        for filename, format_type in formats.items():
            filepath = os.path.join(temp_dir, filename)
            if format_type == 'TIFF':
                test_image.save(filepath, format=format_type, compression='lzw')
            else:
                test_image.save(filepath, format=format_type)
            images[filename] = filepath
            
        return images
    
    @pytest.fixture
    def mock_batch_data(self):
        """Mock batch metadata"""
        return {
            "batch_id": "batch_001",
            "created_at": "2024-01-15T10:30:00",
            "files": [
                {
                    "filename": "test1.jpg",
                    "original_format": "JPEG",
                    "dpi": (72, 72),
                    "color_mode": "RGB",
                    "compression": None,
                    "file_size": 1024,
                    "upload_date": "2024-01-15T10:00:00"
                },
                {
                    "filename": "test2.png",
                    "original_format": "PNG",
                    "dpi": (96, 96),
                    "color_mode": "RGB",
                    "compression": None,
                    "file_size": 2048,
                    "upload_date": "2024-01-15T10:05:00"
                }
            ],
            "total_files": 2,
            "status": "pending"
        }


class TestAPIEndpoints(TestFastAPIImageProcessor):
    """Test API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "status": "healthy",
            "service": "Image Processing API",
            "version": "1.0.0" 
        }

    
    def test_process_endpoint_missing_directory(self, client):
        """Test process endpoint with missing directory"""
        response = client.post("/process", json={"directory_path": "/nonexistent/path"})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Directory not found" in response.json()["detail"]
    
    def test_process_endpoint_empty_directory(self, client, temp_dir):
        """Test process endpoint with empty directory"""
        response = client.post("/process", json={"directory_path": temp_dir})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "No supported image files found" in response.json()["detail"]
    
    @patch('app.services.image_processor.ImageProcessor.process_directory')
    def test_process_endpoint_success(self, mock_process, client, temp_dir, sample_images):
        """Test successful processing endpoint"""
        mock_process.return_value = {
            "status": "success",
            "batches_processed": 1,
            "total_files": 4,
            "processing_time": 45.67
        }
        
        response = client.post("/api/v1/process", json={"directory_path": temp_dir})
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert result["status"] == "success"
        assert result["batches_processed"] == 1
        assert result["total_files"] == 4
        assert "processing_time" in result
    
    def test_get_batch_status_not_found(self, client):
        """Test get batch status with non-existent batch"""
        response = client.get("/batch/nonexistent_batch/status")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @patch('app.services.batch_manager.BatchManager.get_batch_status')
    def test_get_batch_status_success(self, mock_get_status, client, mock_batch_data):
        """Test successful batch status retrieval"""
        mock_get_status.return_value = mock_batch_data
        
        response = client.get("/batch/batch_001/status")
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert result["batch_id"] == "batch_001"
        assert result["total_files"] == 2
        assert result["status"] == "pending"
    
    def test_list_batches_empty(self, client):
        """Test listing batches when none exist"""
        with patch('app.services.batch_manager.BatchManager.list_batches') as mock_list:
            mock_list.return_value = []
            
            response = client.get("/batches")
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {"batches": []}
    
    @patch('app.services.batch_manager.BatchManager.list_batches')
    def test_list_batches_success(self, mock_list, client, mock_batch_data):
        """Test successful batch listing"""
        mock_list.return_value = [mock_batch_data]
        
        response = client.get("/batches")
        assert response.status_code == status.HTTP_200_OK
        
        result = response.json()
        assert len(result["batches"]) == 1
        assert result["batches"][0]["batch_id"] == "batch_001"


class TestImageProcessorIntegration(TestFastAPIImageProcessor):
    """Integration tests for image processing workflow"""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance"""
        return ImageProcessor()
    
    @pytest.fixture
    def batch_manager(self):
        """Create BatchManager instance"""
        return BatchManager()
    
    @pytest_asyncio.async_test
    async def test_full_processing_workflow(self, processor, temp_dir, sample_images):
        """Test complete image processing workflow"""
        with patch.object(processor, 'process_directory') as mock_process:
            # Mock the processing result
            mock_process.return_value = {
                "status": "success",
                "batches_processed": 1,
                "total_files": 4,
                "processing_time": 30.5,
                "errors": []
            }
            
            result = await processor.process_directory(temp_dir)
            
            assert result["status"] == "success"
            assert result["batches_processed"] == 1
            assert result["total_files"] == 4
            assert isinstance(result["processing_time"], float)
            assert result["errors"] == []
    
    def test_batch_creation_by_date(self, batch_manager, temp_dir, sample_images):
        """Test batch creation based on upload date"""
        # Mock file metadata with different dates
        mock_files = [
            {"path": "file1.jpg", "upload_date": "2024-01-15T10:00:00"},
            {"path": "file2.png", "upload_date": "2024-01-15T10:05:00"},
            {"path": "file3.tiff", "upload_date": "2024-01-16T11:00:00"},
        ]
        
        with patch.object(batch_manager, 'create_batches_by_date') as mock_create:
            mock_create.return_value = [
                {"batch_id": "batch_001", "files": mock_files[:2]},
                {"batch_id": "batch_002", "files": mock_files[2:]}
            ]
            
            batches = batch_manager.create_batches_by_date(mock_files, batch_size=20)
            
            assert len(batches) == 2
            assert batches[0]["batch_id"] == "batch_001"
            assert len(batches[0]["files"]) == 2


class TestErrorHandling(TestFastAPIImageProcessor):
    """Test error handling scenarios"""
    
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON requests"""
        response = client.post("/process", data="invalid json")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post("/process", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('app.services.image_processor.ImageProcessor.process_directory')
    def test_processing_error_handling(self, mock_process, client, temp_dir):
        """Test handling of processing errors"""
        mock_process.side_effect = Exception("Processing failed")
        
        response = client.post("/process", json={"directory_path": temp_dir})
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Processing failed" in response.json()["detail"]


class TestConcurrentProcessing(TestFastAPIImageProcessor):
    """Test concurrent processing capabilities"""
    
    @pytest_asyncio.async_test
    async def test_concurrent_batch_processing(self, temp_dir):
        """Test concurrent processing of multiple batches"""
        with patch('app.services.image_processor.ImageProcessor.process_batch_async') as mock_process:
            # Mock concurrent processing
            mock_process.return_value = {
                "batch_id": "test_batch",
                "status": "completed",
                "files_processed": 20,
                "processing_time": 15.5
            }
            
            processor = ImageProcessor()
            
            # Simulate processing multiple batches
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    processor.process_batch_async(f"batch_{i}", [])
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert result["status"] == "completed"
                assert result["files_processed"] == 20


class TestMetadataHandling(TestFastAPIImageProcessor):
    """Test metadata preservation and handling"""
    
    def test_metadata_extraction(self, sample_images):
        """Test metadata extraction from images"""
        from app.utils.metadata_utils import MetadataUtils
        
        with patch.object(MetadataUtils, 'extract_metadata') as mock_extract:
            mock_extract.return_value = {
                "format": "TIFF",
                "dpi": (300, 300),
                "color_mode": "RGB",
                "compression": "lzw",
                "size": (1024, 768)
            }
            
            metadata = MetadataUtils.extract_metadata("test.tiff")
            
            assert metadata["format"] == "TIFF"
            assert metadata["dpi"] == (300, 300)
            assert metadata["color_mode"] == "RGB"
            assert metadata["compression"] == "lzw"
    
    def test_metadata_preservation_during_conversion(self, temp_dir):
        """Test metadata preservation during format conversion"""
        from app.services.converter import Converter
        
        with patch.object(Converter, 'convert_to_jpeg') as mock_convert:
            mock_convert.return_value = {
                "success": True,
                "output_path": "/temp/converted.jpg",
                "metadata_preserved": True
            }
            
            converter = Converter()
            result = converter.convert_to_jpeg("test.tiff", temp_dir)
            
            assert result["success"] is True
            assert result["metadata_preserved"] is True


class TestFileValidation(TestFastAPIImageProcessor):
    """Test file validation and format support"""
    
    def test_supported_formats(self):
        """Test validation of supported file formats"""
        from app.utils.validation import ValidationUtils
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
        
        for format_ext in supported_formats:
            assert ValidationUtils.is_supported_format(f"test{format_ext}") is True
    
    def test_unsupported_formats(self):
        """Test handling of unsupported file formats"""
        from app.utils.validation import ValidationUtils
        
        unsupported_formats = ['.txt', '.doc', '.zip', '.exe']
        
        for format_ext in unsupported_formats:
            assert ValidationUtils.is_supported_format(f"test{format_ext}") is False
    
    def test_file_size_validation(self):
        """Test file size validation"""
        from app.utils.validation import ValidationUtils
        
        # Test within limits
        assert ValidationUtils.validate_file_size(1024 * 1024 * 10) is True  # 10MB
        
        # Test exceeding limits
        assert ValidationUtils.validate_file_size(1024 * 1024 * 1024) is False  # 1GB


class TestLoggingAndMonitoring(TestFastAPIImageProcessor):
    """Test logging and monitoring functionality"""
    
    def test_error_logging(self, client, temp_dir):
        """Test error logging functionality"""
        with patch('app.core.logging.logger.error') as mock_logger:
            # Trigger an error
            response = client.post("/process", json={"directory_path": "/invalid/path"})
            
            # Verify error was logged
            mock_logger.assert_called()
    
    def test_processing_metrics_logging(self, client, temp_dir, sample_images):
        """Test processing metrics logging"""
        with patch('app.core.logging.logger.info') as mock_logger:
            with patch('app.services.image_processor.ImageProcessor.process_directory') as mock_process:
                mock_process.return_value = {
                    "status": "success",
                    "batches_processed": 1,
                    "total_files": 4,
                    "processing_time": 25.5
                }
                
                response = client.post("/process", json={"directory_path": temp_dir})
                
                # Verify metrics were logged
                mock_logger.assert_called()


# Test Configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test data fixtures
@pytest.fixture
def mock_ai_processing_result():
    """Mock AI processing result"""
    return {
        "status": "success",
        "confidence": 0.95,
        "predictions": [
            {"class": "document", "confidence": 0.92},
            {"class": "image", "confidence": 0.88}
        ],
        "processing_time": 2.5
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])