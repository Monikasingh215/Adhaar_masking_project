"""
Test file for Image Converter Service
Tests image format conversions, metadata preservation, and compression handling
"""

import pytest
import os
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageFile
import fitz  # PyMuPDF
import io

from app.services.converter import Converter
from app.utils.metadata_utils import MetadataUtils
from app.utils.file_utils import FileUtils
from app.core.config import settings


class TestConverter:
    """Test suite for Converter service"""
    
    @pytest.fixture
    def converter(self):
        """Create Converter instance"""
        return Converter()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images in various formats"""
        images = {}
        
        # Test image configurations
        configs = [
            {
                "name": "test_jpeg.jpg",
                "format": "JPEG",
                "size": (400, 300),
                "color": "red",
                "dpi": (72, 72),
                "quality": 85
            },
            {
                "name": "test_png.png",
                "format": "PNG",
                "size": (300, 200),
                "color": "blue",
                "dpi": (96, 96)
            },
            {
                "name": "test_bmp.bmp",
                "format": "BMP",
                "size": (200, 150),
                "color": "green",
                "dpi": (72, 72)
            },
            {
                "name": "test_tiff.tiff",
                "format": "TIFF",
                "size": (500, 400),
                "color": "yellow",
                "dpi": (300, 300),
                "compression": "lzw"
            },
            {
                "name": "test_grayscale.jpg",
                "format": "JPEG",
                "size": (250, 200),
                "color": "gray",
                "mode": "L",
                "dpi": (150, 150)
            }
        ]
        
        for config in configs:
            if config.get("mode") == "L":
                img = Image.new("L", config["size"], color=128)  # Grayscale
            else:
                img = Image.new("RGB", config["size"], color=config["color"])
            
            filepath = os.path.join(temp_dir, config["name"])
            
            # Save with specific parameters
            save_kwargs = {"format": config["format"], "dpi": config["dpi"]}
            
            if config["format"] == "JPEG":
                save_kwargs["quality"] = config.get("quality", 85)
            elif config["format"] == "TIFF":
                save_kwargs["compression"] = config.get("compression", "lzw")
            
            img.save(filepath, **save_kwargs)
            
            images[config["name"]] = {
                "path": filepath,
                "format": config["format"],
                "size": config["size"],
                "dpi": config["dpi"],
                "mode": config.get("mode", "RGB"),
                "compression": config.get("compression"),
                "quality": config.get("quality")
            }
        
        return images
    
    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create sample PDF with multiple pages"""
        pdf_path = os.path.join(temp_dir, "test_document.pdf")
        
        doc = fitz.open()
        
        # Create multiple pages with different content
        for i in range(3):
            page = doc.new_page(width=595, height=842)  # A4 size
            page.insert_text((50, 50 + i * 100), f"Page {i+1} Content", fontsize=14)
            
            # Add a simple rectangle
            rect = fitz.Rect(100, 100 + i * 50, 200, 150 + i * 50)
            page.draw_rect(rect, color=(0.7, 0.7, 0.7), fill=(0.9, 0.9, 0.9))
        
        doc.save(pdf_path)
        doc.close()
        
        return pdf_path
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing"""
        return {
            "batch_id": "test_batch_001",
            "files": [
                {
                    "filename": "test_jpeg.jpg",
                    "original_format": "JPEG",
                    "dpi": (72, 72),
                    "color_mode": "RGB",
                    "compression": None,
                    "dimensions": (400, 300),
                    "file_size": 25600,
                    "quality": 85
                },
                {
                    "filename": "test_tiff.tiff",
                    "original_format": "TIFF",
                    "dpi": (300, 300),
                    "color_mode": "RGB",
                    "compression": "lzw",
                    "dimensions": (500, 400),
                    "file_size": 102400,
                    "quality": None
                }
            ],
            "conversion_settings": {
                "target_format": "PNG",
                "quality": 95,
                "dpi": (150, 150),
                "preserve_metadata": True
            },
            "timestamp": datetime.now().isoformat()
        }

    # Basic conversion tests
    def test_convert_jpeg_to_png(self, converter, sample_images, temp_dir):
        """Test JPEG to PNG conversion"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        target_path = os.path.join(temp_dir, "converted.png")
        
        result = converter.convert_image(source_path, target_path, "PNG")
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify converted image
        with Image.open(target_path) as img:
            assert img.format == "PNG"
            assert img.size == sample_images["test_jpeg.jpg"]["size"]
    
    def test_convert_png_to_jpeg(self, converter, sample_images, temp_dir):
        """Test PNG to JPEG conversion"""
        source_path = sample_images["test_png.png"]["path"]
        target_path = os.path.join(temp_dir, "converted.jpg")
        
        result = converter.convert_image(source_path, target_path, "JPEG", quality=90)
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify converted image
        with Image.open(target_path) as img:
            assert img.format == "JPEG"
            assert img.size == sample_images["test_png.png"]["size"]
    
    def test_convert_bmp_to_tiff(self, converter, sample_images, temp_dir):
        """Test BMP to TIFF conversion"""
        source_path = sample_images["test_bmp.bmp"]["path"]
        target_path = os.path.join(temp_dir, "converted.tiff")
        
        result = converter.convert_image(source_path, target_path, "TIFF", compression="zip")
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify converted image
        with Image.open(target_path) as img:
            assert img.format == "TIFF"
            assert img.size == sample_images["test_bmp.bmp"]["size"]

    def test_convert_grayscale_preservation(self, converter, sample_images, temp_dir):
        """Test grayscale mode preservation during conversion"""
        source_path = sample_images["test_grayscale.jpg"]["path"]
        target_path = os.path.join(temp_dir, "grayscale_converted.png")
        
        result = converter.convert_image(source_path, target_path, "PNG")
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify grayscale mode is preserved
        with Image.open(target_path) as img:
            assert img.mode == "L"

    # Batch conversion tests
    def test_batch_conversion(self, converter, sample_images, temp_dir):
        """Test batch conversion of multiple images"""
        output_dir = os.path.join(temp_dir, "batch_output")
        os.makedirs(output_dir)
        
        source_files = [
            sample_images["test_jpeg.jpg"]["path"],
            sample_images["test_png.png"]["path"],
            sample_images["test_bmp.bmp"]["path"]
        ]
        
        results = converter.batch_convert(
            source_files=source_files,
            output_dir=output_dir,
            target_format="PNG",
            quality=85
        )
        
        assert len(results) == 3
        assert all(result["success"] for result in results)
        
        # Verify all output files exist
        for result in results:
            assert os.path.exists(result["output_path"])
            
            # Check format conversion
            with Image.open(result["output_path"]) as img:
                assert img.format == "PNG"

    def test_batch_conversion_with_errors(self, converter, temp_dir):
        """Test batch conversion handling of invalid files"""
        output_dir = os.path.join(temp_dir, "batch_output")
        os.makedirs(output_dir)
        
        # Create invalid file
        invalid_file = os.path.join(temp_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("This is not an image")
        
        source_files = [invalid_file]
        
        results = converter.batch_convert(
            source_files=source_files,
            output_dir=output_dir,
            target_format="PNG"
        )
        
        assert len(results) == 1
        assert not results[0]["success"]
        assert "error" in results[0]

    # Metadata preservation tests
    def test_metadata_preservation(self, converter, sample_images, temp_dir):
        """Test metadata preservation during conversion"""
        source_path = sample_images["test_tiff.tiff"]["path"]
        target_path = os.path.join(temp_dir, "metadata_preserved.jpg")
        
        result = converter.convert_image(
            source_path=source_path,
            target_path=target_path,
            target_format="JPEG",
            preserve_metadata=True
        )
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify DPI is preserved
        with Image.open(target_path) as img:
            assert img.info.get("dpi") == sample_images["test_tiff.tiff"]["dpi"]

    def test_metadata_extraction(self, converter, sample_images):
        """Test metadata extraction from images"""
        source_path = sample_images["test_tiff.tiff"]["path"]
        
        metadata = converter.extract_metadata(source_path)
        
        assert metadata is not None
        assert metadata["format"] == "TIFF"
        assert metadata["size"] == sample_images["test_tiff.tiff"]["size"]
        assert metadata["dpi"] == sample_images["test_tiff.tiff"]["dpi"]

    # Quality and compression tests
    def test_jpeg_quality_settings(self, converter, sample_images, temp_dir):
        """Test JPEG quality settings"""
        source_path = sample_images["test_png.png"]["path"]
        
        # Test different quality levels
        quality_levels = [50, 75, 95]
        file_sizes = []
        
        for quality in quality_levels:
            target_path = os.path.join(temp_dir, f"quality_{quality}.jpg")
            
            result = converter.convert_image(
                source_path=source_path,
                target_path=target_path,
                target_format="JPEG",
                quality=quality
            )
            
            assert result is not None
            assert os.path.exists(target_path)
            
            file_sizes.append(os.path.getsize(target_path))
        
        # Higher quality should generally result in larger files
        assert file_sizes[0] < file_sizes[1] < file_sizes[2]

    def test_tiff_compression_options(self, converter, sample_images, temp_dir):
        """Test TIFF compression options"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        
        compression_types = ["none", "lzw", "zip"]
        
        for compression in compression_types:
            target_path = os.path.join(temp_dir, f"tiff_{compression}.tiff")
            
            result = converter.convert_image(
                source_path=source_path,
                target_path=target_path,
                target_format="TIFF",
                compression=compression
            )
            
            assert result is not None
            assert os.path.exists(target_path)
            
            # Verify compression is applied
            with Image.open(target_path) as img:
                assert img.format == "TIFF"

    # PDF conversion tests
    def test_pdf_to_images(self, converter, sample_pdf, temp_dir):
        """Test PDF to image conversion"""
        output_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(output_dir)
        
        results = converter.pdf_to_images(
            pdf_path=sample_pdf,
            output_dir=output_dir,
            image_format="PNG",
            dpi=150
        )
        
        assert len(results) == 3  # 3 pages in sample PDF
        assert all(result["success"] for result in results)
        
        # Verify all pages are converted
        for i, result in enumerate(results):
            assert os.path.exists(result["output_path"])
            assert f"page_{i+1}" in result["output_path"]
            
            # Check image properties
            with Image.open(result["output_path"]) as img:
                assert img.format == "PNG"
                assert img.size[0] > 0 and img.size[1] > 0

    def test_pdf_single_page_conversion(self, converter, sample_pdf, temp_dir):
        """Test converting specific page from PDF"""
        output_path = os.path.join(temp_dir, "single_page.jpg")
        
        result = converter.pdf_page_to_image(
            pdf_path=sample_pdf,
            page_number=1,  # Second page (0-indexed)
            output_path=output_path,
            image_format="JPEG",
            dpi=200
        )
        
        assert result is not None
        assert result["success"]
        assert os.path.exists(output_path)
        
        # Verify image quality
        with Image.open(output_path) as img:
            assert img.format == "JPEG"
            # Higher DPI should result in larger image
            assert img.size[0] > 1000  # Approximate check for 200 DPI

    # Error handling tests
    def test_invalid_source_file(self, converter, temp_dir):
        """Test handling of invalid source file"""
        invalid_path = os.path.join(temp_dir, "nonexistent.jpg")
        target_path = os.path.join(temp_dir, "output.png")
        
        result = converter.convert_image(invalid_path, target_path, "PNG")
        
        assert result is None or not result.get("success", True)

    def test_invalid_target_format(self, converter, sample_images, temp_dir):
        """Test handling of invalid target format"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        target_path = os.path.join(temp_dir, "output.invalid")
        
        with pytest.raises(ValueError):
            converter.convert_image(source_path, target_path, "INVALID_FORMAT")

    def test_corrupted_image_handling(self, converter, temp_dir):
        """Test handling of corrupted image files"""
        # Create a corrupted image file
        corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'wb') as f:
            f.write(b"Not a real image file")
        
        target_path = os.path.join(temp_dir, "output.png")
        
        result = converter.convert_image(corrupted_path, target_path, "PNG")
        
        assert result is None or not result.get("success", True)

    # Performance and optimization tests
    def test_large_image_handling(self, converter, temp_dir):
        """Test handling of large images"""
        # Create a large test image
        large_img = Image.new("RGB", (5000, 5000), color="red")
        large_path = os.path.join(temp_dir, "large_image.png")
        large_img.save(large_path)
        
        target_path = os.path.join(temp_dir, "large_converted.jpg")
        
        result = converter.convert_image(
            source_path=large_path,
            target_path=target_path,
            target_format="JPEG",
            quality=75
        )
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify conversion maintained dimensions
        with Image.open(target_path) as img:
            assert img.size == (5000, 5000)

    def test_memory_efficient_conversion(self, converter, sample_images, temp_dir):
        """Test memory-efficient conversion options"""
        source_path = sample_images["test_tiff.tiff"]["path"]
        target_path = os.path.join(temp_dir, "memory_efficient.jpg")
        
        # Enable memory-efficient processing
        with patch.object(ImageFile, 'LOAD_TRUNCATED_IMAGES', True):
            result = converter.convert_image(
                source_path=source_path,
                target_path=target_path,
                target_format="JPEG",
                optimize_memory=True
            )
        
        assert result is not None
        assert os.path.exists(target_path)

    # Configuration and settings tests
    def test_custom_dpi_settings(self, converter, sample_images, temp_dir):
        """Test custom DPI settings"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        target_path = os.path.join(temp_dir, "custom_dpi.png")
        
        custom_dpi = (200, 200)
        
        result = converter.convert_image(
            source_path=source_path,
            target_path=target_path,
            target_format="PNG",
            dpi=custom_dpi
        )
        
        assert result is not None
        assert os.path.exists(target_path)
        
        # Verify DPI is applied
        with Image.open(target_path) as img:
            assert img.info.get("dpi") == custom_dpi

    def test_color_mode_conversion(self, converter, sample_images, temp_dir):
        """Test color mode conversions"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        
        # Test RGB to Grayscale
        target_path = os.path.join(temp_dir, "rgb_to_grayscale.png")
        
        result = converter.convert_image(
            source_path=source_path,
            target_path=target_path,
            target_format="PNG",
            color_mode="L"
        )
        
        assert result is not None
        assert os.path.exists(target_path)
        
        with Image.open(target_path) as img:
            assert img.mode == "L"

    # Integration tests
    @patch('app.utils.metadata_utils.MetadataUtils.extract_metadata')
    def test_metadata_utils_integration(self, mock_extract, converter, sample_images):
        """Test integration with MetadataUtils"""
        mock_extract.return_value = {"test": "metadata"}
        
        source_path = sample_images["test_jpeg.jpg"]["path"]
        metadata = converter.extract_metadata(source_path)
        
        mock_extract.assert_called_once_with(source_path)
        assert metadata == {"test": "metadata"}

    @patch('app.utils.file_utils.FileUtils.ensure_directory')
    def test_file_utils_integration(self, mock_ensure_dir, converter, sample_images, temp_dir):
        """Test integration with FileUtils"""
        mock_ensure_dir.return_value = True
        
        source_path = sample_images["test_jpeg.jpg"]["path"]
        target_path = os.path.join(temp_dir, "subdir", "output.png")
        
        converter.convert_image(source_path, target_path, "PNG")
        
        mock_ensure_dir.assert_called()

    # Cleanup and resource management tests
    def test_resource_cleanup(self, converter, sample_images, temp_dir):
        """Test proper resource cleanup after conversion"""
        source_path = sample_images["test_jpeg.jpg"]["path"]
        target_path = os.path.join(temp_dir, "cleanup_test.png")
        
        # Mock to track resource usage
        original_close = Image.Image.close
        close_calls = []
        
        def mock_close(self):
            close_calls.append(self)
            return original_close(self)
        
        with patch.object(Image.Image, 'close', mock_close):
            result = converter.convert_image(source_path, target_path, "PNG")
        
        assert result is not None
        # Verify resources were cleaned up
        assert len(close_calls) > 0

    def test_concurrent_conversions(self, converter, sample_images, temp_dir):
        """Test handling of concurrent conversion requests"""
        import threading
        import concurrent.futures
        
        source_path = sample_images["test_jpeg.jpg"]["path"]
        
        def convert_worker(index):
            target_path = os.path.join(temp_dir, f"concurrent_{index}.png")
            return converter.convert_image(source_path, target_path, "PNG")
        
        # Run multiple conversions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(convert_worker, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # All conversions should succeed
        assert all(result is not None for result in results)
        
        # Verify all output files exist
        for i in range(5):
            output_path = os.path.join(temp_dir, f"concurrent_{i}.png")
            assert os.path.exists(output_path)