from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Application settings
    app_name: str = "FastAPI Image Processor"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Processing settings
    batch_size: int = 20
    max_workers: int = 20
    
    # File paths
    base_dir: Path = Path(__file__).parent.parent.parent
    input_dir: Path = base_dir / "data" / "input"
    output_dir: Path = base_dir / "data" / "output"
    temp_dir: Path = base_dir / "data" / "temp"
    metadata_dir: Path = base_dir / "data" / "metadata"
    logs_dir: Path = base_dir / "data" / "logs"
    
    # Supported formats
    supported_formats: list = [".tiff", ".tif", ".pdf", ".png", ".bmp", ".jpg", ".jpeg"]
    
    # Image processing settings
    jpeg_quality: int = 95
    default_dpi: tuple = (300, 300)
    
    # AI model settings
    ai_model_name: str = "masking_model"
    ai_processing_delay: float = 0.5  # Simulated processing delay
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.input_dir,
            self.output_dir,
            self.temp_dir,
            self.temp_dir / "jpeg_converted",
            self.metadata_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()