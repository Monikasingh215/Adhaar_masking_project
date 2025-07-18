def ensure_directories(settings):
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)

