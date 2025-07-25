
import json
from pathlib import Path

def read_json_config(filename):
    config_path = Path(filename)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)
    
def get_max_retries():
    config = read_json_config("aadhaar_start.json")
    return config.get("max_retries", 3)

def get_retry_delay():
    config = read_json_config("aadhaar_start.json")
    return config.get("retry_delay_seconds", 5)


def is_cold_stop():
    config = read_json_config("aadhaar_stop.json")
    return config.get("cold_stop", 0) == 1

def get_batch_size():
    config = read_json_config("aadhaar_start.json")
    return config.get("batch_size", 20)