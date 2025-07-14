
import json
from pathlib import Path

def read_json_config(filename):
    config_path = Path(filename)
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)

