from pathlib import Path
from datetime import datetime

folder = Path("C:/Users/RD/Downloads/images_aadhaar/images")

for f in folder.glob("*"):
    if f.is_file():
        mod_date = datetime.fromtimestamp(f.stat().st_mtime).date()
        print(f"{f.name} → Modified: {mod_date}")
