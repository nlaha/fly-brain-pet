"""
Fetch male-cns connectome files.

The sandbox this was authored in can't reach janelia.org, so this
just wraps plain HTTP downloads for you to run locally. Check
https://male-cns.janelia.org/download/ for current filenames/URLs —
these are the v1.0 names as of Sep 2026.
"""
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",   # 13 MB
    "body-neurotransmitters-male-cns-v1.0.feather",         # 42 MB
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather", # 1.1 GB
]

OUT_DIR = Path(__file__).parent / "raw"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for fname in FILES:
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"skip {fname}, already downloaded")
            continue
        url = f"{BASE}/{fname}"
        print(f"downloading {url}")
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"  failed: {e} — grab this one manually from the download page")


if __name__ == "__main__":
    main()
