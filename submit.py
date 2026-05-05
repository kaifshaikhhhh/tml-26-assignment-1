import argparse
import os
import sys
from pathlib import Path

import requests


BASE_URL = "http://34.63.153.158"
TASK_ID = "01-mia"


def die(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Submit a membership inference CSV to the leaderboard.")
    parser.add_argument("--csv", type=Path, default=Path("submission.csv"), help="Path to submission.csv")
    parser.add_argument("--api-key", default=os.environ.get("MIA_API_KEY"), help="API key, or set MIA_API_KEY")
    args = parser.parse_args()

    submit_path = args.csv.resolve()
    if not submit_path.exists():
        die(f"File not found: {submit_path}")
    if not args.api_key:
        die("Missing API key. Set MIA_API_KEY or pass --api-key.")

    with submit_path.open("rb") as f:
        response = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": args.api_key},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )

    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}

    print("Status:", response.status_code)
    print("Server response:", body)
    response.raise_for_status()


if __name__ == "__main__":
    main()
