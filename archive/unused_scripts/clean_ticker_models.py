from __future__ import annotations
import argparse, os
from settings import MODEL_DIR

def main():
    parser = argparse.ArgumentParser(description="Delete all saved model artifacts for one ticker")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{ticker}_")]
    print("Found:", files)
    if args.delete:
        for f in files:
            os.remove(os.path.join(MODEL_DIR, f))
        print("Deleted.")
if __name__ == "__main__":
    main()
