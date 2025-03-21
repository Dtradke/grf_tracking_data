import os
import csv
import sys
import urllib.request

# Subset sizes
subsets = {
    "tiny_set": 4,
    "small_set": 50,
    "medium_set": 200,
    "large_set": 500,
    "full_set": 1000,
}

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in subsets:
        print("Usage: python3 scripts/download_game_data.py [tiny_set|small_set|medium_set|large_set|full_set]")
        sys.exit(1)

    subset = sys.argv[1]
    num_files = subsets[subset]

    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "names_ids_links.csv")
    output_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {num_files} files for subset: {subset}")
    count = 0

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if count >= num_files:
                break

            filename = row["Filename"]
            url = row["Download Link"]
            destination = os.path.join(output_dir, filename)

            if os.path.exists(destination):
                print(f"[SKIP] {filename} already exists.")
            else:
                print(f"[DOWNLOADING] {filename}...")
                try:
                    urllib.request.urlretrieve(url, destination)
                except Exception as e:
                    print(f"[ERROR] Failed to download {filename}: {e}")

            count += 1

    print("Download complete.")

if __name__ == "__main__":
    main()
