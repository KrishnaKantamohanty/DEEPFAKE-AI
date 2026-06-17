"""
Download deepfake vs real images for model retraining.

Strategy: Use Hugging Face Datasets Server API to get Parquet file URLs,
then load them with pandas for fast, reliable access (no streaming stalls).

Falls back through multiple datasets until one works.
"""

import os
import sys
import csv
import time
import io
import requests
import pandas as pd
from PIL import Image


def download_dataset(num_per_class: int = 1000):
    output_dir = os.path.join('data', 'modern')
    os.makedirs(output_dir, exist_ok=True)

    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    csv_path = os.path.join('data', 'dataset_modern.csv')

    # Try multiple datasets via the Datasets Server API
    datasets_to_try = [
        'Hemg/deepfake-and-real-images',
        'prithivMLmods/Deepfake-vs-Real-60K',
    ]

    parquet_urls = []
    chosen_dataset = None

    for ds_id in datasets_to_try:
        try:
            print(f"Trying dataset '{ds_id}' ...")
            api_url = f"https://datasets-server.huggingface.co/parquet?dataset={ds_id}"
            resp = requests.get(api_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Extract parquet file URLs
            urls = []
            if 'parquet_files' in data:
                for pf in data['parquet_files']:
                    urls.append(pf['url'])
            elif 'parquet' in data:
                # Alternate response format
                for config_name, splits in data['parquet'].items():
                    if isinstance(splits, list):
                        for item in splits:
                            if isinstance(item, dict) and 'url' in item:
                                urls.append(item['url'])
                            elif isinstance(item, str):
                                urls.append(item)

            if urls:
                parquet_urls = urls
                chosen_dataset = ds_id
                print(f"  [OK] Found {len(urls)} parquet file(s) for '{ds_id}'")
                break
            else:
                print(f"  [SKIP] No parquet URLs found in API response")
        except Exception as exc:
            print(f"  [SKIP] {exc}")
            continue

    if not parquet_urls:
        print("\nParquet API approach failed. Trying direct load_dataset...")
        return download_with_hf_library(num_per_class)

    # Download and process parquet files
    print(f"\nProcessing parquet files from '{chosen_dataset}'...")
    real_count = 0
    fake_count = 0
    rows = []
    start = time.time()

    for purl_idx, purl in enumerate(parquet_urls):
        if real_count >= num_per_class and fake_count >= num_per_class:
            break

        print(f"  Downloading parquet file {purl_idx + 1}/{len(parquet_urls)}...")
        try:
            resp = requests.get(purl, timeout=120)
            resp.raise_for_status()

            # Read parquet into pandas
            df = pd.read_parquet(io.BytesIO(resp.content))
            print(f"    Loaded {len(df)} rows. Columns: {list(df.columns)}")

            # Detect image and label columns
            img_col = None
            label_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'image' in col_lower or 'img' in col_lower:
                    img_col = col
                elif 'label' in col_lower or 'class' in col_lower:
                    label_col = col

            if img_col is None or label_col is None:
                print(f"    [SKIP] Could not identify image/label columns")
                continue

            for row_idx, row in df.iterrows():
                if real_count >= num_per_class and fake_count >= num_per_class:
                    break

                try:
                    # Handle image data (could be bytes or dict with 'bytes' key)
                    img_data = row[img_col]
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        continue

                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                    # Determine label
                    raw_label = row[label_col]
                    if isinstance(raw_label, (int, float)):
                        is_fake = int(raw_label) == 1
                    else:
                        label_str = str(raw_label).lower()
                        is_fake = any(kw in label_str for kw in ['fake', 'ai', 'generated', 'synthetic', 'deepfake'])

                    if is_fake:
                        if fake_count >= num_per_class:
                            continue
                        target_dir = fake_dir
                        target_label = 1
                        fake_count += 1
                        prefix = 'fake_'
                    else:
                        if real_count >= num_per_class:
                            continue
                        target_dir = real_dir
                        target_label = 0
                        real_count += 1
                        prefix = 'real_'

                    img_path = os.path.join(target_dir, f"{prefix}{purl_idx}_{row_idx}.jpg")
                    image.save(img_path, 'JPEG', quality=95)

                    rows.append({
                        'image_id': f"{prefix}{purl_idx}_{row_idx}",
                        'image_path': img_path,
                        'label': target_label,
                        'source': chosen_dataset,
                    })

                    total = real_count + fake_count
                    if total % 100 == 0:
                        elapsed = time.time() - start
                        print(f"    [{elapsed:.0f}s] Saved {total} images (Real: {real_count}, Fake: {fake_count})")

                except Exception:
                    continue

        except Exception as exc:
            print(f"    [ERROR] Failed to download parquet: {exc}")
            continue

    return finalize_csv(csv_path, rows, real_count, fake_count)


def download_with_hf_library(num_per_class: int = 1000):
    """Fallback: use datasets library with streaming."""
    from datasets import load_dataset

    output_dir = os.path.join('data', 'modern')
    real_dir = os.path.join(output_dir, 'real')
    fake_dir = os.path.join(output_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    csv_path = os.path.join('data', 'dataset_modern.csv')

    datasets_to_try = [
        'Hemg/deepfake-and-real-images',
        'prithivMLmods/Deepfake-vs-Real-60K',
    ]

    for ds_id in datasets_to_try:
        try:
            print(f"Trying HF library load for '{ds_id}'...")
            dataset = load_dataset(ds_id, split='train', streaming=True)
            first = next(iter(dataset))
            print(f"  [OK] Connected. Keys: {list(first.keys())}")

            real_count = 0
            fake_count = 0
            rows = []
            start = time.time()

            # Determine column names
            img_key = 'image' if 'image' in first else list(first.keys())[0]
            label_key = 'label' if 'label' in first else list(first.keys())[-1]

            for idx, item in enumerate(dataset):
                if real_count >= num_per_class and fake_count >= num_per_class:
                    break

                try:
                    image = item[img_key]
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    raw_label = item[label_key]
                    if isinstance(raw_label, int):
                        is_fake = raw_label == 1
                    else:
                        label_str = str(raw_label).lower()
                        is_fake = any(kw in label_str for kw in ['fake', 'ai', 'generated', 'synthetic'])

                    if is_fake:
                        if fake_count >= num_per_class:
                            continue
                        target_dir = fake_dir
                        target_label = 1
                        fake_count += 1
                        prefix = 'fake_'
                    else:
                        if real_count >= num_per_class:
                            continue
                        target_dir = real_dir
                        target_label = 0
                        real_count += 1
                        prefix = 'real_'

                    img_path = os.path.join(target_dir, f"{prefix}{idx}.jpg")
                    image.save(img_path, 'JPEG', quality=95)

                    rows.append({
                        'image_id': f"{prefix}{idx}",
                        'image_path': img_path,
                        'label': target_label,
                        'source': ds_id,
                    })

                    total = real_count + fake_count
                    if total % 100 == 0:
                        elapsed = time.time() - start
                        print(f"    [{elapsed:.0f}s] Saved {total} images (Real: {real_count}, Fake: {fake_count})")

                except Exception:
                    continue

            return finalize_csv(csv_path, rows, real_count, fake_count)

        except Exception as exc:
            print(f"  [SKIP] {exc}")
            continue

    print("ERROR: All dataset sources failed.")
    return False


def finalize_csv(csv_path, rows, real_count, fake_count):
    """Write the CSV and report results."""
    print(f"\nFinished. Real: {real_count}, Fake: {fake_count}, Total: {real_count + fake_count}")

    if real_count == 0 or fake_count == 0:
        print("ERROR: Could not get images for both classes.")
        return False

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'image_path', 'label', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {csv_path}")
    print(f"Run training: python train.py --csv_path {csv_path} --epochs 5")
    return True


if __name__ == '__main__':
    num = 1000
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except ValueError:
            pass

    success = download_dataset(num_per_class=num)
    if not success:
        print("\nDataset preparation FAILED.")
        sys.exit(1)
    else:
        print("\nDataset preparation COMPLETE.")
        sys.exit(0)
