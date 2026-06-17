import os
import argparse
import pandas as pd
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

def download_image(url, save_path, timeout=10, retries=2):
    """
    Downloads an image from a URL and saves it locally.
    Includes customized User-Agent headers to avoid HTTP 403 Forbidden errors.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                img_data = response.read()
                
            # Verify image integrity using PIL
            image = Image.open(io.BytesIO(img_data))
            image.verify()  # Verify it's a valid image
            
            # Save the valid image
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)
            return True, None
        except Exception as e:
            if attempt == retries:
                return False, str(e)
    return False, "Failed after retries"

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded Fake Image Dataset Downloader")
    parser.add_argument('--limit', type=str, default='200', 
                        help="Total number of images to download. Set to 'all' to download the entire dataset. Defaults to 200.")
    parser.add_argument('--threads', type=int, default=16, 
                        help="Number of concurrent download threads. Defaults to 16.")
    parser.add_argument('--label', choices=['all', 'real', 'fake'], default='all',
                        help="Filter downloads by label. Defaults to 'all'.")
    parser.add_argument('--output_csv', type=str, default='dataset_local.csv',
                        help="Metadata CSV to write. Defaults to 'dataset_local.csv'.")
    args = parser.parse_args()

    # Load dataset CSV
    csv_file = 'FINAL_DATASET.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Missing {csv_file} in the current directory.")

    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    print(f"Dataset contains {total_rows} entries.")

    # Determine limit
    label_filter = args.label.upper()
    if args.limit.lower() == 'all':
        limit = None
        if args.label == 'all':
            print("Downloading all images in the dataset.")
        else:
            print(f"Downloading all {label_filter} images in the dataset.")
    else:
        try:
            limit = int(args.limit)
            if args.label == 'all':
                print(f"Targeting a balanced download of {limit} images ({limit//2} REAL, {limit//2} FAKE).")
            else:
                print(f"Targeting {limit} {label_filter} images.")
        except ValueError:
            print(f"Invalid limit value '{args.limit}'. Defaulting to 200.")
            limit = 200

    # Filter and sample to ensure balance
    real_df = df[df['label'] == 'REAL']
    fake_df = df[df['label'] == 'FAKE']

    if args.label == 'real':
        selected_df = real_df.copy()
        if limit is not None:
            selected_df = selected_df.sample(n=min(limit, len(selected_df)), random_state=42)
    elif args.label == 'fake':
        selected_df = fake_df.copy()
        if limit is not None:
            selected_df = selected_df.sample(n=min(limit, len(selected_df)), random_state=42)
    elif limit is not None:
        half_limit = limit // 2
        # Sample if we have enough, otherwise take what is available
        sampled_real = real_df.sample(n=min(half_limit, len(real_df)), random_state=42)
        sampled_fake = fake_df.sample(n=min(half_limit, len(fake_df)), random_state=42)
        selected_df = pd.concat([sampled_real, sampled_fake]).reset_index(drop=True)
    else:
        selected_df = df.copy()

    # Remap label to binary: REAL = 0, FAKE = 1
    selected_df['label_binary'] = selected_df['label'].map({'REAL': 0, 'FAKE': 1})
    
    # Prepare download parameters
    download_tasks = []
    local_records = []
    
    # Create main data directories
    os.makedirs('data/real', exist_ok=True)
    os.makedirs('data/fake', exist_ok=True)

    print(f"Preparing download queue for {len(selected_df)} images...")
    
    for idx, row in selected_df.iterrows():
        img_id = row['image_id']
        url = row['image_url']
        lbl_str = row['label'].lower() # 'real' or 'fake'
        binary_lbl = row['label_binary']
        
        # Save image with ID as name inside its label folder
        ext = 'jpg'  # standard extension for these URLs
        save_path = os.path.join('data', lbl_str, f"{img_id}.{ext}")
        
        download_tasks.append((url, save_path, img_id, binary_lbl))

    # Download in parallel using ThreadPoolExecutor
    success_count = 0
    failure_count = 0
    
    print(f"Starting download using {args.threads} threads...")
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit tasks
        future_to_task = {
            executor.submit(download_image, task[0], task[1]): task 
            for task in download_tasks
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            url, save_path, img_id, binary_lbl = task
            success, err = future.result()
            
            if success:
                success_count += 1
                # Save to local metadata list
                local_records.append({
                    'image_id': img_id,
                    'image_path': save_path.replace('\\', '/'),  # Uniform forward slashes
                    'label': binary_lbl
                })
            else:
                failure_count += 1
                # We do not append failed downloads to training csv
                if failure_count <= 10:  # Avoid flooding output
                    print(f"Failed to download image {img_id}: {err}")
                elif failure_count == 11:
                    print("Suppressing further failure logs...")

            # Simple progress reporting
            if i % 10 == 0 or i == len(download_tasks):
                print(f"Progress: {i}/{len(download_tasks)} completed. Successes: {success_count}, Failures: {failure_count}")

    # Write local metadata file for PyTorch Dataset
    local_df = pd.DataFrame(local_records)
    local_df.to_csv(args.output_csv, index=False)
    print("\nDownload process completed.")
    print(f"Successfully downloaded: {success_count} images.")
    print(f"Failed to download: {failure_count} images.")
    print(f"Metadata file written to '{args.output_csv}' with {len(local_df)} valid image paths.")

if __name__ == '__main__':
    main()
