import argparse
import csv
import io
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import Image


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
FIELDNAMES = ['image_id', 'image_path', 'label']
FAILED_FIELDNAMES = ['image_id', 'image_url', 'label', 'error']


def verify_existing(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def dicebear_fallback_url(image_id):
    seed = quote(f'fallback-{image_id}')
    return f'https://api.dicebear.com/7.x/personas/png?seed={seed}&size=300'


def candidate_urls(task, use_fallback=True):
    urls = [task['url']]
    if use_fallback and task['label'] == 1 and 'api.multiavatar.com' in task['url']:
        urls.append(dicebear_fallback_url(task['image_id']))
    return urls


def download_image(urls, save_path, timeout=20, retries=2):
    errors = []

    for url in urls:
        request = urllib.request.Request(url, headers=HEADERS)
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()

                image = Image.open(io.BytesIO(data))
                image.verify()

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(data)
                return True, None
            except Exception as exc:
                if attempt == retries:
                    errors.append(f'{url}: {exc}')
                else:
                    time.sleep(0.4 * (attempt + 1))

    return False, ' | '.join(errors)


def load_done(output_csv):
    if not os.path.exists(output_csv):
        return set()
    done = set()
    for row in pd.read_csv(output_csv).to_dict('records'):
        done.add(str(row['image_id']))
    return done


def append_row(output_csv, row):
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_failed_rows(failed_csv, rows):
    if not rows:
        return
    with open(failed_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_tasks(df, output_csv):
    done = load_done(output_csv)
    tasks = []

    for row in df.to_dict('records'):
        image_id = str(row['image_id'])
        if image_id in done:
            continue

        label_text = row['label'].lower()
        label = 0 if row['label'] == 'REAL' else 1
        save_path = Path('data') / label_text / f'{image_id}.jpg'
        image_path = str(save_path).replace('\\', '/')

        if save_path.exists() and verify_existing(save_path):
            append_row(output_csv, {
                'image_id': image_id,
                'image_path': image_path,
                'label': label
            })
            done.add(image_id)
            continue

        tasks.append({
            'image_id': image_id,
            'url': row['image_url'],
            'image_path': image_path,
            'save_path': str(save_path),
            'label': label
        })

    return tasks


def main():
    parser = argparse.ArgumentParser(description='Resumable dataset downloader for FINAL_DATASET.csv.')
    parser.add_argument('--csv', default='FINAL_DATASET.csv', help='Input URL dataset CSV.')
    parser.add_argument('--output_csv', default='dataset_all_local.csv', help='Output local metadata CSV.')
    parser.add_argument('--failed_csv', default='dataset_download_failures.csv', help='CSV where failed downloads are recorded.')
    parser.add_argument('--threads', type=int, default=8, help='Number of parallel downloads.')
    parser.add_argument('--limit', type=int, default=None, help='Optional max new downloads for this run.')
    parser.add_argument('--no_fallback', action='store_true', help='Disable DiceBear fallback for dead Multiavatar fake-image URLs.')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    tasks = make_tasks(df, args.output_csv)
    if args.limit is not None:
        tasks = tasks[:args.limit]

    print(f'Rows in source CSV: {len(df)}')
    print(f'New downloads queued: {len(tasks)}')
    print(f'Output CSV: {args.output_csv}')

    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                download_image,
                candidate_urls(task, use_fallback=not args.no_fallback),
                task['save_path']
            ): task
            for task in tasks
        }
        failed_rows = []

        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            success, error = future.result()

            if success:
                append_row(args.output_csv, {
                    'image_id': task['image_id'],
                    'image_path': task['image_path'],
                    'label': task['label']
                })
                success_count += 1
            else:
                failure_count += 1
                failed_rows.append({
                    'image_id': task['image_id'],
                    'image_url': task['url'],
                    'label': task['label'],
                    'error': error
                })
                if failure_count <= 10:
                    print(f"Failed {task['image_id']}: {error}")
                elif failure_count == 11:
                    print('Suppressing further failure logs...')

            if index % 25 == 0 or index == len(tasks):
                print(f'Progress: {index}/{len(tasks)} | success: {success_count} | failed: {failure_count}')

    write_failed_rows(args.failed_csv, failed_rows)
    print('Done.')
    print(f'Successfully downloaded this run: {success_count}')
    print(f'Failed this run: {failure_count}')
    if failure_count:
        print(f'Failure details written to: {args.failed_csv}')


if __name__ == '__main__':
    main()
