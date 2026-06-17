import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path


DATASET = 'xhlulu/140k-real-and-fake-faces'
PAGE_TOKEN_RE = re.compile(r'^Next Page Token = (.+)$')
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def run_kaggle(args):
    command = ['kaggle', *args]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


def parse_files_output(output):
    token = None
    paths = []

    for raw_line in output.splitlines():
        line = raw_line.strip().replace('\r', '')
        if not line:
            continue

        token_match = PAGE_TOKEN_RE.match(line)
        if token_match:
            token = token_match.group(1).strip()
            continue

        if line.startswith('name,') or line.startswith('---') or line.startswith('Dataset URL:'):
            continue

        if ',' not in line:
            continue

        path = line.split(',', 1)[0].strip()
        if Path(path).suffix.lower() in VALID_EXTENSIONS:
            paths.append(path)

    return paths, token


def collect_file_paths(per_class):
    selected = {'real': [], 'fake': []}
    token = None

    while len(selected['real']) < per_class or len(selected['fake']) < per_class:
        args = ['datasets', 'files', DATASET, '--page-size', '200', '--csv']
        if token:
            args.extend(['--page-token', token])

        output = run_kaggle(args)
        paths, token = parse_files_output(output)

        for path in paths:
            normalized = path.replace('\\', '/')
            if '/test/real/' in normalized and len(selected['real']) < per_class:
                selected['real'].append(normalized)
            elif '/test/fake/' in normalized and len(selected['fake']) < per_class:
                selected['fake'].append(normalized)

        print(
            f"Selected real={len(selected['real'])}/{per_class}, "
            f"fake={len(selected['fake'])}/{per_class}"
        )

        if not token:
            break

    if len(selected['real']) < per_class or len(selected['fake']) < per_class:
        raise RuntimeError(
            f"Could only find real={len(selected['real'])}, fake={len(selected['fake'])}; "
            f"requested {per_class} per class."
        )

    return selected


def download_one(remote_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / Path(remote_path).name
    if local_path.exists():
        return local_path

    run_kaggle([
        'datasets',
        'download',
        DATASET,
        '-f',
        remote_path,
        '-p',
        str(output_dir),
        '--force',
        '--quiet',
    ])
    return local_path


def write_csv(rows, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'image_path', 'label', 'source'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Download an external Kaggle real/fake face test subset.')
    parser.add_argument('--per_class', type=int, default=300, help='Images to download per class.')
    parser.add_argument('--output_root', default='data/external', help='Destination root containing real/ and fake/.')
    parser.add_argument('--output_csv', default='dataset_external.csv', help='External test metadata CSV.')
    parser.add_argument('--clean', action='store_true', help='Remove existing files under output_root before downloading.')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    real_dir = output_root / 'real'
    fake_dir = output_root / 'fake'

    selected = collect_file_paths(args.per_class)
    rows = []

    for label_name, label_value, output_dir in [('real', 0, real_dir), ('fake', 1, fake_dir)]:
        for index, remote_path in enumerate(selected[label_name], 1):
            local_path = download_one(remote_path, output_dir)
            rows.append({
                'image_id': f'kaggle_{label_name}_{Path(remote_path).stem}',
                'image_path': str(local_path).replace('\\', '/'),
                'label': label_value,
                'source': DATASET,
            })

            if index % 25 == 0 or index == len(selected[label_name]):
                print(f'Downloaded {label_name}: {index}/{len(selected[label_name])}')

    write_csv(rows, args.output_csv)
    print(f'Wrote {len(rows)} rows to {args.output_csv}')


if __name__ == '__main__':
    main()
