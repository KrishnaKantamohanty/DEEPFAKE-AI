import argparse
import csv
from pathlib import Path


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(folder, label):
    rows = []
    folder = Path(folder)
    if not folder.exists():
        return rows

    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            rows.append({
                'image_id': path.stem,
                'image_path': str(path),
                'label': label
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Build a CSV from external real/fake image folders.')
    parser.add_argument('--real_dir', default='data/external/real', help='Folder containing real images.')
    parser.add_argument('--fake_dir', default='data/external/fake', help='Folder containing fake/generated images.')
    parser.add_argument('--output', default='dataset_external.csv', help='Output CSV path.')
    args = parser.parse_args()

    rows = []
    rows.extend(collect_images(args.real_dir, 0))
    rows.extend(collect_images(args.fake_dir, 1))

    if not rows:
        raise SystemExit('No valid images found. Add images under data/external/real and data/external/fake first.')

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'image_path', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    real_count = sum(row['label'] == 0 for row in rows)
    fake_count = sum(row['label'] == 1 for row in rows)
    print(f'Wrote {len(rows)} rows to {args.output}')
    print(f'Real: {real_count} | Fake: {fake_count}')


if __name__ == '__main__':
    main()
