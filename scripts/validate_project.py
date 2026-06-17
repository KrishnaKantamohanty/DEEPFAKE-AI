import argparse
import json
import os
from pathlib import Path

import pandas as pd


CSV_FILES = [
    'FINAL_DATASET.csv',
    'dataset_local.csv',
    'dataset_real_local.csv',
    'dataset_fake_local.csv',
    'dataset_all_local.csv',
]
MODEL_FILES = ['best_model.pth', 'best_model_finetuned.pth', 'best_model_full.pth']


def csv_summary(path):
    df = pd.read_csv(path)
    summary = {
        'rows': len(df),
        'columns': list(df.columns),
    }
    if 'label' in df.columns:
        summary['labels'] = {str(k): int(v) for k, v in df['label'].value_counts(dropna=False).to_dict().items()}
    if 'image_path' in df.columns:
        summary['missing_paths'] = int((~df['image_path'].map(os.path.exists)).sum())
    return summary


def directory_count(path):
    folder = Path(path)
    if not folder.exists():
        return 0
    return sum(1 for item in folder.iterdir() if item.is_file())


def build_report():
    report = {
        'csvs': {},
        'models': {},
        'image_directories': {
            'data/real': directory_count('data/real'),
            'data/fake': directory_count('data/fake'),
            'data/external/real': directory_count('data/external/real'),
            'data/external/fake': directory_count('data/external/fake'),
        },
        'web_ui': {
            'templates/index.html': Path('templates/index.html').exists(),
            'static/styles.css': Path('static/styles.css').exists(),
        }
    }

    for csv_file in CSV_FILES:
        if Path(csv_file).exists():
            report['csvs'][csv_file] = csv_summary(csv_file)
        else:
            report['csvs'][csv_file] = {'missing': True}

    for model_file in MODEL_FILES:
        path = Path(model_file)
        report['models'][model_file] = {
            'exists': path.exists(),
            'size_mb': round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description='Validate the local fake-image detector project state.')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON.')
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print('Project validation report')
    print('=' * 32)
    print('\nCSV files')
    for name, summary in report['csvs'].items():
        if summary.get('missing'):
            print(f' - {name}: missing')
            continue
        details = f"{summary['rows']} rows"
        if 'labels' in summary:
            details += f", labels={summary['labels']}"
        if 'missing_paths' in summary:
            details += f", missing_paths={summary['missing_paths']}"
        print(f' - {name}: {details}')

    print('\nImage directories')
    for name, count in report['image_directories'].items():
        print(f' - {name}: {count} files')

    print('\nModels')
    for name, summary in report['models'].items():
        status = 'present' if summary['exists'] else 'missing'
        print(f" - {name}: {status}, {summary['size_mb']} MB")

    print('\nWeb UI')
    for name, exists in report['web_ui'].items():
        print(f" - {name}: {'present' if exists else 'missing'}")


if __name__ == '__main__':
    main()
