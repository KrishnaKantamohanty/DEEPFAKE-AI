import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import FakeImageDataset
from model import get_model
from predict import preprocess_image


def build_loader(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    missing = df[~df['image_path'].map(lambda path: Path(path).exists())]
    if not missing.empty:
        raise FileNotFoundError(f'{len(missing)} image paths from {csv_path} do not exist.')

    dataset = FakeImageDataset(df, transform=lambda image: preprocess_image_from_pil(image))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), df


def preprocess_image_from_pil(image):
    # Reuse the exact predict.py preprocessing by saving no state and matching transforms.
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return preprocess(image)


def save_confusion_matrix(labels, preds, output_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE'],
    )
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('External Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate(model_path, csv_path, batch_size, output_json, output_matrix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader, df = build_loader(csv_path, batch_size)

    model = get_model(fine_tune=False, device=device, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    labels = []
    preds = []
    probs = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            batch_probs = torch.sigmoid(outputs)
            batch_preds = (batch_probs >= 0.5).long().cpu().numpy()

            probs.extend(batch_probs.cpu().numpy().tolist())
            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.long().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    metrics = {
        'csv_path': csv_path,
        'model_path': model_path,
        'samples': len(df),
        'label_counts': {str(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
        'accuracy': accuracy_score(labels, preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=['REAL (0)', 'FAKE (1)'],
            zero_division=0,
            output_dict=True,
        ),
    }

    save_confusion_matrix(labels, preds, output_matrix)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('External evaluation complete')
    print(f"Samples:   {metrics['samples']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f'Wrote metrics: {output_json}')
    print(f'Wrote matrix:  {output_matrix}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on an independent external CSV.')
    parser.add_argument('--csv_path', default='dataset_external.csv')
    parser.add_argument('--model_path', default='best_model_full.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_json', default='external_evaluation_summary.json')
    parser.add_argument('--output_matrix', default='external_confusion_matrix.png')
    args = parser.parse_args()

    evaluate(args.model_path, args.csv_path, args.batch_size, args.output_json, args.output_matrix)


if __name__ == '__main__':
    main()
