import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class FakeImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading face images.
    Loads images lazily to prevent memory exhaustion when working with large volumes.
    """
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'label']

        # Load image and convert to RGB (forces 3 channels)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback in case of corrupted file at runtime
            print(f"Error loading image {img_path}: {e}. Returning a black image placeholder.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Return image tensor and float label (required for BCEWithLogitsLoss)
        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(csv_path='dataset_local.csv', batch_size=32, train_size=0.8, random_state=42):
    """
    Splits the dataset and returns train and validation (test) DataLoader instances.
    
    Args:
        csv_path (str | list[str]): Path or paths to local metadata CSV files.
        batch_size (int): Batch size for training and evaluation.
        train_size (float): Proportion of dataset to include in the train split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        train_loader, val_loader (DataLoader, DataLoader): The PyTorch data loaders.
    """
    csv_paths = [csv_path] if isinstance(csv_path, str) else csv_path
    missing_paths = [path for path in csv_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"Local dataset metadata not found: {', '.join(missing_paths)}. Please run the downloader script first.")

    dataframes = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dataframes, ignore_index=True)
    label_counts = df['label'].value_counts()
    if len(label_counts) > 1 and label_counts.min() > 1:
        stratify_labels = df['label']
    else:
        stratify_labels = None
        print("Warning: dataset contains a single class or is too unbalanced for stratification; using a non-stratified split.")
    
    # 80/20 train/test split (stratified to maintain class balance)
    train_df, val_df = train_test_split(
        df, 
        test_size=(1 - train_size), 
        random_state=random_state, 
        stratify=stratify_labels
    )

    print(f"Dataset split summary:")
    print(f" - Training samples: {len(train_df)} (Real: {len(train_df[train_df['label'] == 0])}, Fake: {len(train_df[train_df['label'] == 1])})")
    print(f" - Validation samples: {len(val_df)} (Real: {len(val_df[val_df['label'] == 0])}, Fake: {len(val_df[val_df['label'] == 1])})")

    # ImageNet normalization statistics for pre-trained backbones
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Preprocessing and Augmentations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Instantiate datasets
    train_dataset = FakeImageDataset(train_df, transform=train_transforms)
    val_dataset = FakeImageDataset(val_df, transform=val_transforms)
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders (setting num_workers=0 to avoid multiprocessing overhead/issues in Windows)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
