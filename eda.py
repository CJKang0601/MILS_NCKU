import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import zipfile
import random
from collections import Counter

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Global configs
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ZIP_PATH = "images.zip"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Extract the zip file if not already extracted
def extract_zip_if_needed():
    if not os.path.exists(IMAGE_DIR):
        print("Extracting images.zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete!")
    else:
        print("Images directory already exists. Skipping extraction.")

# Load and parse data files
def load_data():
    train_file = os.path.join(DATA_DIR, "train.txt")
    val_file = os.path.join(DATA_DIR, "validation.txt")
    test_file = os.path.join(DATA_DIR, "test.txt")
    
    def parse_file(file_path):
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist!")
            return []
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                data.append((img_path, int(label)))
        return data
    
    train_data = parse_file(train_file)
    val_data = parse_file(val_file)
    test_data = parse_file(test_file)
    
    return train_data, val_data, test_data

# Create a custom Dataset class
class MiniImageNetDataset(Dataset):
    def __init__(self, data, base_dir, transform=None):
        self.data = data
        self.base_dir = base_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_full_path = os.path.join(self.base_dir, img_path)
        
        try:
            image = Image.open(img_full_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_full_path}: {e}")
            # Return a placeholder in case of error
            placeholder = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), color='black')
            return placeholder, label

# EDA Functions
def analyze_dataset_statistics(train_data, val_data, test_data):
    print("Dataset Statistics:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Count unique labels
    train_labels = [label for _, label in train_data]
    val_labels = [label for _, label in val_data]
    test_labels = [label for _, label in test_data]
    
    unique_train_labels = set(train_labels)
    unique_val_labels = set(val_labels)
    unique_test_labels = set(test_labels)
    
    print(f"Unique classes in training set: {len(unique_train_labels)}")
    print(f"Unique classes in validation set: {len(unique_val_labels)}")
    print(f"Unique classes in test set: {len(unique_test_labels)}")
    
    # Class distribution
    train_label_counts = Counter(train_labels)
    
    print("\nClass Distribution (top 5 classes):")
    for label, count in train_label_counts.most_common(5):
        print(f"Class {label}: {count} samples")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x=train_labels, order=sorted(unique_train_labels))
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

def analyze_image_properties(data, base_dir, sample_size=100):
    print("\nAnalyzing image properties...")
    
    # Sample some images for analysis
    if len(data) > sample_size:
        sample_data = random.sample(data, sample_size)
    else:
        sample_data = data
    
    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    
    for img_path, _ in sample_data:
        img_full_path = os.path.join(base_dir, img_path)
        try:
            img = Image.open(img_full_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w/h)
            channels.append(len(img.getbands()))
        except Exception as e:
            print(f"Error analyzing image {img_full_path}: {e}")
    
    print(f"Image dimensions summary (from {len(widths)} samples):")
    print(f"Width - Mean: {np.mean(widths):.2f}, Min: {min(widths)}, Max: {max(widths)}")
    print(f"Height - Mean: {np.mean(heights):.2f}, Min: {min(heights)}, Max: {max(heights)}")
    print(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}")
    print(f"Channels - Most common: {Counter(channels).most_common(1)[0][0]}")
    
    # Plot distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.hist(widths, bins=20)
    ax1.set_title('Width Distribution')
    ax1.set_xlabel('Width')
    
    ax2.hist(heights, bins=20)
    ax2.set_title('Height Distribution')
    ax2.set_xlabel('Height')
    
    ax3.hist(aspect_ratios, bins=20)
    ax3.set_title('Aspect Ratio Distribution')
    ax3.set_xlabel('Aspect Ratio')
    
    plt.tight_layout()
    plt.savefig('image_property_distribution.png')
    plt.close()

def visualize_sample_images(data, base_dir, num_samples=5):
    print("\nVisualizing sample images...")
    
    # Sample one image per class (up to num_samples)
    labels = [label for _, label in data]
    unique_labels = sorted(set(labels))[:num_samples]
    
    plt.figure(figsize=(15, 3))
    
    for i, label in enumerate(unique_labels, 1):
        # Find first image with this label
        for img_path, img_label in data:
            if img_label == label:
                img_full_path = os.path.join(base_dir, img_path)
                try:
                    img = Image.open(img_full_path).convert('RGB')
                    plt.subplot(1, len(unique_labels), i)
                    plt.imshow(img)
                    plt.title(f"Class {label}")
                    plt.axis('off')
                    break
                except Exception as e:
                    print(f"Error visualizing image {img_full_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()

def examine_channel_distributions(data, base_dir, num_samples=5):
    print("\nExamining channel distributions...")
    
    # Sample random images
    if len(data) > num_samples:
        sample_data = random.sample(data, num_samples)
    else:
        sample_data = data
    
    plt.figure(figsize=(15, num_samples * 4))
    
    for i, (img_path, label) in enumerate(sample_data, 1):
        img_full_path = os.path.join(base_dir, img_path)
        try:
            img = Image.open(img_full_path).convert('RGB')
            img_array = np.array(img)
            
            plt.subplot(num_samples, 4, (i-1)*4 + 1)
            plt.imshow(img)
            plt.title(f"Image {i} (Class {label})")
            plt.axis('off')
            
            plt.subplot(num_samples, 4, (i-1)*4 + 2)
            plt.hist(img_array[:,:,0].flatten(), bins=50, color='r', alpha=0.7)
            plt.title(f"Red Channel Histogram")
            
            plt.subplot(num_samples, 4, (i-1)*4 + 3)
            plt.hist(img_array[:,:,1].flatten(), bins=50, color='g', alpha=0.7)
            plt.title(f"Green Channel Histogram")
            
            plt.subplot(num_samples, 4, (i-1)*4 + 4)
            plt.hist(img_array[:,:,2].flatten(), bins=50, color='b', alpha=0.7)
            plt.title(f"Blue Channel Histogram")
            
        except Exception as e:
            print(f"Error examining channels for {img_full_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('channel_distributions.png')
    plt.close()

def analyze_for_missing_data(train_data, val_data, test_data, base_dir):
    print("\nChecking for missing data...")
    
    missing_count = 0
    invalid_count = 0
    
    # Function to check a single dataset
    def check_dataset(data_list, name):
        nonlocal missing_count, invalid_count
        local_missing = 0
        local_invalid = 0
        
        for img_path, _ in data_list:
            img_full_path = os.path.join(base_dir, img_path)
            
            if not os.path.exists(img_full_path):
                local_missing += 1
                missing_count += 1
            else:
                try:
                    # Try to open the image to check if it's valid
                    with Image.open(img_full_path) as img:
                        img.verify()  # Verify it's actually an image
                except Exception:
                    local_invalid += 1
                    invalid_count += 1
        
        print(f"{name} set: {local_missing} missing files, {local_invalid} invalid images")
    
    check_dataset(train_data, "Training")
    check_dataset(val_data, "Validation")
    check_dataset(test_data, "Test")
    
    print(f"Total: {missing_count} missing files, {invalid_count} invalid images")

def main():
    # Extract data if needed
    extract_zip_if_needed()
    
    # Load data
    train_data, val_data, test_data = load_data()
    
    # Perform EDA
    analyze_dataset_statistics(train_data, val_data, test_data)
    analyze_image_properties(train_data, DATA_DIR)
    visualize_sample_images(train_data, DATA_DIR)
    examine_channel_distributions(train_data, DATA_DIR)
    analyze_for_missing_data(train_data, val_data, test_data, DATA_DIR)
    
    print("\nEDA completed successfully!")

if __name__ == "__main__":
    main()