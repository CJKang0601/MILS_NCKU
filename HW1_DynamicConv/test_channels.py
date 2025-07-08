"""
MILS Assignment I - Test Different Channel Combinations
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import logging

from utils import set_seed, load_data_from_files, MiniImageNetDataset
from dynamic_convolution import DynamicCNN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("channel_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

def test_with_channel_combinations(model, test_data, base_dir, device, transform, batch_size=64):
    """Test model with different channel combinations"""
    channel_modes = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
    results = {}
    
    model.eval()
    
    for mode in channel_modes:
        logger.info(f"Testing with channel mode: {mode}")
        
        # Create test dataset with specified channel mode
        test_dataset = MiniImageNetDataset(
            test_data,
            base_dir,
            transform=transform,
            channel_mode=mode
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Testing {mode}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                # Statistics
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        results[mode] = accuracy
        logger.info(f"Accuracy with {mode} channels: {accuracy:.2f}%")
    
    return results

def plot_channel_comparison(results, save_path='channel_comparison.png'):
    """Plot the accuracy comparison for different channel combinations"""
    modes = list(results.keys())
    accuracies = [results[mode] for mode in modes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, accuracies)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.xlabel('Channel Mode')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison for Different Channel Combinations')
    plt.ylim(0, max(accuracies) + 10)
    plt.savefig(save_path)
    plt.close()

def load_model(model_path, num_classes, device):
    """Load dynamic CNN model from checkpoint"""
    model = DynamicCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='Test model with different channel combinations')
    parser.add_argument('--model_path', type=str, default=os.path.join(CHECKPOINTS_DIR, 'dynamic_cnn_best.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load data
    _, _, test_data = load_data_from_files()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Number of classes
    num_classes = len(set(label for _, label in test_data))
    logger.info(f"Number of classes: {num_classes}")
    
    # Load model
    model = load_model(args.model_path, num_classes, device)
    logger.info(f"Loaded model from {args.model_path}")
    
    # Test with different channel combinations
    results = test_with_channel_combinations(
        model, test_data, DATA_DIR, device, transform, args.batch_size
    )
    
    # Plot results
    plot_channel_comparison(
        results, save_path=os.path.join(PLOT_DIR, 'channel_comparison_detailed.png')
    )
    
    # Save results to file
    with open(os.path.join(RESULTS_DIR, 'channel_results.txt'), 'w') as f:
        f.write("Channel Combination Results\n")
        f.write("=========================\n\n")
        for mode, acc in results.items():
            f.write(f"{mode}: {acc:.2f}%\n")
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()