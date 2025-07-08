"""
MILS Assignment I - Utility Functions
"""

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from torchvision import transforms # 確保導入 transforms
import torchvision.transforms.functional as TF # 導入 functional 以便單獨調用 normalize

# Setup logging
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_data_from_files():
    """Load data from train.txt, validation.txt, and test.txt files"""
    train_file = os.path.join(DATA_DIR, "train.txt")
    val_file = os.path.join(DATA_DIR, "validation.txt")
    test_file = os.path.join(DATA_DIR, "test.txt")
    
    def parse_file(file_path):
        if not os.path.exists(file_path):
            logger.warning(f"Warning: {file_path} does not exist!")
            return []
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts
                    data.append((img_path, int(label)))
                else:
                    logger.warning(f"Skipping malformed line: {line.strip()}")
        return data
    
    train_data = parse_file(train_file)
    val_data = parse_file(val_file)
    test_data = parse_file(test_file)
    
    logger.info(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples")
    return train_data, val_data, test_data

class MiniImageNetDataset(Dataset):
    """
    用於 mini-ImageNet 的自定義數據集。
    支持在訓練時隨機選擇通道模式，並正確處理歸一化。
    """
    def __init__(self, data_list, base_dir, transform_base=None, channel_mode='rgb', is_training=False):
        """
        Args:
            data_list: List of (image_path, label) tuples.
            base_dir: 數據集基礎目錄 (e.g., 'data').
            transform_base: 基礎圖像變換 (不包含 Normalize)。
            channel_mode: 默認通道模式 (用於非訓練模式).
            is_training (bool): 如果是 True，則在 __getitem__ 中隨機選擇通道模式。
        """
        self.data_list = data_list
        self.base_dir = base_dir
        self.transform_base = transform_base
        self.channel_mode = channel_mode # 用於非訓練模式
        self.is_training = is_training

        self.all_modes = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
        if not is_training and channel_mode not in self.all_modes:
            raise ValueError(f"Invalid channel_mode '{channel_mode}' for non-training dataset.")

        # 標準 ImageNet 均值和標準差 (RGB)
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        logger.info(f"MiniImageNetDataset initialized: Training Mode={is_training}, Default Channel Mode={channel_mode}, Base Dir={base_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx >= len(self.data_list):
             logger.error(f"Index {idx} out of bounds for dataset length {len(self.data_list)}")
             return None, None

        img_rel_path, label = self.data_list[idx]

        # 確定當前樣本使用的通道模式
        if self.is_training:
            current_mode = random.choice(self.all_modes)
        else:
            current_mode = self.channel_mode

        # --- 正確組合路徑 (假設 .txt 文件包含 'images/' 前綴) ---
        full_path = os.path.join(self.base_dir, img_rel_path) # base_dir 應為 'data'
        # ---

        try:
            # 1. 加載原始 RGB 圖像
            img = Image.open(full_path).convert('RGB')

            # 2. 應用基礎變換 (Resize, Crop, Flip, ToTensor)
            if self.transform_base:
                img_tensor = self.transform_base(img) # 輸出應為 (3, H, W) 的 Tensor
            else:
                # 如果沒有提供基礎變換，至少要轉成 Tensor
                img_tensor = transforms.ToTensor()(img)

            # 3. 根據 current_mode 提取通道
            c, h, w = img_tensor.shape
            if c != 3: # 確保基礎變換後是 3 通道
                 logger.warning(f"Image tensor shape is {img_tensor.shape} after base transform, expected 3 channels. Skipping sample {full_path}.")
                 return None, None

            if current_mode == 'rgb':
                extracted_tensor = img_tensor
                current_mean = self.mean_rgb
                current_std = self.std_rgb
            elif current_mode == 'r':
                extracted_tensor = img_tensor[0:1, :, :]
                current_mean = [self.mean_rgb[0]]
                current_std = [self.std_rgb[0]]
            elif current_mode == 'g':
                extracted_tensor = img_tensor[1:2, :, :]
                current_mean = [self.mean_rgb[1]]
                current_std = [self.std_rgb[1]]
            elif current_mode == 'b':
                extracted_tensor = img_tensor[2:3, :, :]
                current_mean = [self.mean_rgb[2]]
                current_std = [self.std_rgb[2]]
            elif current_mode == 'rg':
                extracted_tensor = img_tensor[0:2, :, :]
                current_mean = [self.mean_rgb[0], self.mean_rgb[1]]
                current_std = [self.std_rgb[0], self.std_rgb[1]]
            elif current_mode == 'rb':
                extracted_tensor = torch.cat([img_tensor[0:1, :, :], img_tensor[2:3, :, :]], dim=0)
                current_mean = [self.mean_rgb[0], self.mean_rgb[2]]
                current_std = [self.std_rgb[0], self.std_rgb[2]]
            elif current_mode == 'gb':
                extracted_tensor = img_tensor[1:3, :, :]
                current_mean = [self.mean_rgb[1], self.mean_rgb[2]]
                current_std = [self.std_rgb[1], self.std_rgb[2]]
            else:
                 # This should not happen due to random.choice or init check
                 logger.error(f"Unexpected current_mode '{current_mode}'. Returning None.")
                 return None, None

            # 4. 應用對應通道的歸一化
            # 使用 transforms.functional.normalize
            final_tensor = TF.normalize(extracted_tensor, mean=current_mean, std=current_std)

            return final_tensor, label

        except FileNotFoundError:
            logger.debug(f"[FileNotFoundError] Error loading image {full_path}")
            return None, None
        except Exception as e:
            logger.exception(f"Error processing image {full_path} with mode {current_mode}: {e}") # 使用 exception 記錄 traceback
            return None, None
        
# Import here to avoid circular imports in other files
from torchvision import transforms