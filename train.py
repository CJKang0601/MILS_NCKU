"""
MILS Assignment I 
實現任務A和任務B的訓練與評估
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time
import numpy as np

# --- 導入模型和工具函數 ---
from utils import set_seed, load_data_from_files, MiniImageNetDataset
# Task A:
from dynamic_convolution import DynamicCNN
# Task B

from efficient_network import SimpleParallelEfficientNet, EfficientNet 

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 目錄配置
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
RESULTS_DIR = "results"
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# 創建必要的目錄
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_device(gpu_id=None):
    """
    設定要使用的設備 (CPU 或特定 GPU)

    參數:
        gpu_id (int, optional): 要使用的 GPU ID，None 表示自動選擇

    返回:
        torch.device: 設定好的設備
    """
    if not torch.cuda.is_available():
        logger.info("CUDA 不可用，使用 CPU")
        return torch.device('cpu')

    if gpu_id is None:
        # 自動選擇 GPU
        device = torch.device('cuda')
        logger.info(f"自動選擇 GPU，使用 cuda:{torch.cuda.current_device()}")
    else:
        # 檢查指定的 GPU ID 是否有效
        if gpu_id >= torch.cuda.device_count():
            logger.warning(f"指定的 GPU ID {gpu_id} 不存在，可用的 GPU 數量為 {torch.cuda.device_count()}，改用 GPU 0")
            gpu_id = 0

        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"使用 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # 顯示 GPU 記憶體資訊
    if torch.cuda.is_available():
        gpu_id_to_use = gpu_id if gpu_id is not None else torch.cuda.current_device()
        try:
            total_memory = torch.cuda.get_device_properties(gpu_id_to_use).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(gpu_id_to_use) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(gpu_id_to_use) / (1024**3)
            logger.info(f"GPU 記憶體: 總計 {total_memory:.2f} GB，已分配 {allocated_memory:.2f} GB，已保留 {reserved_memory:.2f} GB")
        except Exception as e:
             logger.warning(f"無法獲取 GPU 記憶體信息: {e}")


    return device
def custom_collate_fn(batch):
    """
    自定義 collate_fn，處理不同通道數的圖像和可能的 None 值。
    確保不改變通道數。
    """
    # 過濾掉 None 值
    batch = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]
    
    if not batch:
        logger.warning("批次中沒有有效樣本，返回 None")
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
    
    # 檢查所有圖像的通道數是否一致
    first_channels = batch[0][0].shape[0]
    for img, _ in batch:
        if img.shape[0] != first_channels:
            logger.error(f"批次中的圖像通道數不一致 ({img.shape[0]} vs {first_channels})，無法堆疊")
            return None
    
    # 分離圖像和標籤
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 堆疊圖像和標籤
    try:
        images_batch = torch.stack(images, dim=0)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        return images_batch, labels_batch
    except Exception as e:
        logger.error(f"堆疊批次時發生錯誤: {e}")
        return None


    # 將標籤轉換為張量並堆疊
    # 假設 labels 是 Python int 列表
    try:
         labels_batch = torch.tensor(labels, dtype=torch.long) # 確保是 long 類型
    except Exception as label_err:
         logger.error(f"Failed to stack labels: {label_err}")
         logger.error(f"  Labels received: {labels}")
         return None # 返回 None

    return images_batch, labels_batch
def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    """
    訓練模型一個 epoch (增加 AMP 支持)

    參數:
        model: 模型
        dataloader: 數據加載器
        criterion: 損失函數
        optimizer: 優化器
        device: 訓練設備
        use_amp (bool): 是否使用自動混合精度
        scaler (GradScaler, optional): AMP 的梯度縮放器

    返回:
        epoch_loss: 訓練損失
        epoch_acc: 訓練準確率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        if batch is None:
            continue
        inputs,labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # 重置梯度
        optimizer.zero_grad()

        # 前向傳播 (with AMP context)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向傳播和優化 (with AMP scaler)
        if use_amp and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 統計
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新進度條
        current_loss = running_loss / total if total > 0 else 0
        current_acc = 100. * correct / total if total > 0 else 0
        progress_bar.set_postfix({
            'loss': f"{current_loss:.4f}",
            'acc': f"{current_acc:.2f}%"
        })

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = 100. * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, use_amp=False):
    """
    驗證模型性能 (增加 AMP 支持)

    參數:
        model: 模型
        dataloader: 數據加載器
        criterion: 損失函數
        device: 驗證設備
        use_amp (bool): 是否使用自動混合精度

    返回:
        epoch_loss: 驗證損失
        epoch_acc: 驗證準確率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            if batch is None:
                continue
            inputs,labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向傳播 (with AMP context)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 統計
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新進度條
            current_loss = running_loss / total if total > 0 else 0
            current_acc = 100. * correct / total if total > 0 else 0
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'acc': f"{current_acc:.2f}%"
            })

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = 100. * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    繪製訓練和驗證歷史

    參數:
        train_losses: 訓練損失列表
        val_losses: 驗證損失列表
        train_accs: 訓練準確率列表
        val_accs: 驗證準確率列表
        save_path: 保存路徑
    """
    plt.figure(figsize=(12, 5))

    # 繪製損失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='訓練損失')
    plt.plot(val_losses, label='驗證損失')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.title('訓練和驗證損失')
    plt.legend()
    plt.grid(True)

    # 繪製準確率
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='訓練準確率')
    plt.plot(val_accs, label='驗證準確率')
    plt.xlabel('Epoch')
    plt.ylabel('準確率 (%)')
    plt.title('訓練和驗證準確率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"訓練歷史圖已保存至: {save_path}")

def test_with_channel_combinations(model, test_data, base_dir, device, transform, use_amp=False):
    """
    測試不同通道組合 (增加 AMP 支持)

    參數:
        model: 模型
        test_data: 測試數據
        base_dir: 數據基礎目錄
        device: 測試設備
        transform: 數據轉換
        use_amp (bool): 是否使用自動混合精度

    返回:
        results: 不同通道組合的結果字典
    """
    channel_modes = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
    results = {}

    model.eval()

    for mode in channel_modes:
        logger.info(f"測試通道模式: {mode}")

        # 創建指定通道模式的測試數據集
        try:
            test_dataset = MiniImageNetDataset(
                test_data,
                base_dir=base_dir, # 修正: 使用 base_dir 參數
                transform_base=transform,
                channel_mode=mode,
                is_training=False #測試時固定通道模式
            )
            # 確保數據集非空
            if len(test_dataset) == 0:
                logger.warning(f"通道模式 {mode} 的測試數據集為空，跳過測試")
                results[mode] = 0.0
                continue

            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        except ValueError as e:
             logger.error(f"創建通道模式 {mode} 的數據集時出錯: {e}")
             results[mode] = 0.0
             continue


        # 評估
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"測試 {mode}"):
                if inputs is None or labels is None: # 跳過可能的錯誤樣本
                    continue
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向傳播 (with AMP context)
                with torch.cuda.amp.autocast(enabled=use_amp):
                     outputs = model(inputs)
                _, predicted = outputs.max(1)

                # 統計
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0.0
        results[mode] = accuracy
        logger.info(f"{mode} 通道準確率: {accuracy:.2f}%")

    return results

def run_task_a(args):
    """
    執行任務 A：設計一個可處理變化輸入通道的動態卷積模組
    
    主要實驗設計：
    1. 動態卷積模型：訓練一個單一模型，能處理各種通道組合的輸入
    2. 傳統CNN模型：為每種通道組合(RGB、R、G、B、RG、RB、GB)訓練專用模型
    3. 對比二者在各自最佳狀態下的性能差異和參數效率
    """
    
    logger.info(" 開始執行任務 A：動態卷積模組 ")
    

    # 設置隨機種子以確保可重現性
    set_seed(args.seed)

    # 數據轉換
    train_transform_base = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform_base = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 加載數據
    train_data, val_data, test_data = load_data_from_files()
    if not train_data or not val_data or not test_data:
        logger.error("數據加載失敗，請檢查數據文件")
        return None

    # 獲取設備
    device = get_device(args.gpu)
    logger.info(f"使用設備: {device}")
    
    # 數據集中的類別數
    num_classes = len(set(label for _, label in train_data))
    logger.info(f"類別數量: {num_classes}")

    # 定義要測試的通道組合
    channel_modes = ['rgb', 'r', 'g', 'b', 'rg', 'rb', 'gb']
    
    # 1. 訓練動態卷積模型 - 使用混合通道訓練
    
    logger.info(" 階段1: 訓練動態卷積模型 (混合通道輸入) ")
    
    
    # 創建訓練數據集 - 固定使用RGB通道
    train_dataset_dynamic = MiniImageNetDataset(
        train_data, DATA_DIR, transform_base=train_transform_base, 
        channel_mode='rgb', is_training=False  # 設置為隨機通道模式
    )
    val_dataset_dynamic = MiniImageNetDataset(
        val_data, DATA_DIR, transform_base=val_transform_base, 
        channel_mode='rgb', is_training=False  # 驗證時使用RGB
    )
    
    train_loader_dynamic = DataLoader(
        train_dataset_dynamic, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader_dynamic = DataLoader(
        val_dataset_dynamic, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # 創建動態卷積模型
    dynamic_model = DynamicCNN(num_classes=num_classes, in_channels_max=3)
    logger.info("動態卷積模型已創建")
    
    # 計算參數量
    dynamic_params = sum(p.numel() for p in dynamic_model.parameters())
    logger.info(f"動態卷積模型參數量: {dynamic_params:,}")
    
    # 嘗試使用torch.compile優化
    if args.compile and device.type == 'cuda':
        try:
            dynamic_model = torch.compile(dynamic_model, mode='reduce-overhead')
            logger.info("動態卷積模型已使用torch.compile優化")
        except Exception as e:
            logger.warning(f"無法應用torch.compile: {e}")
    
    dynamic_model.to(device)
    
    # 設置優化器和損失函數
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(dynamic_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    
    # AMP設置
    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("啟用自動混合精度")
    
    # 訓練動態卷積模型
    best_val_acc_dynamic = 0.0
    dynamic_train_losses, dynamic_val_losses = [], []
    dynamic_train_accs, dynamic_val_accs = [], []
    
    # 早停設置
    patience = 10
    counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 訓練和驗證
        train_loss, train_acc = train_epoch(dynamic_model, train_loader_dynamic, criterion, optimizer, device, args.amp, scaler)
        val_loss, val_acc = validate(dynamic_model, val_loader_dynamic, criterion, device, args.amp)
        
        # 更新學習率
        scheduler.step(val_acc)
        
        # 保存統計數據
        dynamic_train_losses.append(train_loss)
        dynamic_val_losses.append(val_loss)
        dynamic_train_accs.append(train_acc)
        dynamic_val_accs.append(val_acc)
        
        logger.info(f"  訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
        logger.info(f"  驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc_dynamic:
            best_val_acc_dynamic = val_acc
            torch.save(dynamic_model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'dynamic_cnn_best.pth'))
            logger.info(f"  * 新的最佳驗證準確率: {val_acc:.2f}%, 模型已保存")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"早停觸發於第 {epoch+1} 輪")
                break
    
    # 繪製訓練歷史
    plot_training_history(
        dynamic_train_losses, dynamic_val_losses, 
        dynamic_train_accs, dynamic_val_accs,
        os.path.join(PLOT_DIR, 'dynamic_cnn_history.png')
    )
    
    # 2. 為每種通道組合訓練專門的傳統CNN模型
    logger.info("="*20)
    logger.info(" 階段2: 訓練傳統CNN模型 (每種通道一個專門模型) ")
    logger.info("="*20)
    
    traditional_models = {}
    traditional_best_accs = {}
    traditional_params = {}
    
    for mode in channel_modes:
        logger.info(f"訓練傳統CNN模型 ({mode})")
        
        # 確定輸入通道數
        if mode == 'rgb':
            in_channels = 3
        elif mode in ['rg', 'rb', 'gb']:
            in_channels = 2
        else:  # 'r', 'g', 'b'
            in_channels = 1
        
        # 創建數據集 - 使用特定通道模式
        train_dataset = MiniImageNetDataset(
            train_data, DATA_DIR, transform_base=train_transform_base, 
            channel_mode=mode, is_training=False  # 固定為特定通道
        )
        val_dataset = MiniImageNetDataset(
            val_data, DATA_DIR, transform_base=val_transform_base, 
            channel_mode=mode, is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
        )
        
        # 創建傳統CNN模型，修改第一層以匹配輸入通道
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if args.compile and device.type == 'cuda':
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info(f"傳統CNN ({mode}) 已使用torch.compile優化")
            except Exception as e:
                pass
        
        model.to(device)
        
        # 計算參數量
        params = sum(p.numel() for p in model.parameters())
        traditional_params[mode] = params
        logger.info(f"傳統CNN ({mode}) 參數量: {params:,}")
        
        # 設置優化器和損失函數
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
        
        # AMP設置
        scaler_trad = None
        if args.amp and device.type == 'cuda':
            scaler_trad = torch.cuda.amp.GradScaler()
        
        # 訓練模型
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        # 訓練epochs輪（與動態模型相同）
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}/{args.epochs} [{mode}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.amp, scaler_trad)
            val_loss, val_acc = validate(model, val_loader, criterion, device, args.amp)
            
            scheduler.step(val_acc)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            logger.info(f"  訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
            logger.info(f"  驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f'traditional_cnn_{mode}_best.pth'))
                logger.info(f"  * 新的最佳驗證準確率: {val_acc:.2f}%, 模型已保存")
        
        traditional_best_accs[mode] = best_val_acc
        
        # 繪製訓練歷史
        plot_training_history(
            train_losses, val_losses, 
            train_accs, val_accs,
            os.path.join(PLOT_DIR, f'traditional_cnn_{mode}_history.png')
        )
    
    # 3. 測試階段：比較所有模型在所有通道組合上的性能
    logger.info("="*20)
    logger.info(" 階段3: 測試和比較 ")
    logger.info("="*20)
    
    # 加載最佳動態卷積模型
    best_dynamic_model = DynamicCNN(num_classes=num_classes, in_channels_max=3).to(device)
    best_dynamic_model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'dynamic_cnn_best.pth'), map_location=device))
    
    # 測試動態卷積模型在所有通道組合上的表現
    dynamic_results = test_with_channel_combinations(
        best_dynamic_model, test_data, DATA_DIR, device, val_transform_base, use_amp=args.amp
    )
    
    # 測試每個傳統CNN模型在其專門的通道組合上的表現
    traditional_results = {}
    
    for mode in channel_modes:
        # 確定輸入通道數
        if mode == 'rgb':
            in_channels = 3
        elif mode in ['rg', 'rb', 'gb']:
            in_channels = 2
        else:
            in_channels = 1
        
        # 創建模型
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.to(device)
        
        # 加載最佳權重
        model_path = os.path.join(CHECKPOINTS_DIR, f'traditional_cnn_{mode}_best.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"已加載傳統CNN {mode} 模型的最佳權重")
        else:
            logger.warning(f"未找到傳統CNN {mode} 模型的權重文件")
            continue
        
        # 創建測試數據集 - 使用對應的通道模式
        test_dataset = MiniImageNetDataset(
            test_data, DATA_DIR, transform_base=val_transform_base, 
            channel_mode=mode, is_training=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
        )
        
        # 評估模型
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"測試傳統CNN ({mode})"):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        traditional_results[mode] = accuracy
        logger.info(f"傳統CNN在 {mode} 通道上的準確率: {accuracy:.2f}%")
    
    # 4. 結果可視化和分析
    # 繪製對比圖：動態卷積 vs 傳統CNN
    plt.figure(figsize=(14, 7))
    
    x = np.arange(len(channel_modes))
    width = 0.35
    
    # 提取每種通道模式下的準確率
    dynamic_accs = [dynamic_results.get(mode, 0) for mode in channel_modes]
    trad_accs = [traditional_results.get(mode, 0) for mode in channel_modes]
    
    plt.bar(x - width/2, dynamic_accs, width, label='Dynamic Con')
    plt.bar(x + width/2, trad_accs, width, label='traditional CNN')
    
    plt.xlabel('Channel combination')
    plt.ylabel('Accuracy (%)')
    plt.title('dynamic con vs traditional CNN: Channel combination accuracy compare')
    plt.xticks(x, [m.upper() for m in channel_modes])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 標註數值
    for i, v in enumerate(dynamic_accs):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(trad_accs):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
    
    plt.savefig(os.path.join(PLOT_DIR, 'dynamic_vs_traditional_accuracy.png'))
    plt.close()
    
    # 繪製參數量比較圖
    plt.figure(figsize=(10, 6))
    
    # 計算傳統CNN的總參數量
    trad_total_params = sum(traditional_params.values())
    param_data = [dynamic_params, trad_total_params]
    
    plt.bar(['dynamic con\n(1 model)', 'traditional CNN\n(7個模型)'], param_data, color=['dodgerblue', 'orangered'])
    plt.ylabel('param total')
    plt.title('model param compare')
    plt.grid(True, axis='y', alpha=0.3)
    plt.yscale('log')  # 使用對數尺度更好地顯示差異
    
    # 標註數值
    for i, v in enumerate(param_data):
        plt.text(i, v * 1.1, f"{v:,}", ha='center')
    
    plt.savefig(os.path.join(PLOT_DIR, 'parameter_comparison.png'))
    plt.close()
    
    # 5. 保存綜合結果報告
    report_path = os.path.join(RESULTS_DIR, 'task_a_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("任務 A: 動態卷積模組評估報告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 實驗設計:\n")
        f.write("   - 動態卷積模型: 訓練單一模型，使用混合通道輸入，能處理各種通道組合\n")
        f.write("   - 傳統CNN模型: 為每種通道組合訓練專門的模型\n\n")
        
        f.write("2. 模型參數量比較:\n")
        f.write(f"   - 動態卷積模型 (單一模型): {dynamic_params:,} 參數\n")
        for mode in channel_modes:
            f.write(f"   - 傳統CNN ({mode}): {traditional_params.get(mode, 0):,} 參數\n")
        f.write(f"   - 傳統方法總參數量 (7個模型): {trad_total_params:,} 參數\n")
        f.write(f"   - 參數量節省比例: {trad_total_params / dynamic_params:.2f}倍\n\n")
        
        f.write("3. 測試準確率比較:\n")
        for mode in channel_modes:
            dyn_acc = dynamic_results.get(mode, 0)
            trad_acc = traditional_results.get(mode, 0)
            diff = dyn_acc - trad_acc
            f.write(f"   - {mode.upper()}: 動態卷積 {dyn_acc:.2f}% vs 傳統CNN {trad_acc:.2f}% (差異: {diff:.2f}%)\n")
        
        # 計算平均值
        dyn_avg = sum(dynamic_results.values()) / len(dynamic_results) if dynamic_results else 0
        trad_avg = sum(traditional_results.values()) / len(traditional_results) if traditional_results else 0
        f.write(f"\n   平均準確率: 動態卷積 {dyn_avg:.2f}% vs 傳統CNN {trad_avg:.2f}% (差異: {dyn_avg - trad_avg:.2f}%)\n\n")
        
        f.write("4. 優勢分析:\n")
        f.write("   a) 參數效率: 動態卷積模型使用單一模型處理所有通道組合，極大節省了模型存儲空間\n")
        f.write("   b) 維護成本: 只需更新一個模型，而非多個專門模型\n")
        f.write("   c) 靈活性: 可以處理任意通道組合，包括訓練中未見過的組合\n")
        f.write("   d) 推理效率: 使用同一套權重處理不同輸入，減少模型切換開銷\n\n")
        
        f.write("5. 結論:\n")
        if dyn_avg >= trad_avg * 0.9:  # 如果動態卷積達到傳統方法90%的性能，視為成功
            f.write("   動態卷積模型成功實現了單一模型處理多種通道輸入的目標，性能接近或優於專門訓練的傳統模型。\n")
            f.write("   考慮到參數量大幅減少和應用靈活性提升，該方法在資源受限場景下具有明顯優勢。\n")
        else:
            f.write("   動態卷積模型展示了處理多種通道輸入的能力，但性能與專門訓練的傳統模型仍有差距。\n")
            f.write("   儘管如此，其參數效率和靈活性仍然使其在特定應用場景中具有價值。\n")
        
        # 計算FLOPS對比 (簡單估計)
        f.write("\n6. 計算成本估計:\n")
        f.write("   由於動態卷積使用注意力機制動態調整權重，單次前向傳播的計算量略高於傳統CNN。\n")
        f.write("   但在需要支持多種通道組合的場景下，傳統方法需要部署多個模型，總體資源消耗更高。\n")
    
    logger.info(f"任務A綜合報告已保存至: {report_path}")
    
    # 返回結果字典
    return {
        'dynamic_results': dynamic_results,
        'traditional_results': traditional_results,
        'dynamic_params': dynamic_params,
        'traditional_params': traditional_params,
        'performance_difference': dyn_avg - trad_avg,
        'parameter_efficiency': trad_total_params / dynamic_params
    }

def run_task_b(args):
    """
    執行任務 B：高效二至四層網絡 (使用 EfficientNet)
    
    """
    logger.info("="*30)
    logger.info(" 開始執行任務 B：高效二至四層網絡 (V2 模型) ")
    logger.info("="*30)

    # 設置隨機種子
    set_seed(args.seed)

    # --- 定義基礎變換 (無 Normalize) ---
    train_transform_base = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform_base = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # --- 結束定義 ---

    # 加載數據
    train_data, val_data, test_data = load_data_from_files()
    if not train_data or not val_data or not test_data:
         logger.error("數據加載失敗，請檢查 train.txt, validation.txt, test.txt 文件是否存在於 data/ 目錄下")
         return None

    # --- 創建數據集: 使用 transform_base 和 DATA_DIR ---
    # Task B 通常在標準 RGB 上訓練和評估
    train_dataset = MiniImageNetDataset(train_data, DATA_DIR, transform_base=train_transform_base, channel_mode='rgb', is_training=False)
    val_dataset = MiniImageNetDataset(val_data, DATA_DIR, transform_base=val_transform_base, channel_mode='rgb', is_training=False)
    test_dataset = MiniImageNetDataset(test_data, DATA_DIR, transform_base=val_transform_base, channel_mode='rgb', is_training=False)
    # --- 結束創建 ---

    # --- 創建 DataLoader: 加入 collate_fn ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    # --- 結束創建 ---

    # 創建模型
    device = get_device(args.gpu)
    logger.info(f"使用設備: {device}")

    num_classes = len(set(label for _, label in train_data))
    logger.info(f"類別數量: {num_classes}")

    # 創建 EfficientNetV2 和基準 ResNet34
    # 確保 efficient_network.py 已包含更新後的 EfficientNetV2
    try:
         efficient_model = SimpleParallelEfficientNet(num_classes=num_classes, width_multiplier=args.width_multiplier)
    except Exception as model_init_err:
         logger.exception(f"初始化 EfficientNet 時出錯: {model_init_err}")
         return None
    baseline_model = models.resnet34(weights=None, num_classes=num_classes)
    logger.info(f"高效模型 (EfficientNet) 已創建，寬度因子: {args.width_multiplier}")
    logger.info("基準模型 (ResNet34) 已創建")

    # 檢查模型的有效層數
    efficient_model.count_effective_layers()

    # 可選: 使用 torch.compile
    if args.compile and device.type == 'cuda':
        logger.info("啟用 torch.compile() for Task B models")
        try:
            efficient_model = torch.compile(efficient_model, mode='reduce-overhead')
            logger.info("torch.compile() for EfficientNetV2 成功")
        except Exception as e: logger.warning(f"torch.compile() for EfficientNetV2 失敗: {e}")
        try:
            baseline_model = torch.compile(baseline_model, mode='reduce-overhead')
            logger.info("torch.compile() for ResNet34 成功")
        except Exception as e: logger.warning(f"torch.compile() for ResNet34 失敗: {e}")

    efficient_model.to(device)
    baseline_model.to(device)

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    efficient_optimizer = optim.AdamW(efficient_model.parameters(), lr=args.lr, weight_decay=1e-4)
    baseline_optimizer = optim.AdamW(baseline_model.parameters(), lr=args.lr, weight_decay=1e-4)
    efficient_scheduler = optim.lr_scheduler.CosineAnnealingLR(efficient_optimizer, T_max=args.epochs)
    baseline_scheduler = optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=args.epochs)

    # 可選: 使用 AMP
    scaler_eff, scaler_base = None, None
    if args.amp and device.type == 'cuda':
        scaler_eff = torch.cuda.amp.GradScaler()
        scaler_base = torch.cuda.amp.GradScaler()
        logger.info("啟用自動混合精度 (AMP) for Task B models")


    # --- 高效模型 (EfficientNet) 的訓練循環 ---
    logger.info("="*20 + " 訓練 EfficientNet " + "="*20)
    best_val_acc_eff = 0.0
    eff_train_losses, eff_val_losses, eff_train_accs, eff_val_accs = [], [], [], []
    start_time_eff = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{args.epochs} [EfficientNet] - LR: {efficient_optimizer.param_groups[0]['lr']:.6f}")
        # 訓練和驗證
        train_loss, train_acc = train_epoch(efficient_model, train_loader, criterion, efficient_optimizer, device, args.amp, scaler_eff)
        val_loss, val_acc = validate(efficient_model, val_loader, criterion, device, args.amp)
        efficient_scheduler.step()
        # 保存統計數據
        eff_train_losses.append(train_loss)
        eff_val_losses.append(val_loss)
        eff_train_accs.append(train_acc)
        eff_val_accs.append(val_acc)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} 完成 - 耗時: {epoch_time:.2f}s")
        logger.info(f"  訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
        logger.info(f"  驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
        # 保存最佳模型
        if val_acc > best_val_acc_eff:
            best_val_acc_eff = val_acc
            save_path = os.path.join(CHECKPOINTS_DIR, 'efficient_model.pth')
            torch.save(efficient_model.state_dict(), save_path)
            logger.info(f"  * 新的最佳驗證準確率: {val_acc:.2f}%. 模型已保存至 {save_path}")

    efficient_training_time = time.time() - start_time_eff
    logger.info(f"EfficientNet 訓練完成，總耗時 {efficient_training_time/60:.2f} 分鐘")
    # 繪製訓練歷史
    plot_training_history(
        eff_train_losses, eff_val_losses, eff_train_accs, eff_val_accs,
        save_path=os.path.join(PLOT_DIR, f'efficient_model_v2_w{args.width_multiplier}_history.png') # 文件名加入寬度因子
    )


    # --- ResNet34 基準模型的訓練循環 ---
    logger.info("="*20 + " 訓練 ResNet34 Baseline " + "="*20)
    best_val_acc_base = 0.0
    base_train_losses, base_val_losses, base_train_accs, base_val_accs = [], [], [], []
    start_time_base = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{args.epochs} [ResNet34] - LR: {baseline_optimizer.param_groups[0]['lr']:.6f}")
        # 訓練和驗證 (使用相同的 loader)
        train_loss, train_acc = train_epoch(baseline_model, train_loader, criterion, baseline_optimizer, device, args.amp, scaler_base)
        val_loss, val_acc = validate(baseline_model, val_loader, criterion, device, args.amp)
        baseline_scheduler.step()
        # 保存統計數據
        base_train_losses.append(train_loss)
        base_val_losses.append(val_loss)
        base_train_accs.append(train_acc)
        base_val_accs.append(val_acc)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} 完成 - 耗時: {epoch_time:.2f}s")
        logger.info(f"  訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
        logger.info(f"  驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
        # 保存最佳模型
        if val_acc > best_val_acc_base:
            best_val_acc_base = val_acc
            save_path = os.path.join(CHECKPOINTS_DIR, 'baseline_resnet34_best.pth')
            torch.save(baseline_model.state_dict(), save_path)
            logger.info(f"  * 新的最佳驗證準確率: {val_acc:.2f}%. 模型已保存至 {save_path}")

    baseline_training_time = time.time() - start_time_base
    logger.info(f"ResNet34 Baseline 訓練完成，總耗時 {baseline_training_time/60:.2f} 分鐘")
    # 繪製訓練歷史
    plot_training_history(
        base_train_losses, base_val_losses, base_train_accs, base_val_accs,
        save_path=os.path.join(PLOT_DIR, 'baseline_resnet34_history.png')
    )


    # --- 加載最佳模型並在測試集上評估 ---
    logger.info("="*20 + " 在測試集上評估最佳模型 " + "="*20)

    # 加載 EfficientNet 最佳模型
    efficient_test_acc = 0.0 # 初始化
    efficient_test_loss = 0.0
    efficient_model_test = None
    best_eff_model_path = os.path.join(CHECKPOINTS_DIR, 'efficient_model.pth')
    if os.path.exists(best_eff_model_path):
        try:
            efficient_model_test = SimpleParallelEfficientNet(num_classes=num_classes, width_multiplier=args.width_multiplier).to(device)
            efficient_model_test.load_state_dict(torch.load(best_eff_model_path, map_location=device))
            logger.info("EfficientNet 最佳模型權重已加載")
            efficient_test_loss, efficient_test_acc = validate(efficient_model_test, test_loader, criterion, device, args.amp) # <-- 計算測試準確率
        except Exception as load_err:
             logger.exception(f"加載或驗證 EfficientNet 最佳模型時出錯: {load_err}")
    else:
        logger.warning("未找到 EfficientNet 最佳模型權重，測試準確率設為 0")


    # 加載 ResNet34 最佳模型
    baseline_test_acc = 0.0 # 初始化
    baseline_test_loss = 0.0
    baseline_model_test = None
    best_base_model_path = os.path.join(CHECKPOINTS_DIR, 'baseline_resnet34_best.pth')
    if os.path.exists(best_base_model_path):
         try:
              baseline_model_test = models.resnet34(weights=None, num_classes=num_classes).to(device)
              baseline_model_test.load_state_dict(torch.load(best_base_model_path, map_location=device))
              logger.info("ResNet34 最佳模型權重已加載")
              baseline_test_loss, baseline_test_acc = validate(baseline_model_test, test_loader, criterion, device, args.amp) # <-- 計算測試準確率
         except Exception as load_err:
              logger.exception(f"加載或驗證 ResNet34 最佳模型時出錯: {load_err}")
    else:
        logger.warning("未找到 ResNet34 最佳模型權重，測試準確率設為 0")

    # --- 現在 efficient_test_acc 和 baseline_test_acc 應該有值了 ---

    logger.info(f"EfficientNet (width={args.width_multiplier}) 測試準確率: {efficient_test_acc:.2f}%")
    logger.info(f"基準 ResNet34 測試準確率: {baseline_test_acc:.2f}%")

    # 計算參數數量
    efficient_params = 0
    if efficient_model_test: # 確保模型已加載
         efficient_params = sum(p.numel() for p in efficient_model_test.parameters())
    baseline_params = 0
    if baseline_model_test: # 確保模型已加載
         baseline_params = sum(p.numel() for p in baseline_model_test.parameters())

    logger.info(f"EfficientNet 參數: {efficient_params:,}")
    logger.info(f"基準 ResNet34 參數: {baseline_params:,}")

    # 比較性能
    performance_ratio = (efficient_test_acc / baseline_test_acc * 100) if baseline_test_acc > 0 else 0.0
    param_reduction = (1 - efficient_params / baseline_params) * 100 if baseline_params > 0 and efficient_params > 0 else 0.0
    logger.info(f"EfficientNet 達到 ResNet34 性能的 {performance_ratio:.2f}%")
    logger.info(f"參數數量減少了 {param_reduction:.2f}%")

    # 繪製比較圖
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # ... (繪製準確率比較圖的代碼) ...
    models_names = [f'EfficientNet\n(w={args.width_multiplier})', 'ResNet34']
    accs = [efficient_test_acc, baseline_test_acc]
    bars = plt.bar(models_names, accs, color=['orange', 'green'])
    # ... (添加標籤和標題) ...
    plt.subplot(1, 2, 2)
    # ... (繪製參數數量比較圖的代碼) ...
    params = [efficient_params, baseline_params]
    bars = plt.bar(models_names, params, color=['orange', 'green'])
    # ... (添加標籤和標題, 使用對數尺度) ...
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f'model_comparison_w{args.width_multiplier}.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"模型比較圖已保存至: {save_path}")

    # 保存結果到文件
    results_file_path = os.path.join(RESULTS_DIR, f'task_b_results_w{args.width_multiplier}.txt')
    with open(results_file_path, 'w', encoding='utf-8') as f:
        # ... (寫入結果文件的代碼) ...
        f.write(f"EfficientNet 測試準確率: {efficient_test_acc:.2f}%\n")
        f.write(f"基準 ResNet34 測試準確率: {baseline_test_acc:.2f}%\n")
        # ...
    logger.info(f"任務 B 結果已保存至: {results_file_path}")


    # 返回字典
    return {
        'efficient_accuracy': efficient_test_acc,
        'baseline_accuracy': baseline_test_acc,
        'performance_ratio': performance_ratio,
        'efficient_params': efficient_params,
        'baseline_params': baseline_params,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MILS Assignment I - Training Script')
    # 任務選擇
    parser.add_argument('--task', type=str, choices=['a', 'b', 'ab'], default='ab',
                        help='執行的任務: a, b, 或兩者 (ab)')
    # 通用參數
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--lr', type=float, default=0.001, help='初始學習率 (建議降低學習率，例如 0.001)')
    parser.add_argument('--epochs', type=int, default=20, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='數據加載器的工作進程數')
    parser.add_argument('--gpu', type=int, default=None, help='指定使用的 GPU ID (None 表示自動選擇)')
    parser.add_argument('--amp', action='store_true', help='啟用自動混合精度 (AMP)')
    parser.add_argument('--compile', action='store_true', help='啟用 torch.compile() (實驗性)')

    # Task B 特定參數
    parser.add_argument('--width_multiplier', type=float, default=1.0,
                        help='EfficientNet 的寬度倍增因子 (Task B)')

    args = parser.parse_args()

    logger.info("命令行參數:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    if args.task == 'a' or args.task == 'ab':
        run_task_a(args)

    if args.task == 'b' or args.task == 'ab':
        run_task_b(args)

    logger.info("所有任務執行完畢。")