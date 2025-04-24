# MILS Assignment I - 高效卷積網絡設計

本儲存庫包含 MILS 課程作業一的實現，專注於變化通道輸入的動態卷積模組設計以及高效二至四層網絡的設計。

## 專案概述

本作業實現了兩個主要任務：

### 任務 A：變化輸入通道的動態卷積模組

- 設計了一個能處理任意輸入通道數量的特殊卷積模組
- 基於注意力機制（Attention over Kernels）動態調整卷積核
- 比較不同通道組合（RGB、RG、GB、R、G、B 等）的性能
- 與傳統模型（每種通道組合訓練一個專門模型）進行比較
- 分析計算成本與參數效率

### 任務 B：高效二至四層圖像分類網絡

- 實現了一個僅有 4 個有效層的高效網絡架構
- 使用並行路徑（卷積路徑和注意力路徑）擴展感受野
- 達到 ResNet34 至少 90% 的性能表現
- 使用自注意力機制提升特徵提取效率

## 數據集

本作業使用 mini-ImageNet 數據集，組織如下：
- `train.txt`：訓練圖像的文件路徑和標籤
- `validation.txt`：驗證圖像的文件路徑和標籤
- `test.txt`：測試圖像的文件路徑和標籤

## 開始使用

### 環境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- tqdm

### 安裝

1. 克隆此儲存庫：
```bash
git clone https://github.com/CJKang0601/MILS_NCKU.git
cd MILS_NCKU
```

2. 安裝依賴項：
```bash
pip install -r requirements.txt
```

3. 下載數據集：
```bash
# 從提供的 URL 下載
wget https://cchsu.info/files/images.zip

# 解壓到 data 目錄
mkdir -p data
unzip images.zip -d data/
```

### 項目結構

```
.
├── README.md
├── requirements.txt
├── dynamic_convolution.py     # 任務 A 實現：動態卷積模組
├── efficient_network.py       # 任務 B 實現：高效網絡
├── utils.py                   # 工具函數
├── train.py                   # 訓練腳本
└── data/                      # 數據集文件夾
    ├── images/                # 圖像文件
    ├── train.txt              # 訓練列表
    ├── validation.txt         # 驗證列表
    └── test.txt               # 測試列表
```

## 運行實驗

### 任務 A：動態卷積模組

運行任務 A 的訓練和評估：

```bash
python train.py --task a
```

此操作將：
1. 訓練一個能處理變化通道輸入的動態卷積網絡
2. 測試網絡在不同通道組合（RGB、R、G、B、RG、GB、RB）上的性能
3. 與傳統基準模型進行比較
4. 生成對比圖表並保存結果

### 任務 B：高效網絡

運行任務 B 的訓練和評估：

```bash
python train.py --task b
```

此操作將：
1. 訓練四層高效網絡
2. 訓練 ResNet34 模型作為基準
3. 比較性能和模型大小
4. 生成對比圖表

### 運行兩個任務

要依次運行兩個任務：

```bash
python train.py --task ab
```

### 自定義訓練參數

您可以調整以下參數來自定義訓練過程：

```bash
python train.py --task ab --epochs 30 --batch_size 128 --lr 0.0005 --width_multiplier 1.5 --amp
```

參數說明：
- `--epochs`：訓練輪數
- `--batch_size`：批次大小
- `--lr`：學習率
- `--width_multiplier`：任務 B 中網絡的寬度乘數
- `--amp`：啟用自動混合精度訓練
- `--seed`：隨機種子
- `--num_workers`：數據加載線程數
- `--gpu`：指定 GPU ID
- `--compile`：啟用 torch.compile() 優化

## 模型架構

### 動態卷積模組（任務 A）

動態卷積模組基於 "Dynamic Convolution: Attention over Convolution Kernels" 論文，設計為處理變化輸入通道：

1. 核心思想：使用注意力機制動態生成和組合卷積核
2. 自適應輸入通道：能處理任意通道數的輸入圖像
3. 保持輸出維度一致

實現亮點：
- 注意力機制為每個樣本動態計算卷積核權重
- 參數在不同通道配置間高效共享
- 單一模型支持所有通道組合，無需重新訓練

### 高效網絡（任務 B）

高效網絡僅使用 4 個有效層但實現了與 ResNet34 相當的性能：

1. 使用多路徑並行架構：常規卷積 + 自注意力
2. 增加通道數和特徵表示能力
3. 在特徵提取過程中融合空間和通道信息

實現亮點：
- 第 1 層：共享初始卷積層
- 第 2 層：並聯路徑（卷積 + 自注意力）
- 第 3 層：特徵融合層
- 第 4 層：最終特徵提取層

## 實驗結果

實驗結果將保存在 `results` 文件夾下：
- `results/plots`：包含訓練曲線和性能比較圖
- `results/checkpoints`：保存最佳模型權重
- `results/task_a_report.txt` 和 `results/task_b_results_*.txt`：詳細實驗報告

## 參考文獻

1. Wu, Y., et al. (2020). Dynamic Convolution: Attention over Convolution Kernels. CVPR 2020.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
3. Wang, X., et al. (2018). Non-local Neural Networks. CVPR 2018.

## 作者

Cheng Jun Kang  
re6144051@gs.ncku.edu.tw