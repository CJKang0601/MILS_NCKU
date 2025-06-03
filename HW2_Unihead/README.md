# EfficientNet-B0 多任務學習解決方案

## 📝 專案描述

本專案實現了一個採用 EfficientNet-B0 作為骨幹網路 (backbone) 的統一輸出頭 (Unified Head) 多任務學習模型。該模型經過訓練，可同時執行三種不同的任務：語義分割 (Semantic Segmentation)、物件偵測 (Object Detection) 和圖像分類 (Image Classification)。此解決方案旨在高效運行於 GPU，並採用混合精度 (Mixed Precision) 訓練以優化性能和記憶體使用。

主要目標是控制災難性遺忘 (Catastrophic Forgetting)，確保模型在每個任務上的性能與單獨為該任務訓練的基準模型 (baselines) 相比，下降幅度不超過 5%。

## ✨ 主要特點

* **多任務學習 (Multi-Task Learning)**：訓練單一模型執行分割、偵測和分類任務。
* **統一輸出頭架構**：在各任務特定的輸出層之前，使用少量共享的卷積層 (2-3層)。
* **EfficientNet-B0 骨幹網路**：使用在 ImageNet 上預訓練的 EfficientNet-B0 (530萬參數) 作為骨幹網路，並確保模型總參數不超過800萬。
* **控制災難性遺忘**：旨在保持各任務性能接近其單獨訓練的基準水平 (性能下降 ≤ 5%)。
* **獨立數據集**：為每個任務使用三個獨立的迷你數據集 (mini\_voc\_seg, mini-coco-det, imagenette-160)。
* **GPU 優化**：
    * 使用 `torch.cuda.amp` 進行混合精度訓練。
    * 根據可用 GPU 記憶體動態調整批次大小 (Dynamic Batch Size)。
    * GPU 記憶體管理，以避免記憶體不足 (OOM) 錯誤。
    * 使用 `pin_memory` 和 `non_blocking` 優化數據加載。
* **基準模型訓練**：訓練單任務模型作為性能比較參考。
* **檢查點 (Checkpointing)**：在訓練過程中定期保存檢查點 (尤其在 Colab 環境中)。
* **評估與視覺化**：評估模型在各任務上的性能，並生成訓練歷史和性能比較的圖表。
* **報告生成**：生成總結實驗結果的最終報告。
* **彈性環境支援**：支援 Google Colab 和本地環境，並提供相應的路徑設置。

## 🏗️ 模型架構

模型使用 EfficientNet-B0 作為骨幹網路提取特徵。這些特徵隨後通過一個 "頸部" (Neck) 層以減少維度，接著是數個共享的卷積層 (統一輸出頭)。最後，特徵分支到各任務特定的輸出層。


![image](https://github.com/user-attachments/assets/bc495793-f2b5-495a-8d02-d6d9f5e7b85f)



## 實驗結果比較(單任務v.s.多任務)
![image](https://github.com/user-attachments/assets/c5444a75-c209-4fc0-b5f5-7c2a6c1adac1)


## LLM 對話 (Claude)
(1)[https://claude.ai/share/1f9887cb-adec-42b9-a81d-e6eb8193b7cb]
(2)[https://claude.ai/share/07863ba0-629c-4644-b763-27dcbb345ae2]

