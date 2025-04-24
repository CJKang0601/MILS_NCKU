# dynamic_convolution.py (恢復為 Attention over Kernels 版本)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DynamicConvolution(nn.Module):
    """
    動態卷積模組 (基於 Attention over Kernels)
    參考: https://arxiv.org/abs/1912.03458
    """
    def __init__(self, in_channels_max, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4, reduction=4):
        """
        Args:
            in_channels_max (int): 預計最大輸入通道數 (e.g., 3 for RGB).
            out_channels (int): 輸出通道數.
            kernel_size (int): 卷積核大小.
            stride (int): 步長.
            padding (int): 填充.
            dilation (int): 膨脹率.
            groups (int): 分組卷積組數.
            bias (bool): 是否使用偏置.
            K (int): 要生成和組合的卷積核數量.
            reduction (int): 注意力網路中間層的縮減因子.
        """
        super(DynamicConvolution, self).__init__()
        if in_channels_max <= 0 or reduction <= 0:
             raise ValueError("in_channels_max and reduction must be positive.")
        reduced_dim = max(1, in_channels_max // reduction) # 確保至少為 1

        self.in_channels_max = in_channels_max
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K

        # 注意力網路 (恢復使用 nn.Sequential)
        # 注意：輸入維度固定為 in_channels_max
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels_max, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, K),
            nn.Softmax(dim=1) # 在 dim=1 (K 維度) 上 Softmax
        )

        # K 組卷積核的權重參數
        self.weights = nn.Parameter(
            torch.Tensor(K, out_channels, in_channels_max, kernel_size, kernel_size),
            requires_grad=True
        )
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # Kaiming He initialization

        # 偏置參數
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        # logger.info(f"DynamicConvolution (Attention over Kernels) initialized: K={K}, reduction={reduction}")


    def _adapt_input_padding(self, x):
        """將輸入 x 的通道數調整為 self.in_channels_max (使用零填充)"""
        b, c, h, w = x.shape
        if c == self.in_channels_max:
            return x
        elif c < self.in_channels_max:
            padding_size = self.in_channels_max - c
            if padding_size > 0 :
                 padding = torch.zeros(b, padding_size, h, w, device=x.device, dtype=x.dtype)
                 adapted_x = torch.cat([x, padding], dim=1)
                 return adapted_x
            else:
                 return x # Should not happen if c < in_channels_max
        else: # c > self.in_channels_max
            # logger.warning(f"Input channels ({c}) > max_channels ({self.in_channels_max}). Slicing.")
            return x[:, :self.in_channels_max, :, :]

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map (batch, in_channels, height, width).
                       in_channels can be variable.
        """
        if x is None or x.numel() == 0:
            logger.error("DynamicConvolution received None or empty input tensor.")
            bs = 1 if x is None else x.shape[0]
            out_h = 1; out_w = 1 # Dummy H, W
            try: # Try to calculate output size if possible
                 h_in = 1 if x is None else x.shape[2]
                 w_in = 1 if x is None else x.shape[3]
                 out_h = (h_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
                 out_w = (w_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
            except: pass
            return torch.zeros((bs, self.out_channels, out_h, out_w), device=self.weights.device)


        batch_size, in_channels, height, width = x.shape

        # 1. 適應輸入通道數 (使用零填充)
        adapted_x = self._adapt_input_padding(x) # Shape: (B, C_max, H, W)

        # 2. 計算注意力權重
        att_weights = None # Default
        try:
             # 注意力網路期望輸入 (B, C_max, H, W)
             att = self.attention(adapted_x) # Output should be (B, K)

             # 檢查維度 (以防萬一，例如 batch_size=1)
             if att.dim() == 1 and batch_size == 1:
                  att = att.unsqueeze(0) # Reshape to (1, K)

             if att.shape == (batch_size, self.K):
                  att_weights = att
             else:
                  logger.error(f"Attention output shape mismatch: expected ({batch_size}, {self.K}), got {att.shape}. Using uniform weights.")
                  att_weights = torch.ones(batch_size, self.K, device=x.device, dtype=x.dtype) / self.K

        except Exception as attn_err:
             logger.exception(f"ERROR [DynConv Attn]: Error during attention calculation: {attn_err}")
             att_weights = torch.ones(batch_size, self.K, device=x.device, dtype=x.dtype) / self.K


        # 確保 att_weights 有定義
        if att_weights is None:
             logger.error("ERROR [DynConv]: att_weights is None after attention block. Using default weights.")
             att_weights = torch.ones(batch_size, self.K, device=x.device, dtype=x.dtype) / self.K


        # 3. 動態組合卷積核權重並執行卷積 (逐樣本)
        outputs = []
        # Pre-calculate output size
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        for i in range(batch_size):
            try:
                if att_weights is None or att_weights.dim() != 2 or i >= att_weights.shape[0]:
                    logger.error(f"ERROR [DynConv ConvLoop]: Invalid att_weights for sample {i}. Shape: {att_weights.shape if att_weights is not None else 'None'}. Skipping.")
                    outputs.append(torch.zeros((1, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype))
                    continue

                sample_att = att_weights[i] # Shape: (K,)
                if sample_att.dim() != 1 or sample_att.shape[0] != self.K:
                    logger.error(f"ERROR [DynConv ConvLoop]: Invalid sample_att dimension for sample {i}. Shape: {sample_att.shape}. Expected ({self.K},). Skipping.")
                    outputs.append(torch.zeros((1, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype))
                    continue

                # Combine kernels
                # Use view for broadcasting: (K,) -> (K, 1, 1, 1, 1)
                weighted_kernel = torch.sum(sample_att.view(self.K, 1, 1, 1, 1) * self.weights, dim=0)

                # Combine biases if they exist
                current_bias = None
                if self.bias is not None:
                    # Use view for broadcasting: (K,) -> (K, 1)
                    weighted_bias = torch.sum(sample_att.view(self.K, 1) * self.bias, dim=0)
                    current_bias = weighted_bias

                # Apply convolution
                sample_output = F.conv2d(
                    adapted_x[i:i+1], # Input: (1, C_max, H, W)
                    weight=weighted_kernel,
                    bias=current_bias,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
                )
                outputs.append(sample_output)

            except Exception as loop_err:
                logger.exception(f"ERROR [DynConv ConvLoop]: Unexpected error for sample {i}: {loop_err}")
                outputs.append(torch.zeros((1, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype))
                continue

        # Concatenate results
        if not outputs or len(outputs) != batch_size:
            logger.error(f"ERROR [DynConv]: Output list size ({len(outputs)}) != batch size ({batch_size}). Returning zeros.")
            return torch.zeros((batch_size, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype)

        try:
            output = torch.cat(outputs, dim=0)
        except Exception as cat_err:
            logger.exception(f"ERROR [DynConv]: Failed to concatenate outputs: {cat_err}")
            # Attempt to return zeros with correct shape
            return torch.zeros((batch_size, self.out_channels, out_h, out_w), device=x.device, dtype=x.dtype)

        return output


# 構建使用 DynamicConvolution 的 CNN 模型 (與之前的 DynamicCNN 相同)
import math # 確保導入 math
class DynamicCNN(nn.Module):
    def __init__(self, num_classes=100, in_channels_max=3):
        super(DynamicCNN, self).__init__()
        logger.info("Initializing DynamicCNN (Attention over Kernels version)")

        self.dyn_conv1 = DynamicConvolution(
            in_channels_max=in_channels_max,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            K=4,
            reduction=4
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, stride=1):
        # 簡化的層
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        # logger.info("Initializing weights for standard layers in DynamicCNN")
        for name, m in self.named_modules():
             is_in_dyn_conv = False
             for n, mod in self.named_modules():
                  if mod == m and n.startswith('dyn_conv1.'):
                       is_in_dyn_conv = True
                       break

             if is_in_dyn_conv: continue # 跳過動態卷積內部的層

             if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
             elif isinstance(m, nn.Linear) and m != self.fc:
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.constant_(m.bias, 0)

        if hasattr(self, 'fc') and isinstance(self.fc, nn.Linear):
             nn.init.normal_(self.fc.weight, 0, 0.01)
             nn.init.constant_(self.fc.bias, 0)


    def forward(self, x):
        # logger.debug(f"DynamicCNN Input shape: {x.shape}")
        x = self.dyn_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x