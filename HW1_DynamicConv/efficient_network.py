# simple_parallel_efficient_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

# --- 簡化版自注意力模塊 ---
class SimpleSelfAttention(nn.Module):
    """
    簡化版自注意力模塊，不使用複雜的混合結構
    """
    def __init__(self, in_channels, out_channels=None):
        super(SimpleSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        # 降維以減少計算量
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        
        # 輸出投影（可選）
        if self.out_channels != self.in_channels:
            self.projection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()
        
        # 初始化伽馬參數為0
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 計算查詢、鍵、值
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C'
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C' x HW
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # 計算注意力權重 (使用縮放點積注意力)
        energy = torch.bmm(proj_query, proj_key)  # B x HW x HW
        scaling_factor = math.sqrt(self.in_channels // 8)
        attention = F.softmax(energy / scaling_factor, dim=-1)  # B x HW x HW
        
        # 應用注意力權重
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, self.out_channels, height, width)  # B x C x H x W
        
        # 殘差連接 (加權混合)
        out = self.gamma * out + self.projection(x)
        
        return out
    # def __init__(self, in_channels, out_channels=None, heads=4):
    #     super(SimpleSelfAttention, self).__init__()
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels or in_channels
    #     self.heads = heads
        
    #     # 每個頭的特徵維度
    #     self.head_dim = in_channels // heads
        
    #     # 降維以減少計算量
    #     self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    #     self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    #     self.value_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        
    #     # 輸出投影
    #     self.projection = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        
    #     # 初始化伽馬參數
    #     self.gamma = nn.Parameter(torch.zeros(1))
    
    # def forward(self, x):
    #     batch_size, C, height, width = x.size()
        
    #     # 多頭處理
    #     q = self.query_conv(x).view(batch_size, self.heads, self.head_dim, height * width)
    #     k = self.key_conv(x).view(batch_size, self.heads, self.head_dim, height * width)
    #     v = self.value_conv(x).view(batch_size, self.heads, self.out_channels // self.heads, height * width)
        
    #     # 轉置為適合注意力計算的形狀
    #     q = q.permute(0, 1, 3, 2)  # B x Heads x HW x C/Heads
    #     k = k.permute(0, 1, 2, 3)  # B x Heads x C/Heads x HW
    #     v = v.permute(0, 1, 2, 3)  # B x Heads x C/Heads x HW
        
    #     # 計算注意力 (使用縮放點積注意力)
    #     attn = torch.matmul(q, k)  # B x Heads x HW x HW
    #     attn = attn / math.sqrt(self.head_dim)
    #     attn = F.softmax(attn, dim=-1)
        
    #     # 應用注意力權重
    #     out = torch.matmul(attn, v.transpose(-2, -1))  # B x Heads x HW x C/Heads
    #     out = out.permute(0, 1, 3, 2).contiguous()  # B x Heads x C/Heads x HW
    #     out = out.view(batch_size, self.out_channels, height, width)  # B x C x H x W
        
    #     # 殘差連接 (加權混合)
    #     out = self.gamma * self.projection(out) + x
        
    #     return out

# --- 簡化的並聯高效網絡 ---
class SimpleParallelEfficientNet(nn.Module):
    """
    簡化的並聯架構高效網絡
    
    設計特點:
    1. 使用4個基本有效層
    2. 兩條並行路徑: 卷積路徑和注意力路徑
    3. 不使用複雜的混合結構如SEBlock
    4. 確保有效層數符合任務要求
    """
    def __init__(self, num_classes=100, in_channels=3, width_multiplier=1.0):
        super(SimpleParallelEfficientNet, self).__init__()
        logger.info(f"初始化簡化版並聯高效網絡，有效層數=4，寬度乘數={width_multiplier}")
        
        # 通道縮放函數
        ch = lambda c: max(1, int(c * width_multiplier))
        
        # --- 第1層: 共享初始層 ---
        # 這是第一個有效層
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, ch(64), kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch(64)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # --- 第2層: 並聯路徑 ---
        # 卷積路徑 (第二個有效層，路徑 A)
        # self.conv_path = nn.Sequential(
        #     nn.Conv2d(ch(64), ch(128), kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(ch(128)),
        #     nn.ReLU(inplace=True)
        # )
        
        # 注意力路徑 (第二個有效層的替代路徑，路徑 B)
        # self.attention_path = nn.Sequential(
        #     SimpleSelfAttention(ch(64), ch(128)),
        #     nn.BatchNorm2d(ch(128)),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # 保持空間大小與卷積路徑一致
        # )
        # 卷積路徑 (第二個有效層，路徑 A)

        self.conv_path = nn.Sequential(
            nn.Conv2d(ch(64), ch(256), kernel_size=3, stride=2, padding=1, bias=False),  # 增加通道數
            nn.BatchNorm2d(ch(256)),
            nn.ReLU(inplace=True)
        )

        # 注意力路徑 (第二個有效層的替代路徑，路徑 B)
        self.attention_path = nn.Sequential(
            SimpleSelfAttention(ch(64), ch(256)),  # 匹配通道數
            nn.BatchNorm2d(ch(256)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 修改融合層以適應更寬的輸入
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(ch(256) * 2, ch(384), kernel_size=1, bias=False),  # 增加通道數
            nn.BatchNorm2d(ch(384)),
            nn.ReLU(inplace=True)
        )
        # --- 第3層: 特徵融合層 ---
        # 第三個有效層
        # self.fusion_layer = nn.Sequential(
        #     nn.Conv2d(ch(128) * 2, ch(256), kernel_size=1, bias=False),  # 1x1 卷積融合
        #     nn.BatchNorm2d(ch(256)),
        #     nn.ReLU(inplace=True)
        # )
        
        # --- 第4層: 最終特徵提取層 ---
        # 第四個有效層
        self.final_layer = nn.Sequential(
            nn.Conv2d(ch(384), ch(512), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(512)),
            nn.ReLU(inplace=True)
        )
        
        # --- 分類頭 (不計為有效層) ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch(512), num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_effective_layers(self):
        """返回模型中的有效層數"""
        # 根據設計，我們有4個有效層
        # 1. 初始卷積層
        # 2. 並聯路徑 (卷積或注意力，算作1層)
        # 3. 融合層
        # 4. 最終層
        return 4
    
    def forward(self, x):
        # 第1層: 共享初始層
        x = self.initial_layer(x)
        
        # 第2層: 並聯路徑
        conv_features = self.conv_path(x)
        attn_features = self.attention_path(x)
        
        # 特徵融合 (通道維度上拼接)
        combined_features = torch.cat([conv_features, attn_features], dim=1)
        
        # 第3層: 融合層
        x = self.fusion_layer(combined_features)
        
        # 第4層: 最終特徵提取層
        x = self.final_layer(x)
        
        # 分類頭
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 為了兼容性，保留原始的 EfficientNet 類
class EfficientNet(nn.Module):
    """
    原始的 EfficientNet 實現，使用4個連續的卷積層
    """
    def __init__(self, num_classes=100, in_channels=3, width_multiplier=1.0):
        super(EfficientNet, self).__init__()
        logger.info(f"初始化原始 EfficientNet，4個有效卷積層，寬度乘數={width_multiplier}")
        
        ch = lambda c: max(1, int(c * width_multiplier))
        
        # Layer 1: 7x7 Conv, stride 2 -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ch(64), kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch(64)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer 2: 3x3 Conv, stride 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(ch(64), ch(128), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(128)),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: 3x3 Conv, stride 1, dilation 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(ch(128), ch(256), kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(ch(256)),
            nn.ReLU(inplace=True)
        )
        
        # Layer 4: 3x3 Conv, stride 1
        self.layer4 = nn.Sequential(
            nn.Conv2d(ch(256), ch(512), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch(512)),
            nn.ReLU(inplace=True)
        )
        
        # 分類頭
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ch(512), num_classes)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_effective_layers(self):
        return 4
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x