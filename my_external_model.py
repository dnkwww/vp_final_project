import torch
import torch.nn as nn

class ExternalPriorNetwork(nn.Module):
    """
    【幫手模型】
    模擬一個外部預訓練模型 (Pre-fitter)。
    它的工作是看一眼原圖，給出一個大概的預測。
    """
    def __init__(self):
        super().__init__()
        # 這裡不需訓練，只是一個佔位符
        self.dummy = nn.Identity()

    def forward(self, x):
        """
        輸入: [Batch, 3, H, W]
        輸出: [Batch, 3, H, W] (模糊的預測圖)
        """
        # --- 模擬 Pre-fitter ---
        # 為了模擬 "看過很多圖但只記得大概" 的特性，
        # 我們把圖縮小 8 倍再放大，製造模糊效果。
        
        h, w = x.shape[2], x.shape[3]
        
        # 1. 縮小 (遺失細節)
        small_x = nn.functional.interpolate(x, scale_factor=0.5, mode='area')
        
        # 2. 放大 (模糊預測)
        prior_prediction = nn.functional.interpolate(small_x, size=(h, w), mode='bicubic')
        
        return prior_prediction