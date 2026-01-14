import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import argparse
from PIL import Image
import torchvision.transforms.functional as TF

# ==========================================
# 0. 環境設定
# ==========================================
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from my_external_model import ExternalPriorNetwork
except ImportError:
    print("❌ 找不到 my_external_model.py")
    sys.exit(1)

# ==========================================
# 1. 定義架構 (可切換模式)
# ==========================================
class HybridCoolChic(nn.Module):
    def __init__(self, img_h, img_w, dim_latent=32, is_baseline=False):
        super().__init__()
        self.is_baseline = is_baseline
        
        # 引擎 A: 外部幫手
        self.pre_fitter = ExternalPriorNetwork()
        for p in self.pre_fitter.parameters(): p.requires_grad = False
            
        # 引擎 B: Cool-Chic (Latent + 手動 Synthesis)
        self.latent_grid = nn.Parameter(torch.randn(1, dim_latent, img_h // 16, img_w // 16))
        self.synthesis = nn.Sequential(
            nn.Conv2d(dim_latent, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1) 
        )
        self.upsample = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=False)

    def forward(self, x_input):
        # 1. 幫手預測
        with torch.no_grad():
            base_prediction = self.pre_fitter(x_input)
            # 【關鍵】如果是 Baseline 模式，強制把幫手的貢獻歸零
            if self.is_baseline:
                base_prediction = torch.zeros_like(base_prediction)
            
        # 2. Cool-Chic 殘差
        features = self.upsample(self.latent_grid)
        residual = self.synthesis(features)
        
        # 3. 合體
        final_image = base_prediction + residual
        return final_image

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * torch.log10(1.0 / mse)

# ==========================================
# 2. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='執行 Baseline 模式 (停用幫手)')
    parser.add_argument('--logfile', type=str, required=True, help='輸出紀錄檔名稱')
    args = parser.parse_args()

    img_path = "image/kodim01.png"
    n_itr = 1000
    lr = 0.01
    
    mode_name = "Baseline (舊架構)" if args.baseline else "Hybrid (新架構)"
    print(f"=== 啟動實驗: {mode_name} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 讀圖
    img_pil = Image.open(img_path).convert('RGB')
    target_img = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    H, W = target_img.shape[2], target_img.shape[3]
    pad_h, pad_w = (16 - H % 16) % 16, (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        target_img = nn.functional.pad(target_img, (0, pad_w, 0, pad_h), mode='reflection')

    # 建立模型 (傳入是否為 baseline)
    model = HybridCoolChic(target_img.shape[2], target_img.shape[3], is_baseline=args.baseline).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 開啟紀錄檔
    with open(args.logfile, 'w') as f:
        f.write("Iteration,PSNR\n") # 寫入標頭
        
        print(f"開始訓練 ({n_itr} 次)...")
        for i in range(n_itr + 1):
            optimizer.zero_grad()
            recon_img = model(target_img)
            loss = nn.functional.mse_loss(recon_img, target_img)
            loss.backward()
            optimizer.step()
            
            # 每 50 次紀錄一筆資料
            if i % 50 == 0:
                psnr = compute_psnr(recon_img, target_img)
                print(f"Iter {i:4d} | PSNR: {psnr:.2f} dB")
                f.write(f"{i},{psnr.item():.4f}\n") # 寫入檔案

    print(f"=== 實驗結束，紀錄已儲存至 {args.logfile} ===")

if __name__ == "__main__":
    main()