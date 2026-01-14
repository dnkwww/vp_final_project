import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import pandas as pd
import argparse
import urllib.request
import datetime
from PIL import Image
import torchvision.transforms.functional as TF

# ==========================================
# 0. 環境與路徑設定
# ==========================================
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from my_external_model import ExternalPriorNetwork
except ImportError:
    print("❌ 找不到 my_external_model.py")
    sys.exit(1)

# ==========================================
# 1. 自動下載 Kodak 資料集
# ==========================================
def download_kodak(target_dir):
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    base_url = "https://raw.githubusercontent.com/alexandru-dinu/kodak-dataset/master/"
    existing = [f for f in os.listdir(target_dir) if f.endswith('.png')]
    if len(existing) >= 24: return
    print("⬇️ 補齊 Kodak 資料集...")
    for i in range(1, 25):
        fname = f"kodim{i:02d}.png"
        path = os.path.join(target_dir, fname)
        if not os.path.exists(path):
            try: urllib.request.urlretrieve(base_url + fname, path)
            except: pass
    print("✅ 資料集準備完成。")

# ==========================================
# 2. 核心計算函式
# ==========================================
def estimate_bpp(latent, num_pixels):
    quantized = torch.round(latent)
    unique, counts = torch.unique(quantized, return_counts=True)
    probs = counts.float() / quantized.numel()
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9))
    return (entropy * quantized.numel() / num_pixels).item()

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else (10 * torch.log10(1.0 / mse)).item()

# ==========================================
# 3. Hybrid 模型
# ==========================================
class HybridCoolChic(nn.Module):
    def __init__(self, h, w, dim_latent=32, is_baseline=False):
        super().__init__()
        self.is_baseline = is_baseline
        self.pre = ExternalPriorNetwork()
        for p in self.pre.parameters(): p.requires_grad = False
        # 全 0 初始化
        self.latent = nn.Parameter(torch.zeros(1, dim_latent, h//16, w//16))
        # 使用 Sequential 確保梯度
        self.syn = nn.Sequential(
            nn.Conv2d(dim_latent, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )
        self.up = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=False)

    def forward(self, x):
        with torch.no_grad():
            base = self.pre(x)
            if self.is_baseline: base = torch.zeros_like(base)
        return base + self.syn(self.up(self.latent))

# ==========================================
# 4. 單張訓練迴圈
# ==========================================
def run_single(lmbda, is_base, dev, path, itr):
    try: img = Image.open(path).convert('RGB')
    except: return None, None
    x = TF.to_tensor(img).unsqueeze(0).to(dev)
    h, w = x.shape[2], x.shape[3]
    ph, pw = (16-h%16)%16, (16-w%16)%16
    if ph>0 or pw>0: x = nn.functional.pad(x, (0,pw,0,ph), mode='reflection')
    
    model = HybridCoolChic(x.shape[2], x.shape[3], is_baseline=is_base).to(dev)
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練迴圈
    for _ in range(itr+1):
        opt.zero_grad()
        rec = model(x)
        loss = nn.functional.mse_loss(rec, x) + lmbda * torch.mean(model.latent**2)
        loss.backward()
        opt.step()
        
    return estimate_bpp(model.latent, x.shape[2]*x.shape[3]), compute_psnr(rec, x)

# ==========================================
# 5. 主程式
# ==========================================
def main():
    # 🔥 產生工程師標準時間戳記: YYYYMMDD_HHmmss
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_csv = f"rd_results_{timestamp}.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='debug', choices=['single', 'debug', 'full'])
    # 🔥 預設次數增加到 5000 (您可以自己改成 2000 如果想跑快點)
    parser.add_argument('--itr', type=int, default=5000) 
    # 🔥 指定檔案 (用於續傳)
    parser.add_argument('--outfile', type=str, default=None, help='指定舊檔名以續傳')
    args = parser.parse_args()
    
    # 如果沒指定 outfile，就用當下時間的新檔名
    csv_file = args.outfile if args.outfile else default_csv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== RD 實驗 ({args.mode}) | Iters: {args.itr} ===")
    print(f"📄 數據將存入: {csv_file}")
    
    # --- 智慧續傳邏輯 (Skip Logic) ---
    finished = set() # 記錄已經跑過的 (Method, Lambda, Image)
    
    # 如果檔案存在，讀取進度
    if os.path.exists(csv_file):
        print("📂 發現檔案，讀取進度中...")
        try:
            df_exist = pd.read_csv(csv_file)
            for _, r in df_exist.iterrows():
                # 記錄已完成的組合
                finished.add(f"{r['Method']}_{r['Lambda']}_{r['Image']}")
            print(f"✅ 已完成 {len(finished)} 筆任務 (將自動跳過)")
        except: 
            print("⚠️ 讀取舊檔失敗，視為新檔案")
    else:
        # 建立新檔並寫入標頭
        pd.DataFrame(columns=['Method','Lambda','Image','BPP','PSNR']).to_csv(csv_file, index=False)

    download_kodak("image")
    all_files = sorted([os.path.join("image", f) for f in os.listdir("image") if f.endswith(".png")])
    
    if args.mode == 'single': files = all_files[:1]
    elif args.mode == 'debug': files = all_files[:2]
    else: files = all_files # full

    lambdas = [0.01, 0.001, 0.0001, 0.00001]
    
    # 開始跑迴圈
    for method, is_base in [('Baseline', True), ('Hybrid', False)]:
        print(f"\n--- {method} ---")
        for l in lambdas:
            print(f"  > Lambda {l}: ", end="")
            for path in files:
                name = os.path.basename(path)
                
                # 🔥 如果已經跑過，就跳過 (Skip)
                if f"{method}_{l}_{name}" in finished:
                    print("s", end="", flush=True) # s = skip
                    continue
                
                try:
                    # 真正開始訓練
                    bpp, psnr = run_single(l, is_base, dev, path, args.itr)
                    
                    if bpp is not None:
                        # 🔥 跑完一張馬上存檔 (Append 模式)
                        pd.DataFrame([{'Method':method, 'Lambda':l, 'Image':name, 'BPP':bpp, 'PSNR':psnr}])\
                          .to_csv(csv_file, mode='a', header=False, index=False)
                        print(".", end="", flush=True)
                except KeyboardInterrupt:
                    print(f"\n⛔ 中斷! 進度已存於 {csv_file}")
                    sys.exit(0)
                except Exception as e: 
                    print("!", end="", flush=True) # ! = error
            print(" (完成)")

    print(f"\n✅ 實驗結束! 檔案: {csv_file}")
    print(f"👉 請執行畫圖: python plot_rd_curve.py --file {csv_file}")

if __name__ == "__main__":
    main()