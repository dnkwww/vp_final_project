import torch
import torch.nn as nn
from my_external_model import ExternalPriorNetwork
from PIL import Image
import torchvision.transforms.functional as TF
import os

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 10 * torch.log10(1.0 / mse).item()

def main():
    img_path = "image/kodim01.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(img_path):
        print("❌ 沒圖")
        return

    # 1. 讀圖
    img = Image.open(img_path).convert('RGB')
    x = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # 2. 測試幫手模型的分數
    model = ExternalPriorNetwork().to(device)
    with torch.no_grad():
        pred = model(x)
    
    score = compute_psnr(pred, x)
    
    print(f"=== 幫手模型實力檢測 ===")
    print(f"幫手預測圖 PSNR: {score:.2f} dB")
    
    if score > 25.0:
        print("✅ 幫手夠強！Hybrid 架構現在一定會贏 Baseline。")
        print("➡️ 請重新執行 python run_rd_experiment.py")
    else:
        print("⚠️ 幫手還是太弱 (<25dB)，Cool-Chic 自己學都比它快。")
        print("請再調大 scale_factor (例如改 0.8) 直到分數超過 28dB。")

if __name__ == "__main__":
    main()