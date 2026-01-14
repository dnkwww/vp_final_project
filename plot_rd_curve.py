import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='畫出 RD 曲線')
    parser.add_argument('--file', type=str, required=True, help='輸入 CSV 檔案路徑')
    args = parser.parse_args()

    csv_file = args.file

    if not os.path.exists(csv_file):
        print(f"❌ 找不到檔案: {csv_file}")
        sys.exit(1)

    # 自動決定輸出圖片檔名 (把 csv 換成 png, results 換成 curve)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_png = base_name.replace("results", "curve") + ".png"

    print(f"📊 正在讀取 {csv_file} ...")
    df = pd.read_csv(csv_file)

    # 🔥 關鍵：計算平均值
    # 因為原始數據是每張圖一筆，我們要對全資料集取平均
    df_avg = df.groupby(['Method', 'Lambda'])[['BPP', 'PSNR']].mean().reset_index()

    baseline = df_avg[df_avg['Method'] == 'Baseline'].sort_values(by='BPP')
    hybrid = df_avg[df_avg['Method'] == 'Hybrid'].sort_values(by='BPP')
    
    # print出BPP, PSNR
    print("\n🔴 Baseline (Original) RD points:")
    for _, row in baseline.iterrows():
        print(f"(BPP, PSNR) = ({row['BPP']:.6f}, {row['PSNR']:.6f})")

    print("\n🔵 Hybrid (Ours) RD points:")
    for _, row in hybrid.iterrows():
        print(f"(BPP, PSNR) = ({row['BPP']:.6f}, {row['PSNR']:.6f})")

    plt.figure(figsize=(10, 6))

    # 畫 Baseline (紅線)
    plt.plot(baseline['BPP'], baseline['PSNR'], 'o--', color='#D32F2F', label='Baseline (Original)', linewidth=2, markersize=8)

    # 畫 Hybrid (藍線)
    plt.plot(hybrid['BPP'], hybrid['PSNR'], 's-', color='#1976D2', label='Hybrid (Ours)', linewidth=2, markersize=8)

    plt.title(f'Rate-Distortion Performance\nSource: {csv_file}', fontsize=14)
    plt.xlabel('Bitrate (bpp) - Lower is Better', fontsize=12)
    plt.ylabel('PSNR (dB) - Higher is Better', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✅ 圖片已生成: {output_png}")

if __name__ == "__main__":
    main()