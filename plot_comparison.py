import pandas as pd
import matplotlib.pyplot as plt
import os

# 設定檔案路徑
baseline_log = 'log_baseline.csv'
hybrid_log = 'log_hybrid.csv'

# 檢查檔案是否存在
if not os.path.exists(baseline_log) or not os.path.exists(hybrid_log):
    print("❌ 找不到紀錄檔！請先執行 run_hybrid_experiment.py 兩次以產生數據。")
    exit()

# 讀取數據
df_base = pd.read_csv(baseline_log)
df_hybrid = pd.read_csv(hybrid_log)

# 開始畫圖
plt.figure(figsize=(10, 6))

# 畫 Baseline 曲線 (紅色虛線)
plt.plot(df_base['Iteration'], df_base['PSNR'], 
         label='Baseline (Original C3)', color='red', linestyle='--', linewidth=2)

# 畫 Hybrid 曲線 (藍色實線)
plt.plot(df_hybrid['Iteration'], df_hybrid['PSNR'], 
         label='Hybrid (Ours: C3 + Pre-fitter)', color='blue', linewidth=2)

# 加入標題和標籤
plt.title('Training Convergence Comparison: Baseline vs. Hybrid Architecture', fontsize=14)
plt.xlabel('Training Iterations', fontsize=12)
plt.ylabel('PSNR (dB) - Higher is Better', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)

# 儲存圖片
output_img = 'comparison_plot.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 比較圖已產生：{output_img}")

# 顯示圖片 (如果在有介面的環境)
# plt.show()