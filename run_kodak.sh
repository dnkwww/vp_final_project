#!/bin/bash

# 設定路徑 (請依據您的實際路徑微調)
INPUT_DIR="image"
OUTPUT_DIR="results_kodak"
LOG_DIR="logs_kodak"

# 建立輸出資料夾
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 啟動 Conda 環境 (如果在 VS Code 終端機跑，通常不需要這行，但保險起見)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate c3yes

echo "=== 開始執行 Kodak 24 張圖片批次編碼 ==="

# 迴圈讀取所有 png 圖片
for img in "$INPUT_DIR"/*.png; do
    # 取得檔名 (例如 kodim01)
    filename=$(basename -- "$img")
    filename_no_ext="${filename%.*}"
    
    echo "------------------------------------------------"
    echo "正在處理: $filename ..."
    
    # 執行 Cool-Chic 編碼
    # 注意：這裡使用了您的參數，並將 Log 存到個別檔案
    python -m coolchic.encode \
        -i "$img" \
        -o "$OUTPUT_DIR/${filename_no_ext}.bin" \
        --n_itr 100 \
        --workdir "$LOG_DIR/${filename_no_ext}" \
        > "$LOG_DIR/${filename_no_ext}.log" 2>&1

    echo "完成: $filename (Log 已存於 $LOG_DIR/${filename_no_ext}.log)"
done

echo "=== 全部完成！ ==="