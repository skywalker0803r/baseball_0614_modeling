import os
import numpy as np
import pandas as pd

def build_feature_label_dataframe(npy_folder, csv_path):
    # 讀取描述的 CSV
    df_csv = pd.read_csv(csv_path)

    data = []

    # 遍歷每個 .npy 檔案
    for file in os.listdir(npy_folder):
        if file.endswith('.npy'):
            base_name = os.path.splitext(file)[0]  # 去掉 .npy
            video_filename = base_name + '.mp4'    # 在 CSV 中查找

            # 找出對應的 label
            match = df_csv[df_csv['Filename'] == video_filename]

            if not match.empty:
                label = match.iloc[0]['description']
                feature_path = os.path.join(npy_folder, file)
                feature = np.load(feature_path)

                data.append({
                    'filename': file,
                    'feature': feature,
                    'label': label
                })
            else:
                print(f"⚠️ 無法在 CSV 中找到 {video_filename}")

    # 組成 DataFrame
    df = pd.DataFrame(data)
    return df
