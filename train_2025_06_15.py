from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
from collections import Counter

def pad_and_flatten_features(features, max_len=None):
    """
    features: list of np.array, each shape like (T_i, 17, 3)
    max_len: int or None, padding後的最大時間軸長度，None代表用目前最大長度
    """

    # 找最大長度 (若沒指定)
    if max_len is None:
        max_len = max(f.shape[0] for f in features)

    padded_features = []
    for f in features:
        T, J, C = f.shape
        if T < max_len:
            # pad zeros 到 max_len
            pad_width = ((0, max_len - T), (0, 0), (0, 0))
            f_padded = np.pad(f, pad_width=pad_width, mode='constant', constant_values=0)
        else:
            f_padded = f[:max_len]  # 如果長度超過 max_len，截斷

        # flatten 成 1維向量
        f_flat = f_padded.flatten()
        padded_features.append(f_flat)

    return np.array(padded_features)

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

def convert_feature_to_dict(feature_array):
    n_frames = feature_array.shape[0]
    pose_sequence = []
    for frame_idx in range(len(feature_array)):
        pose_sequence.append({
            "frame":frame_idx,
            "keypoints":feature_array[frame_idx]
        })
    return pose_sequence

df1 = build_feature_label_dataframe('/content/drive/MyDrive/Baseball Movies/Gerrit_Cole_SL_videos_4S/npy','/content/drive/MyDrive/Baseball Movies/data_csv/Gerrit_Cole_SL.csv')
df2 = build_feature_label_dataframe('/content/drive/MyDrive/Baseball Movies/釀酒人主場videos_4S/Milwaukee_Brewers_FF_videos_4S/npy','/content/drive/MyDrive/Baseball Movies/釀酒人主場videos_4S/Milwaukee_Brewers_FF_videos_4S/Milwaukee_Brewers_FF.csv')
df = pd.concat([df1, df2])
df['label'] = df['label'].str.contains('strike', case=False).astype(int)
df

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------- Pose Preprocessor ----------

class PosePreprocessor:
    def __init__(self, target_frames=240):
        self.target_frames = target_frames

    def normalize_skeleton(self, pose):
        pelvis = (pose[:, 11] + pose[:, 12]) / 2
        pose = pose - pelvis[:, None, :]
        ref_length = np.linalg.norm(pose[:, 0] - pose[:, 15], axis=1)
        ref_length = ref_length[:, None, None] + 1e-6
        pose = pose / ref_length
        return pose

    def temporal_crop_or_pad(self, pose):
        T = pose.shape[0]
        if T > self.target_frames:
            start = np.random.randint(0, T - self.target_frames)
            pose = pose[start:start + self.target_frames]
        elif T < self.target_frames:
            pad = np.zeros((self.target_frames - T, 17, 3))
            pose = np.concatenate([pose, pad], axis=0)
        return pose

    def __call__(self, pose):
        pose = self.normalize_skeleton(pose)
        pose = self.temporal_crop_or_pad(pose)
        return pose

# ---------- SimCLR Augmentation ----------

class PoseSimCLRAugmentation:
    def __init__(self, noise_std=0.01, flip_prob=0.5):
        self.noise_std = noise_std
        self.flip_prob = flip_prob

    def _augment(self, pose):
        noise = np.random.normal(0, self.noise_std, pose.shape)
        pose = pose + noise
        if np.random.rand() < self.flip_prob:
            pose[..., 0] *= -1
        return pose

    def __call__(self, pose):
        return self._augment(pose), self._augment(pose)

# ---------- Datasets ----------

class SimCLRDataset(Dataset):
    def __init__(self, df, preprocess, augmentation):
        self.df = df
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pose = self.preprocess(self.df.iloc[idx]['feature'])
        aug1, aug2 = self.augmentation(pose)
        return torch.tensor(aug1).float(), torch.tensor(aug2).float()

class ClassificationDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pose = self.preprocess(self.df.iloc[idx]['feature'])
        label = self.df.iloc[idx]['label']
        return torch.tensor(pose).float(), torch.tensor(label).long()

# ---------- Encoder ----------

class PoseEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, proj_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(17 * input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_size),
            nn.ReLU()
        )
        self.projector = nn.Linear(hidden_size, proj_size)  # 只線性，不加ReLU

    def forward(self, x, return_proj=True):
        x = x.permute(0, 2, 1, 3)  # [B, 3, T, 17]
        x = x.reshape(x.size(0), -1, x.size(2))  # [B, 3*17, T]
        feat = self.encoder(x)  # [B, hidden]
        if return_proj:
            z = self.projector(feat)  # [B, proj]
            return F.normalize(z, dim=1)
        else:
            return feat  # 用於分類器輸入

# ---------- SimCLR Loss ----------

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, i + batch_size] = False
            mask[i + batch_size, i] = False
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        N = 2 * batch_size
        z = torch.cat([z_i, z_j], dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        mask = self._get_correlated_mask(batch_size).to(self.device)
        negatives = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# ---------- Training ----------

def train_simclr(encoder, loader, epochs=20, device='cuda'):
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = NTXentLoss()
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0
        for x1, x2 in tqdm(loader, desc=f"SimCLR Epoch {epoch+1}"):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = encoder(x1,return_proj=True)
            z2 = encoder(x2,return_proj=True)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

def train_classifier(encoder, loader, val_loader=None, epochs=10, device='cuda'):
    encoder = encoder.to(device)
    encoder.eval()  # freeze
    classifier = nn.Linear(128, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # 計算類別權重
    label_counts = Counter(train_df['label'])
    total = sum(label_counts.values())
    class_weights = [total / label_counts[i] for i in range(len(label_counts))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_accs, val_accs = [], []

    for epoch in range(epochs):
        classifier.train()
        all_preds, all_labels = [], []
        for x, y in tqdm(loader, desc=f"Classifier Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = encoder(x,return_proj=False)
            out = classifier(feat)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        train_accs.append(acc)
        print(f"Epoch {epoch+1} Train Acc: {acc:.4f}")

        if val_loader:
            classifier.eval()
            val_preds, val_labels = [], []
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    feat = encoder(x,return_proj=False)
                    out = classifier(feat)
                    preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
            val_acc = accuracy_score(val_labels, val_preds)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")

    # Plot accuracy
    plt.plot(train_accs, label='Train Acc')
    if val_loader:
        plt.plot(val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Classification Accuracy")
    plt.show()

    # Classification report
    print("Classification Report:\n", classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

from sklearn.model_selection import train_test_split

# 假設 df 裡有 'feature' (ndarray) 與 'label'
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# ✅ 方案三：過採樣（OverSampling）或欠採樣（UnderSampling）
from sklearn.utils import resample
df_majority = train_df[train_df['label'] == 0]
df_minority = train_df[train_df['label'] == 1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
train_df = train_df_balanced

preprocess = PosePreprocessor(target_frames=240)
augment = PoseSimCLRAugmentation()

# SimCLR pretrain
simclr_dataset = SimCLRDataset(train_df, preprocess, augment)
simclr_loader = DataLoader(simclr_dataset, batch_size=32, shuffle=True, drop_last=True)

encoder = PoseEncoder()
train_simclr(encoder, simclr_loader, epochs=100)

# Linear classifier
clf_train = ClassificationDataset(train_df, preprocess)
clf_val = ClassificationDataset(val_df, preprocess)

# 使用 WeightedRandomSampler 讓 DataLoader 平衡取樣
from torch.utils.data import WeightedRandomSampler
# 計算每個樣本的 weight
labels = train_df['label'].values
class_sample_count = np.array([np.sum(labels == t) for t in np.unique(labels)])
weights = 1. / class_sample_count
sample_weights = np.array([weights[t] for t in labels])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(clf_train, batch_size=32,sampler=sampler)

val_loader = DataLoader(clf_val, batch_size=32)
train_classifier(encoder, train_loader, val_loader, epochs=10)
