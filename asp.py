# Cell 1: Imports and setup
import os
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmoDataset(Dataset):
    def __init__(self, data_dir, sr=16000, n_mels=64, max_len=300):
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.files = []
        self.emos = []

        for actor in sorted(os.listdir(data_dir)):
            actor_dir = os.path.join(data_dir, actor)
            if not os.path.isdir(actor_dir):
                continue
            for fname in os.listdir(actor_dir):
                if fname.endswith(".wav") or fname.endswith(".flac"):
                    emo = int(fname.split("-")[2]) - 1
                    self.files.append(os.path.join(actor_dir, fname))
                    self.emos.append(emo)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wf, orig_sr = torchaudio.load(self.files[idx])
        wf = torchaudio.transforms.Resample(orig_sr, self.sr)(wf)
        wf = wf.mean(0)

        spec = torchaudio.transforms.MelSpectrogram(self.sr, n_mels=self.n_mels)(wf)
        spec_db = torchaudio.transforms.AmplitudeToDB()(spec)

        if spec_db.shape[1] < self.max_len:
            pad_len = self.max_len - spec_db.shape[1]
            spec_db = F.pad(spec_db, (0, pad_len))
        else:
            spec_db = spec_db[:, :self.max_len]

        return spec_db.unsqueeze(0), self.emos[idx]
class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_dim=128, num_classes=8):
        super(CNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=(input_size // 4) * 32,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
data = EmoDataset("https://drive.google.com/file/d/1Bvi3zwadK4VYYIvdZLHirrZtBle5H6pc/view?usp=sharing")
n_train = int(0.8 * len(data))
n_val = len(data) - n_train
train_set, val_set = random_split(data, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

model = CNNBiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, num_epochs=15):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for ep in range(num_epochs):
        model.train()
        train_loss_sum, train_hits = 0, 0

        for mel_batch, lbl_batch in train_loader:
            mel_batch, lbl_batch = mel_batch.to(device), lbl_batch.to(device)
            opt.zero_grad()
            preds = model(mel_batch)
            loss = loss_fn(preds, lbl_batch)
            loss.backward()
            opt.step()

            train_loss_sum += loss.item()
            train_hits += (preds.argmax(1) == lbl_batch).sum().item()

        train_acc = train_hits / len(train_loader.dataset)
        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum, val_hits = 0, 0
        with torch.no_grad():
            for mel_batch, lbl_batch in val_loader:
                mel_batch, lbl_batch = mel_batch.to(device), lbl_batch.to(device)
                preds = model(mel_batch)
                loss = loss_fn(preds, lbl_batch)

                val_loss_sum += loss.item()
                val_hits += (preds.argmax(1) == lbl_batch).sum().item()

        val_acc = val_hits / len(val_loader.dataset)
        avg_val_loss = val_loss_sum / len(val_loader)

        print(f"Epoch {ep+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

train(model, num_epochs=15)

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for mel, label in val_loader:
        mel = mel.to(device)
        output = model(mel)
        preds = output.argmax(1).cpu().numpy()
        y_true.extend(label.numpy())
        y_pred.extend(preds)

print(classification_report(y_true, y_pred, target_names=[
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]))

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_dict = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_dim=128, num_classes=8):
        super(CNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=(input_size // 4) * 32,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = CNNBiLSTM().to(device)
model.load_state_dict(torch.load("emotion_classifier.pth", map_location=device, weights_only=True))
model.eval()

def predict_emotion(file_path, sample_rate=16000, n_mels=64, max_len=300):
    wf, sr = torchaudio.load(file_path)
    wf = torchaudio.transforms.Resample(sr, sample_rate)(wf)
    wf = wf.mean(dim=0)

    mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(wf)
    mel_db = T.AmplitudeToDB()(mel)

    if mel_db.shape[1] < max_len:
        mel_db = torch.nn.functional.pad(mel_db, (0, max_len - mel_db.shape[1]))
    else:
        mel_db = mel_db[:, :max_len]

    mel_input = mel_db.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(mel_input)
        pred = torch.argmax(out, dim=1).item()
    print(f"Predicted Emotion: {emotion_dict[pred]}")
    return pred

# Example usage
predict_emotion("videoplayback.wav")

import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_dict = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# model = CNNBiLSTM().to(device)
# model.load_state_dict(torch.load("emotion_classifier.pth", map_location=device))
# model.eval()

def predict_emotion(path, sample_rate=16000, n_mels=64, max_len=300):
    wf, sr = torchaudio.load(path)
    wf = torchaudio.transforms.Resample(sr, sample_rate)(wf)
    wf = wf.mean(dim=0)

    mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(wf)
    mel_db = T.AmplitudeToDB()(mel)

    if mel_db.shape[1] < max_len:
        mel_db = torch.nn.functional.pad(mel_db, (0, max_len - mel_db.shape[1]))
    else:
        mel_db = mel_db[:, :max_len]

    x = mel_db.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    print(f"Predicted Emotion: {emotion_dict[pred]}")
    return pred, wf, sample_rate

def nlms(x, d, mu=0.05, order=32, eps=1e-6):
    n = len(x)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(order)

    for i in range(order, n):
        x_vec = x[i-order:i][::-1]
        y[i] = np.dot(w, x_vec)
        e[i] = d[i] - y[i]
        norm = np.dot(x_vec, x_vec) + eps
        w += (mu / norm) * e[i] * x_vec

    return y

def enhance(wave, sr, emo_id):
    sig = wave.numpy().squeeze()
    
    emo_map = {
        0: "neutral", 1: "calm", 2: "happy", 3: "sad",
        4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
    }
    emo = emo_map.get(emo_id, "neutral")

    zcr = librosa.feature.zero_crossing_rate(sig)[0]
    eng = np.square(sig)
    cent = librosa.feature.spectral_centroid(y=sig, sr=sr)[0]

    zcr_n = zcr / np.max(zcr)
    eng_n = eng / np.max(eng)
    cent_n = cent / np.max(cent)

    if emo == "sad":
        slow = librosa.effects.time_stretch(sig, rate=0.6)
        out = librosa.effects.pitch_shift(slow, sr=sr, n_steps=-3)

    elif emo == "happy":
        t = np.linspace(0, len(sig) / sr, len(sig))
        mod = 0.3 * np.sin(2 * np.pi * 6 * t)
        vib = sig * (1 + mod)
        out = librosa.effects.pitch_shift(vib, sr=sr, n_steps=1)

    elif emo == "angry":
        out = librosa.effects.pitch_shift(sig, sr=sr, n_steps=6)

    elif emo == "fearful":
        trem = 0.05 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, len(sig)))
        mod = sig * (1 + trem)
        out = librosa.effects.pitch_shift(mod, sr=sr, n_steps=1)

    elif emo == "calm":
        down = librosa.effects.pitch_shift(sig, sr=sr, n_steps=-1)
        filt = np.convolve(down, np.ones(10) / 10, mode='same')
        out = filt

    elif emo == "surprised":
        sig_mod = librosa.effects.pitch_shift(sig, sr=sr, n_steps=0)
        fast = librosa.effects.time_stretch(sig_mod, rate=1.15)
        out = fast[:len(sig)] if len(fast) > len(sig) else np.pad(fast, (0, len(sig) - len(fast)), mode='reflect')

    elif emo == "disgust":
        slow = librosa.effects.time_stretch(sig, rate=0.85)
        deep = librosa.effects.pitch_shift(slow, sr=sr, n_steps=-1)
        out = deep[:len(sig)] if len(deep) > len(sig) else np.pad(deep, (0, len(sig) - len(deep)), mode='reflect')

    else:
        out = sig

    enh = nlms(sig, out, mu=1)
    
    return out, enh


# Replace with your test file
path = "RAVDESS_Audio/Actor_05/03-01-02-01-02-01-05.wav"
pred, wf, sr = predict_emotion(path)
enh, enh_nlms = enhance(wf, sr, pred)

plt.figure(figsize=(14, 5))
plt.plot(wf.cpu().numpy()[:2000], label="Original", alpha=0.7)
plt.plot(enh[:2000], label="Enhanced", alpha=0.7)
plt.plot(enh_nlms[:2000], label="NLMS Enhanced", alpha=0.7)
plt.title(f"Emotion: {emotion_dict[pred]}")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

sf.write("enhanced_normal.wav", enh, sr)
sf.write("enhanced_nlms.wav", enh_nlms, sr)

print("Original:")
display(Audio(wf.cpu().numpy(), rate=sr))
print("Enhanced:")
display(Audio(enh, rate=sr))
print("NLMS Enhanced:")
display(Audio(enh_nlms, rate=sr))