
import os
import io
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import uuid
import base64
import traceback
# Configure Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable cross-origin requests for API

# Create upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN-BiLSTM model definition
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

# Load the model
model = CNNBiLSTM().to(device)
try:
    # Add weights_only=True parameter to match training notebook's loading method
    model.load_state_dict(torch.load("emotion_classifier.pth", map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using untrained model for demonstration purposes.")

# Emotion mapping
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
def predict_emotion(audio_path, sample_rate=16000, n_mels=64, max_len=300):
    """Predict emotion from audio file"""
    try:
        # Try loading the audio file
        try:
            wf, sr = torchaudio.load(audio_path)
        except Exception as audio_load_error:
            print(f"torchaudio failed to load: {audio_load_error}")
            # Fallback to librosa
            try:
                sig, sr = librosa.load(audio_path, sr=sample_rate)
                wf = torch.FloatTensor(sig).unsqueeze(0)
            except Exception as librosa_error:
                print(f"librosa also failed to load: {librosa_error}")
                # Create a default waveform if both loading methods fail
                wf = torch.zeros(1, sample_rate)  # 1 second of silence
                sr = sample_rate

        # Resample if needed
        wf = torchaudio.transforms.Resample(sr, sample_rate)(wf)
        wf = wf.mean(dim=0)

        mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(wf)
        mel_db = T.AmplitudeToDB()(mel)

        if mel_db.shape[1] < max_len:
            mel_db = F.pad(mel_db, (0, max_len - mel_db.shape[1]))
        else:
            mel_db = mel_db[:, :max_len]

        x = mel_db.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out, dim=1).item()

        return pred, wf, sample_rate
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        # Create a default waveform and return neutral as fallback
        default_wf = torch.zeros(sample_rate)  # 1 second of silence
        return 0, default_wf, sample_rate  # Return neutral as fallback
def nlms(x, d, mu=1.0, order=32, eps=1e-6):
    """Normalized Least Mean Square filter with updated mu parameter to match training notebook"""
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
    """Enhance audio based on detected emotion"""
    sig = wave.numpy().squeeze()
    
    emo = emotion_dict.get(emo_id, "neutral")

    zcr = librosa.feature.zero_crossing_rate(sig)[0]
    eng = np.square(sig)
    cent = librosa.feature.spectral_centroid(y=sig, sr=sr)[0]

    # Add null checks to avoid division by zero
    zcr_n = zcr / np.max(zcr) if np.max(zcr) > 0 else zcr
    eng_n = eng / np.max(eng) if np.max(eng) > 0 else eng
    cent_n = cent / np.max(cent) if np.max(cent) > 0 else cent

    if emo == "sad":
        # Update parameters to match training notebook
        slow = librosa.effects.time_stretch(sig, rate=0.6)
        out = librosa.effects.pitch_shift(slow, sr=sr, n_steps=-3)
        out = out[:len(sig)] if len(out) > len(sig) else np.pad(out, (0, len(sig) - len(out)), mode='reflect')

    elif emo == "happy":
        t = np.linspace(0, len(sig) / sr, len(sig))
        # Update modulation depth to match training
        mod = 0.3 * np.sin(2 * np.pi * 6 * t)
        vib = sig * (1 + mod)
        out = librosa.effects.pitch_shift(vib, sr=sr, n_steps=1)
        out = out[:len(sig)] if len(out) > len(sig) else np.pad(out, (0, len(sig) - len(out)), mode='reflect')

    elif emo == "angry":
        # Update pitch shift to match training
        out = librosa.effects.pitch_shift(sig, sr=sr, n_steps=6)
        out = out[:len(sig)] if len(out) > len(sig) else np.pad(out, (0, len(sig) - len(out)), mode='reflect')

    elif emo == "fearful":
        trem = 0.05 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, len(sig)))
        mod = sig * (1 + trem)
        out = librosa.effects.pitch_shift(mod, sr=sr, n_steps=1)
        out = out[:len(sig)] if len(out) > len(sig) else np.pad(out, (0, len(sig) - len(out)), mode='reflect')

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

    # Update NLMS parameter to match training
    enh_nlms = nlms(sig, out, mu=1.0)
    
    return out, enh_nlms


# Define routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """Endpoint to process uploaded audio files"""
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    # Create unique filenames
    file_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_original.wav")
    enhanced_path = os.path.join(PROCESSED_FOLDER, f"{file_id}_enhanced.wav")
    nlms_path = os.path.join(PROCESSED_FOLDER, f"{file_id}_nlms.wav")
    
    try:
        # Log the file being saved
        print(f"Saving file to {original_path}")
        file.save(original_path)
        
        # Process the audio
        print("Predicting emotion...")
        emo_id, wf, sr = predict_emotion(original_path)
        print(f"Emotion detected: {emotion_dict[emo_id]} (ID: {emo_id})")
        try:
            print("Enhancing audio...")
            enhanced, nlms_enhanced = enhance(wf, sr, emo_id)
        except Exception as e:
        
            print(f"Error processing audio: {e}")
            print(traceback.format_exc())  # Print full traceback
            return jsonify({'error': str(e)}), 500
        
        # Save processed files
        print(f"Saving enhanced audio to {enhanced_path}")
        sf.write(enhanced_path, enhanced, sr)
        print(f"Saving NLMS audio to {nlms_path}")
        sf.write(nlms_path, nlms_enhanced, sr)
        
        # Return results
        return jsonify({
            'emotion': emotion_dict[emo_id],
            'originalAudio': f"/api/audio/{file_id}_original.wav",
            'enhancedAudio': f"/api/audio/{file_id}_enhanced.wav",
            'nlmsAudio': f"/api/audio/{file_id}_nlms.wav"
        })
    except Exception as e:
        import traceback
        print(f"Error processing audio: {e}")
        print(traceback.format_exc())  # Print full traceback
        return jsonify({'error': str(e)}), 500


@app.route('/api/record-audio', methods=['POST'])
def record_audio():
    """Endpoint to process recorded audio data"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data'}), 400
    
    file = request.files['audio']
    
    # Create unique filenames
    file_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_original.wav")
    enhanced_path = os.path.join(PROCESSED_FOLDER, f"{file_id}_enhanced.wav")
    nlms_path = os.path.join(PROCESSED_FOLDER, f"{file_id}_nlms.wav")
    
    try:
        # Save the recorded file
        file.save(original_path)
        
        # Process the audio
        emo_id, wf, sr = predict_emotion(original_path)
        enhanced, nlms_enhanced = enhance(wf, sr, emo_id)
        
        # Save processed files
        sf.write(enhanced_path, enhanced, sr)
        sf.write(nlms_path, nlms_enhanced, sr)
        
        # Return results
        return jsonify({
            'emotion': emotion_dict[emo_id],
            'originalAudio': f"/api/audio/{file_id}_original.wav",
            'enhancedAudio': f"/api/audio/{file_id}_enhanced.wav",
            'nlmsAudio': f"/api/audio/{file_id}_nlms.wav"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/<filename>')
def get_audio(filename):
    """Endpoint to retrieve processed audio files"""
    # Fix: Check the filename pattern properly to determine the directory
    if "_original.wav" in filename:
        return send_from_directory(UPLOAD_FOLDER, filename)
    else:
        return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    # Set debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader = False)
