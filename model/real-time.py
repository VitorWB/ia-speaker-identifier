import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from tqdm import tqdm

# ========== CONFIGURAÇÕES ==========
base_dir = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/LibriSpeech/dev-clean'
temp_img = 'temp.png'
sample_rate = 16000
duration = 5  # segundos de gravação
n_mfcc = 20
n_frames_pad = 130
img_height, img_width = 256, 256
modelo_path = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/modelo_identificador_vitor.h5'
silence_threshold = 0.01
# ===================================

# 🧑 Perguntar quem está falando
resposta = input("👤 É o Vitor falando? (s/n): ").strip().lower()
if resposta == 's':
    speaker_name = 'vwb-flac'
else:
    speaker_name = input("📝 Digite o nome da pessoa que está falando (ex: maria, jose): ").strip()
    speaker_name = speaker_name.replace(' ', '_').lower()

# Criar diretório se não existir
speaker_dir = os.path.join(base_dir, speaker_name)
os.makedirs(speaker_dir, exist_ok=True)

# 🎙️ Nome do arquivo
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
flac_filename = f"{speaker_name}_{timestamp}.flac"
flac_path = os.path.join(speaker_dir, flac_filename)

# 🎙️ Gravar áudio com contador
print("🎙️ Gravando... fale agora.")
recording = np.zeros((int(duration * sample_rate), 1), dtype='float32')
stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')

with stream:
    for i in tqdm(range(duration), desc="⏳ Tempo restante"):
        frames = stream.read(sample_rate)[0]
        recording[i * sample_rate:(i + 1) * sample_rate] = frames
        rms = np.sqrt(np.mean(frames**2))
        if rms > silence_threshold:
            print("🎧 Detecção de som...")
        else:
            print("🤫 Silêncio detectado.")

# 💾 Salvar como FLAC
sf.write(flac_path, recording, sample_rate, format='FLAC')
print(f"✅ Áudio salvo como: {flac_path}")

# 🎧 Processar o áudio para gerar imagem MFCC
y, sr = librosa.load(flac_path, sr=sample_rate)
y, _ = librosa.effects.trim(y, top_db=20)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
mfccs = librosa.util.fix_length(mfccs, size=n_frames_pad, axis=1)

# 🖼️ Gerar imagem MFCC
fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
librosa.display.specshow(mfccs, sr=sr, ax=ax)
plt.savefig(temp_img, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"📷 Imagem MFCC salva como: {temp_img}")

# 📷 Preparar imagem para o modelo
img = cv2.imread(temp_img, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img_width, img_height))
img = img.reshape((1, img_height, img_width, 1)).astype("float32") / 255.0

# 🤖 Classificação com modelo
model = load_model(modelo_path)
pred = model.predict(img, verbose=0)[0][0]

# 📊 Mostrar resultado
if pred >= 0.5:
    print(f"✅ Modelo diz: É o Vitor (confiança: {pred:.2f})")
else:
    print(f"❌ Modelo diz: Não é o Vitor (confiança: {1 - pred:.2f})")
