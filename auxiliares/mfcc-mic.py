import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import numpy as np
import time

# ========== CONFIGURAÇÕES ==========
base_audio_dir = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/LibriSpeech/dev-clean'
base_mfcc_dir = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/mfcc'

sample_rate = 16000
duration = 30  # segundos por gravação
n_mfcc = 20
n_frames_pad = 130
silence_threshold = 0.01  # Limite para considerar como silêncio
# ===================================

# 🧑 Identificação do locutor
resposta = input("👤 É o Vitor falando? (s/n): ").strip().lower()
if resposta == 's':
    speaker_name = 'vwb-flac'
else:
    speaker_name = input("📝 Digite o nome da pessoa que está falando (ex: maria, jose): ").strip()
    speaker_name = speaker_name.replace(' ', '_').lower()

# Criar diretórios de saída
audio_output_dir = os.path.join(base_audio_dir, speaker_name)
mfcc_output_dir = os.path.join(base_mfcc_dir, 'outros')
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(mfcc_output_dir, exist_ok=True)

print(f"\n📂 Salvando áudios em: {audio_output_dir}")
print(f"📂 Salvando MFCCs em: {mfcc_output_dir}")
print("🎙️ Iniciando gravação contínua. Pressione Ctrl+C para parar.")

try:
    i = 1
    while True:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"{speaker_name}_{timestamp}.flac"
        mfcc_filename = f"mfcc_{speaker_name}_{timestamp}.png"

        audio_path = os.path.join(audio_output_dir, audio_filename)
        mfcc_path = os.path.join(mfcc_output_dir, mfcc_filename)

        print(f"\n🎙️ Gravando segmento {i} ({duration} segundos)...")

        recording = np.zeros((duration * sample_rate, 1), dtype='float32')
        stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32')
        stream.start()

        for sec in range(duration):
            frames, _ = stream.read(sample_rate)
            start_idx = sec * sample_rate
            recording[start_idx:start_idx + sample_rate] = frames

            # Detecção de som via RMS
            rms = np.sqrt(np.mean(frames**2))
            status = "🎧 Som detectado" if rms > silence_threshold else "🤫 Silêncio"

            print(f"⏱️ {sec+1}/{duration} segundos - {status}")

        stream.stop()
        sf.write(audio_path, recording, sample_rate, format='FLAC')
        print(f"✅ Áudio salvo: {audio_path}")

        # Processar MFCC
        y, sr = librosa.load(audio_path, sr=sample_rate)
        y, _ = librosa.effects.trim(y, top_db=20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = librosa.util.fix_length(mfccs, size=n_frames_pad, axis=1)

        # Gerar e salvar imagem MFCC
        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(mfccs, sr=sr, ax=ax)
        plt.savefig(mfcc_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"📷 MFCC salvo: {mfcc_path}")
        i += 1

except KeyboardInterrupt:
    print("\n🛑 Gravação interrompida pelo usuário.")
