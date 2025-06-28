import soundfile as sf
import os

# ======= CONFIGURAÇÃO =======
caminho_wav = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/temp.wav'  # Altere aqui
caminho_flac = caminho_wav.replace('.wav', '.flac')
# ============================

# 🎧 Carregar e salvar
data, samplerate = sf.read(caminho_wav)
sf.write(caminho_flac, data, samplerate, format='FLAC')

print(f"✅ Arquivo convertido para: {caminho_flac}")
