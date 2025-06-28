import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import shutil

# ===================== CONFIGURAÇÕES =====================
pasta_base = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/LibriSpeech/dev-clean'
saida_base = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/mfcc'
pasta_teste = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/teste'

n_mfcc = 20
n_frames_pad = 130
sample_rate = 16000
random.seed(42)

qtde_teste_vitor = 10
qtde_teste_outros = 20
# =========================================================

# Pastas de saída
saida_vwb = os.path.join(saida_base, 'vwb-flac')
saida_outros = os.path.join(saida_base, 'outros')
teste_vwb = os.path.join(pasta_teste, 'vitor')
teste_outros = os.path.join(pasta_teste, 'outros')

os.makedirs(saida_vwb, exist_ok=True)
os.makedirs(saida_outros, exist_ok=True)
os.makedirs(teste_vwb, exist_ok=True)
os.makedirs(teste_outros, exist_ok=True)

audios_vwb = []
locutores_outros = {}

# 📂 Coletar áudios por locutor
for root, dirs, files in os.walk(pasta_base):
    arquivos_flac = [f for f in files if f.endswith('.flac')]
    if not arquivos_flac:
        continue

    if 'vwb-flac' in root.lower():
        for f in arquivos_flac:
            audios_vwb.append(os.path.join(root, f))
    else:
        locutor_id = os.path.relpath(root, pasta_base).split(os.sep)[0]
        if locutor_id not in locutores_outros:
            locutores_outros[locutor_id] = []
        locutores_outros[locutor_id].extend([os.path.join(root, f) for f in arquivos_flac])

# 📊 Calcular alvos de balanceamento
n_vwb = len(audios_vwb)
n_total = int(n_vwb / 0.35)
n_outros_alvo = n_total - n_vwb

# 🎲 Selecionar locutores diversos
locutores_disponiveis = list(locutores_outros.keys())
random.shuffle(locutores_disponiveis)

audios_outros_selecionados = []
for locutor in locutores_disponiveis:
    random.shuffle(locutores_outros[locutor])
    amostrados = locutores_outros[locutor][:min(6, len(locutores_outros[locutor]))]
    audios_outros_selecionados.extend(amostrados)
    if len(audios_outros_selecionados) >= n_outros_alvo:
        break

audios_outros_selecionados = audios_outros_selecionados[:n_outros_alvo]

print(f"🔢 MFCCs a gerar:")
print(f" - vwb-flac: {n_vwb} arquivos")
print(f" - outros:   {len(audios_outros_selecionados)} arquivos")
print(f" - total:    {n_vwb + len(audios_outros_selecionados)}")

def gerar_mfcc(caminho_audio, caminho_saida):
    try:
        y, sr = librosa.load(caminho_audio, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = librosa.util.fix_length(mfccs, size=n_frames_pad, axis=1)

        fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(mfccs, sr=sr, ax=ax)

        nome_base = os.path.basename(caminho_audio).replace('.flac', '.png')
        caminho_final = os.path.join(caminho_saida, nome_base)
        plt.savefig(caminho_final, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"✅ MFCC salvo: {caminho_final}")
    except Exception as e:
        print(f"❌ Erro ao processar {caminho_audio}: {e}")

# ▶️ Gerar MFCCs
for caminho in audios_vwb:
    gerar_mfcc(caminho, saida_vwb)

for caminho in audios_outros_selecionados:
    gerar_mfcc(caminho, saida_outros)

# 📂 Separar imagens para teste
imagens_vitor = [f for f in os.listdir(saida_vwb) if f.endswith('.png')]
imagens_outros = [f for f in os.listdir(saida_outros) if f.endswith('.png')]

random.shuffle(imagens_vitor)
random.shuffle(imagens_outros)

selecionados_vitor = imagens_vitor[:min(qtde_teste_vitor, len(imagens_vitor))]
selecionados_outros = imagens_outros[:min(qtde_teste_outros, len(imagens_outros))]

print(f"\n📤 Separando {len(selecionados_vitor)} imagens para teste em: {teste_vwb}")
for nome in selecionados_vitor:
    origem = os.path.join(saida_vwb, nome)
    destino = os.path.join(teste_vwb, nome)
    shutil.move(origem, destino)

print(f"📤 Separando {len(selecionados_outros)} imagens para teste em: {teste_outros}")
for nome in selecionados_outros:
    origem = os.path.join(saida_outros, nome)
    destino = os.path.join(teste_outros, nome)
    shutil.move(origem, destino)
