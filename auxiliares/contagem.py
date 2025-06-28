import os

# Caminho base
base_dir = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/mfcc'
nome_vitor = 'vwb-flac'

# Contadores
total_vitor = 0
total_outros = 0

# Loop pelas subpastas
for nome_pasta in os.listdir(base_dir):
    pasta_completa = os.path.join(base_dir, nome_pasta)
    if not os.path.isdir(pasta_completa):
        continue

    imagens = [f for f in os.listdir(pasta_completa) if f.endswith('.png')]
    count = len(imagens)

    if nome_pasta == nome_vitor:
        total_vitor += count
    else:
        total_outros += count

# Cálculo de percentuais
total_geral = total_vitor + total_outros
percent_vitor = (total_vitor / total_geral) * 100 if total_geral else 0
percent_outros = (total_outros / total_geral) * 100 if total_geral else 0

# Resultado final
print(f"MFCC Vitor: {total_vitor} ({percent_vitor:.2f}%)")
print(f"MFCC Outros: {total_outros} ({percent_outros:.2f}%)")
print(f"Total de MFCCs: {total_geral}")
