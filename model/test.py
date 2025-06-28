import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ======== CONFIGURAÇÕES ========
img_dir = r'/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/teste'
modelo_path = '/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/vwb-tests/modelo_identificador_vitor.h5'
img_height, img_width = 256, 256
# ===============================

model = load_model(modelo_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

acertos = 0
erros = 0
total = 0

# Percorre todas as subpastas
for root, _, files in os.walk(img_dir):
    for arquivo in files:
        if not arquivo.lower().endswith('.png'):
            continue

        img_path = os.path.join(root, arquivo)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.reshape((1, img_height, img_width, 1)).astype("float32") / 255.0

        pred = model.predict(img, verbose=0)[0][0]
        rel_path = os.path.relpath(img_path, img_dir)
        subpasta = rel_path.split(os.sep)[0]

        if pred >= 0.5:
            if subpasta == 'vitor':
                acertos += 1
            else:
                erros += 1
            print(f'{arquivo}: ✅ É o Vitor! (confiança: {pred:.2f})')
        else:
            if subpasta == 'outros':
                acertos += 1
            else:
                erros += 1
            print(f'{arquivo}: ❌ Não é o Vitor. (confiança: {1 - pred:.2f})')

        total += 1

# Resultados finais
print('\n======= RESULTADOS =======')
print(f'Total de imagens: {total}')
print(f'Acertos: {acertos} ({(acertos / total) * 100:.2f}%)')
print(f'Erros: {erros} ({(erros / total) * 100:.2f}%)')
