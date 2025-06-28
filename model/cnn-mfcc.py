import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ===================== CONFIGURAÇÕES =====================
base_dir = r'/Users/vitorwolffbordignon/Documents/projetos/speaker-identifier/mfcc'
img_height = 256
img_width = 256
modelo_path = 'modelo_identificador_vitor.h5'
nome_vitor = "vwb-flac"  # Nome da pasta com as imagens do Vitor
# =========================================================

X = []
y = []

# Loop pelas subpastas
for nome_pasta in os.listdir(base_dir):
    pasta_completa = os.path.join(base_dir, nome_pasta)

    if not os.path.isdir(pasta_completa):
        continue

    rotulo = 1 if nome_pasta == nome_vitor else 0

    for nome_arquivo in os.listdir(pasta_completa):
        if nome_arquivo.endswith(".png"):
            caminho_imagem = os.path.join(pasta_completa, nome_arquivo)
            img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            img = img.reshape((img_height, img_width, 1))

            X.append(img)
            y.append(rotulo)

# Conversão para numpy arrays
X = np.array(X).astype("float32") / 255.0
y = np.array(y).astype("float32")  # binário

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== MODELO =====================
if os.path.exists(modelo_path):
    print("🔁 Carregando modelo existente...")
    model = load_model(modelo_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    print("🆕 Criando novo modelo...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===================== TREINAMENTO =====================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stop])

# ===================== AVALIAÇÃO =====================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Acurácia no teste: {test_acc:.2f}")

# ===================== SALVAR MODELO =====================
model.save(modelo_path)
print(f"💾 Modelo salvo em: {modelo_path}")

# ===================== PREVISÕES =====================
predictions = model.predict(X_test).flatten()  # mantém como vetor 1D

# Binariza as previsões
y_pred_bin = (predictions >= 0.5).astype(int)
y_test_flat = y_test.flatten()

# ===================== RELATÓRIO =====================
print("=== Classification Report ===")
print(classification_report(y_test_flat, y_pred_bin, target_names=["Outro", "Vitor"]))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test_flat, y_pred_bin)
print(cm)

print("Valores únicos em y_test:", np.unique(y_test_flat))
print("Valores únicos em predictions binárias:", np.unique(y_pred_bin))

# ===================== CURVA ROC =====================
fpr, tpr, thresholds = roc_curve(y_test_flat, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()
