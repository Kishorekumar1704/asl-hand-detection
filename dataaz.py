import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load merged A–Z dataset
df = pd.read_csv("full_A_Z_dataset.csv", header=None)
df = df[df.iloc[:, 1:].apply(lambda row: row.count() == 42, axis=1)]

X = df.iloc[:, 1:].astype('float32').values
y = df.iloc[:, 0].values

# Encode labels A–Z → 0–25
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Deep model
model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("model_A_Z_best.h5", monitor='val_accuracy',
                             save_best_only=True, verbose=1)

print(f"🧠 Training on {len(X_train)} samples (A–Z)...")
history = model.fit(X_train, y_train, epochs=60, batch_size=16,
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

# Save final model and encoder
model.save("model_A_Z_final.h5")
with open("label_encoder_A_Z.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and encoder saved!")

# Accuracy Graph
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
