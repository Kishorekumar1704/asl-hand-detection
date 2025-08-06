import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# Load dataset
csv_file = "group2_N_Z_dataset.csv"
if not os.path.exists(csv_file):
    print("❌ CSV not found.")
    exit()

df = pd.read_csv(csv_file, header=None)
df = df[df.iloc[:, 1:].apply(lambda row: row.count() == 42, axis=1)]

X = df.iloc[:, 1:].astype('float32').values
y = df.iloc[:, 0].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# 🧠 Deep Neural Network
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

checkpoint = ModelCheckpoint("model_group2_best.h5", monitor='val_accuracy',
                             save_best_only=True, verbose=1)

print(f"🧠 Training N–Z on {len(X_train)} samples...")
history = model.fit(X_train, y_train, epochs=60, batch_size=16,
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Accuracy on N–Z test set: {acc * 100:.2f}%")

model.save("model_group2_final.h5")
with open("label_encoder_group2.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("🎉 Deep N–Z Model + Encoder saved!")
