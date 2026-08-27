import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Load dataset ---
csv_file = "full_A_Z_dataset.csv"
if not os.path.exists(csv_file):
    print("❌ CSV not found. Please check your path.")
    exit()

df = pd.read_csv(csv_file, header=None)
df = df[df.iloc[:, 1:].apply(lambda row: row.count() == 42, axis=1)]

X = df.iloc[:, 1:].astype('float32').values
y = df.iloc[:, 0].values

# --- Encode labels (A–M → 0–12) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# --- Improved Deep Neural Network ---
model = Sequential([
    Dense(512, activation='relu', input_shape=(42,)),
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

# --- Callbacks ---
checkpoint = ModelCheckpoint("model_group1_best.h5", monitor='val_accuracy',
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# --- Train ---
print(f"🧠 Training A-Z on {len(X_train)} samples...")
history = model.fit(X_train, y_train, epochs=80, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stop, lr_reduce])

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Accuracy on A-Z test set: {acc * 100:.2f}%")

# --- Save model and label encoder ---
model.save("model_groupAZ_final.h5")
with open("label_encoder_groupAZ.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("🎉 Improved A-Z Model + Encoder saved!")
