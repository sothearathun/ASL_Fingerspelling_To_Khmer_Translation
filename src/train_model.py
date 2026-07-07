import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from utils import load_data

X, y, class_names = load_data('data')
y_categorical = to_categorical(y)

# Three-way stratified split: the test set is only ever touched once, at the
# final evaluation below, so it stays a clean held-out estimate. (Feeding the
# test set into training as validation_data, as a single train/test split
# would, lets it silently influence early stopping decisions.)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

model.save('asl_model.h5')
np.save('class_names.npy', class_names)

print("Model trained and saved!")

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report (held-out test set):")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_true, y_pred)
print("        " + " ".join(f"{name:>5}" for name in class_names))
for name, row in zip(class_names, cm):
    print(f"{name:>8} " + " ".join(f"{v:>5}" for v in row))

accuracy = accuracy_score(y_true, y_pred)
with open('model_accuracy.json', 'w') as f:
    json.dump({'accuracy': float(accuracy)}, f)
