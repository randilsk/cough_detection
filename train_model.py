import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.load("features_cache.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
classes = data["classes"]

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# normalize pixel values - helps training stability
mean = X.mean()
std = X.std()
X = (X - mean) / (std + 1e-6)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))
print("class weights:", class_weight_dict)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation="softmax"),
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=8, restore_best_weights=True
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=classes)
print(report)
print(confusion_matrix(y_test, y_pred_classes))

# --- Save this version, under a new name ---
import os
os.makedirs("experiments", exist_ok=True)

VERSION = "v2_earlystop"  # change this each time you try something new

model.save(f"experiments/model_{VERSION}.keras")

with open("experiments/results_log.txt", "a") as f:
    f.write(f"=== {VERSION} ===\n")
    f.write(f"test_accuracy: {test_acc:.4f}\n")
    f.write(f"test_loss: {test_loss:.4f}\n")
    f.write(report)
    f.write("\n\n")

print(f"saved experiments/model_{VERSION}.keras")

# --- Training curves ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over training")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over training")
plt.legend()

plt.tight_layout()
plt.savefig(f"experiments/training_curves_{VERSION}.png")
print(f"saved experiments/training_curves_{VERSION}.png")

# --- Confusion matrix heatmap ---
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion Matrix - {VERSION}")

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig(f"experiments/confusion_matrix_{VERSION}.png")
print(f"saved experiments/confusion_matrix_{VERSION}.png")