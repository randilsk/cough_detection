import tensorflow as tf
import numpy as np

# load the cached spectrograms computed earlier for real audio
# examples again here so the quantizer can see realistic input values
data = np.load("features_cache.npz", allow_pickle=True)
X = data["X"]
classes = data["classes"]

# same normalization as used during training - the model was trained on normalized data, so must feed normalized data again here too
mean = X.mean()
std = X.std()
X_norm = ((X - mean) / (std + 1e-6))[..., np.newaxis].astype(np.float32)

# load our best trained model from disk (the one from the early-stopping run)
model = tf.keras.models.load_model("experiments/model_v2_earlystop.keras")

# this function hands the converter a small batch of real samples (200 here)
# so it can figure out sensible min/max ranges for converting float numbers
# into int8 numbers - without this, quantization would guess badly
def representative_dataset():
    for i in range(200):
        sample = X_norm[i:i+1]  # grab one sample at a time, keep it 4D shape
        yield [sample]

# set up the converter that turns our normal float model into a tiny
# int8 version that can run on the ESP32
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# tells the converter "yes, actually quantize this, don't just copy it"
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# gives the converter our real sample data to calibrate int8 ranges
converter.representative_dataset = representative_dataset

# forces EVERY operation to use int8 math (not a mix of int8 and float32) -
# this is required for TFLite Micro on microcontrollers like the ESP32
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# actually run the conversion - this is the step that does all the work
tflite_model = converter.convert()

# save the converted model to a file on disk
with open("tflite_model/cough_model.tflite", "wb") as f:
    f.write(tflite_model)

print("saved tflite_model/cough_model.tflite")
print(f"size: {len(tflite_model) / 1024:.1f} KB")