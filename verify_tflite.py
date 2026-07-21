import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="tflite_model/cough_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

data = np.load("features_cache.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
classes = data["classes"]

mean = X.mean()
std = X.std()
X_norm = ((X - mean) / (std + 1e-6))[..., np.newaxis].astype(np.float32)

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

correct = 0
n_test = 200

for i in range(n_test):
    sample = X_norm[i:i+1]
    sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], sample_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    predicted_class = np.argmax(output)
    if predicted_class == y[i]:
        correct += 1

print(f"Accuracy on {n_test} samples: {correct/n_test:.4f}")