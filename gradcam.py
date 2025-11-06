import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gradcam(model, img_array, layer_name=None):
    if not layer_name:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Failed to compute gradients. Check model and layer.")

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8
    return cam


def save_gradcam(cam, original_img_path, save_path):
    original_img_path = str(original_img_path)

    # Load the original image using OpenCV
    img = cv2.imread(original_img_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {original_img_path}")

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite(save_path, np.uint8(superimposed_img))
