import torch
from torchvision import models
import numpy as np
import tensorflow as tf

from models import TFModel, TorchModel
from utilities import ModelType, DATA_ROOT, SELECTED, MOBILE, Backend, get_image_size, progress

from torchsummary import summary
from tensorflow.keras.applications import mobilenet_v2

y_val = np.load(DATA_ROOT + SELECTED + 'y_val.npy')
labels = np.load(DATA_ROOT + 'labels.npy')
x_val = np.load(DATA_ROOT + SELECTED + 'x_val_' + str(get_image_size()) + '.npy')
y_val = np.argsort(np.argsort(labels[:, 0]))[y_val]


def test_tf():
    interpreter = tf.lite.Interpreter('image_net/tf/mobilenet_v2/mobilenet_v2.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    b = np.zeros(3000, dtype=np.bool)
    for i in progress(3000):
        input_data = tf.keras.applications.mobilenet_v2.preprocess_input(x_val[i:i + 1]).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        b[i] = interpreter.get_tensor(output_details[0]['index'])[0].argmax() == y_val[i]

    print(b.sum())


def test_torch():
    jit_model = torch.jit.load('image_net/torch/mobilenet_v2/mobilenet_v2.pt')
    model = TorchModel(ModelType.RES_NET50, labels)
    # model.model = jit_model
    print((model.predict(model.preprocess(x_val), batch_size=64).argmax(1) == y_val).mean())


test_torch()
