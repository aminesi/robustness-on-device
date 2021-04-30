import os

from models import TFModel, TorchModel
from utilities import Backend, ModelType, DATA_ROOT, SELECTED, get_image_size, set_image_size
import numpy as np
import cv2
from numba import cuda

y_val = np.load(DATA_ROOT + SELECTED + 'y_val.npy')
labels = np.load(DATA_ROOT + 'labels.npy')


def get_images(path):
    files = list(sorted(map(lambda x: x.path, os.scandir(path))))
    images = np.zeros((len(files), get_image_size(), get_image_size(), 3), np.uint8)
    for i in range(len(files)):
        images[i] = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB)
    return images


for backend in [Backend.PY_TORCH, Backend.TENSOR_FLOW]:
    device = cuda.get_current_device()
    device.reset()
    for model_type in [ModelType.MOBILE_NET_V2, ModelType.RES_NET50, ModelType.INCEPTION_V3]:
        if backend == Backend.TENSOR_FLOW:
            model = TFModel(model_type, labels)
        else:
            model = TorchModel(model_type, labels)
        set_image_size(model.image_size)

        path = DATA_ROOT + backend.value + '/' + model_type.value + '/'
        for entry in os.scandir(path):
            if isinstance(entry, os.DirEntry) and entry.is_dir():
                attack = entry.name
                x_val = get_images(entry.path)
                print('')
                print('Model:', model.backend.value, model.model_type.value)
                print('Attack:', attack.upper())
                accuracy = model.correctly_classified(model.preprocess(x_val), y_val, 64).mean() * 100
                success_rate = 100 - accuracy
                print('Accuracy:', accuracy, '%')
                print('Success rate:', success_rate, '%')
