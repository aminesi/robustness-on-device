import os

import numpy as np

from utilities import DATA_ROOT, SELECTED, ModelType, Backend, BATCH_SIZE, get_image_size, set_image_size, progress
from models import TFModel, TorchModel

print('loading started')
y_val = np.load(DATA_ROOT + 'y_val.npy')
labels = np.load(DATA_ROOT + 'labels.npy')
print('loading finished')

intersection = np.array([i for i in range(len(y_val))])

for model_type in [ModelType.MOBILE_NET_V2, ModelType.RES_NET50, ModelType.INCEPTION_V3]:
    if model_type == ModelType.INCEPTION_V3:
        set_image_size(299)
    for backend in [Backend.TENSOR_FLOW, Backend.PY_TORCH]:
        print('start for:', backend.value, model_type.value)
        path = DATA_ROOT + backend.value + '/' + model_type.value + '/'
        os.makedirs(path, exist_ok=True)
        if os.path.isfile(path + 'correct_predictions.npy'):
            print('already exists')
            correct_predictions = np.load(path + 'correct_predictions.npy')
        else:
            if backend == Backend.TENSOR_FLOW:
                model = TFModel(model_type, labels)
            else:
                model = TorchModel(model_type, labels)
            correct_predictions = np.array([True] * len(y_val))
            for i in range(50):
                start = i * 1000
                end = start + 1000
                x = np.load(DATA_ROOT + ('x_val_%d_%d.npy' % (get_image_size(), end)))
                correct_predictions[start:end] = model.correctly_classified(
                    model.preprocess(x),
                    y_val[start:end],
                    BATCH_SIZE)
                print((i + 1) * 2, '%')
            correct_predictions = np.where(correct_predictions)[0]
            np.save(path + 'correct_predictions.npy', correct_predictions)
        print('accuracy:', len(correct_predictions) / len(y_val))
        intersection = np.intersect1d(correct_predictions, intersection)

print('saving selections')
os.makedirs(DATA_ROOT + SELECTED, exist_ok=True)
np.save(DATA_ROOT + SELECTED + 'intersection.npy', intersection)
np.random.shuffle(intersection)
indices = intersection[:min(3000, len(intersection))]
indices.sort()
np.save(DATA_ROOT + SELECTED + 'indices.npy', indices)
y_val = y_val[indices]

for size in [224, 299]:
    start = 0
    end = 0
    set_image_size(size)
    x_val = np.zeros((3000, get_image_size(), get_image_size(), 3), np.uint8)

    for i in progress(50, desc='saving selected images in size: %d' % get_image_size()):
        part = np.load(DATA_ROOT + ('x_val_%d_%d.npy' % (get_image_size(), (i + 1) * 1000)))
        new_indices = indices[start:] - i * 1000
        new_indices = new_indices[new_indices < 1000]
        end = start + len(new_indices)
        x_val[start:end] = part[new_indices]
        start = end
    np.save(DATA_ROOT + SELECTED + ('x_val_%d.npy' % get_image_size()), x_val)

np.save(DATA_ROOT + SELECTED + 'y_val.npy', y_val)
print('save completed')
