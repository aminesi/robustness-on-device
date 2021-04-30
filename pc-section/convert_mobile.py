import numpy as np

from models import TFModel, TorchModel
from utilities import ModelType, DATA_ROOT, SELECTED, MOBILE, Backend, get_image_size, set_image_size


# noinspection PyShadowingBuiltins
def to_json(input, output, function=None):
    if not isinstance(input, np.ndarray):
        input = np.load(input)
    with open(output, 'w') as f:
        if function:
            input = map(function, input)
        f.write(str(list(input)))
        f.close()


labels = np.load(DATA_ROOT + 'labels.npy')

to_json(DATA_ROOT + SELECTED + 'indices.npy', DATA_ROOT + MOBILE + 'selected_indices.json')
to_json(DATA_ROOT + SELECTED + 'y_val.npy', DATA_ROOT + MOBILE + 'selected_y.json')
keras_index = np.argsort(np.argsort(labels[:, 0]))
to_json(
    np.hstack((np.array([i for i in range(1000)])[:, np.newaxis], keras_index[:, np.newaxis], labels)),
    DATA_ROOT + MOBILE + 'labels.json',
    lambda x: {
        'imageNetIndex': x[0],
        'kerasIndex': x[1],
        'id': x[2],
        'text': x[3]
    })

for backend in [Backend.PY_TORCH, Backend.TENSOR_FLOW]:
    for model_type in [ModelType.MOBILE_NET_V2, ModelType.RES_NET50, ModelType.INCEPTION_V3]:
        if backend == Backend.TENSOR_FLOW:
            model = TFModel(model_type, labels)
        else:
            model = TorchModel(model_type, labels)
        set_image_size(model.image_size)
        x = np.load(DATA_ROOT + SELECTED + 'x_val_' + str(get_image_size()) + '.npy')[::3]
        model.convert_mobile(x)
        print('converted: ', backend.value, model_type.value)
