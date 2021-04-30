from utilities import *
from foolbox.attacks import FGSM, LinfBasicIterativeAttack, BoundaryAttack
import foolbox as fb
from models import TFModel, TorchModel, BaseModel
import numpy as np
import eagerpy as ep
import cv2

y_val = np.load(DATA_ROOT + SELECTED + 'y_val.npy')
labels = np.load(DATA_ROOT + 'labels.npy')
indices = np.load(DATA_ROOT + SELECTED + 'indices.npy')
y_val_keras = np.argsort(np.argsort(labels[:, 0]))[y_val]


def attack_model(model: BaseModel, attack: fb.attacks.Attack, batch_save: bool, attack_name: str, **kwargs):
    start_batch = int(os.getenv('START_BATCH', 0))
    total = int(np.ceil(len(x_val) / BATCH_SIZE))
    kwargs = {'epsilons': .005, **kwargs}

    print('Model:', model.backend.value, model.model_type.value)
    print('Attack:', attack_name.upper())

    pre = dict(mean=model.mean, std=model.std, **model.kwargs)
    if isinstance(model, TFModel):
        fmodel: fb.Model = fb.TensorFlowModel(model.model, bounds=(0, 255), preprocessing=pre)
        fmodel = fmodel.transform_bounds((0, 1))
    else:
        fmodel: fb.Model = fb.PyTorchModel(model.model, bounds=(0, 1), preprocessing=pre)

    results = np.zeros((len(x_val), get_image_size(), get_image_size(), 3), np.uint8)
    for i in progress(start_batch, total, desc='create adversarial'):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        if i == total - 1:
            end = len(x_val)

        x = x_val[start:end]
        if isinstance(model, TorchModel):
            x = channel_first(x)
        x = ep.from_numpy(fmodel.dummy, (x / 255).astype(np.float32)).raw

        y = ep.from_numpy(fmodel.dummy, y_val_keras[start:end].astype(np.int32)).raw
        if isinstance(model, TorchModel):
            y = y.type(torch.cuda.LongTensor)

        raw_advs, clipped_advs, success = attack(fmodel, x, y, **kwargs)
        del raw_advs
        del success
        if isinstance(model, TFModel):
            results[start:end] = np.rint(clipped_advs.numpy() * 255).astype(np.uint8)
        else:
            results[start:end] = np.rint(channel_last(cpu(clipped_advs).numpy()) * 255).astype(np.uint8)
        if batch_save:
            save(results, model, attack_name, start, end)
        del clipped_advs
    return results


def save(images, model, name, start=0, end=None):
    path = DATA_ROOT + model.backend.value + '/' + model.model_type.value + '/' + name + '/'
    os.makedirs(path, exist_ok=True)
    if end is None:
        end = len(images)
    for i in range(start, end):
        image = images[i, :, :, ::-1]
        cv2.imwrite(path + ('%04d-%05d.png' % (i, indices[i] + 1)), image)


# model = TFModel(ModelType.RES_NET50, labels)
# adv = attack_model(model, BoundaryAttack(steps=5000), True, 'boundary', epsilons=None)
# print('adversarial accuracy', (model.correctly_classified(model.preprocess(adv), y_val, BATCH_SIZE)).sum() / 30)

for backend in [Backend.PY_TORCH, Backend.TENSOR_FLOW]:
    for model_type in [ModelType.MOBILE_NET_V2, ModelType.RES_NET50, ModelType.INCEPTION_V3]:
        if backend == Backend.TENSOR_FLOW:
            model = TFModel(model_type, labels)
        else:
            model = TorchModel(model_type, labels)
        set_image_size(model.image_size)
        x_val = np.load(DATA_ROOT + SELECTED + 'x_val_' + str(get_image_size()) + '.npy')

        for attack, attack_name in [(FGSM(), 'fgsm'), (LinfBasicIterativeAttack(), 'bim')]:
            adv = attack_model(model, attack, True, attack_name)
            print('adversarial accuracy',
                  (model.correctly_classified(model.preprocess(adv), y_val, BATCH_SIZE)).sum() / 30, '%')
