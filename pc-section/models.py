import copy
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras import applications
from tensorflow.python.keras.models import Model
from torch.nn import Module
from torchvision import models

from utilities import Backend, DATA_ROOT, WEIGHTS, ModelType, cuda, progress, cpu, channel_first

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.get_logger().setLevel("ERROR")


class BaseModel(ABC):
    def __init__(self, backend: Backend, model_type: ModelType, labels):
        self.model = None
        self.model_type: ModelType = model_type
        self.mean = None
        self.std = None
        self.image_size = 224
        self.kwargs = {}
        self.backend = backend
        self.labels = labels
        self.weights_path = DATA_ROOT + WEIGHTS + backend.value + '/'
        self._create_model()

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def predict(self, x, batch_size: int):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def convert_mobile(self, calibration_data=None):
        pass

    def get_y_keras(self, y):
        return np.argsort(np.argsort(self.labels[:, 0]))[y]

    def correctly_classified(self, x, y, batch_size: int):
        return np.argmax(self.predict(x, batch_size), axis=1) == self.get_y_keras(y)


class TFModel(BaseModel):
    def __init__(self, model_type: ModelType, labels):
        self.application = applications.imagenet_utils
        self.model: Model = None
        super().__init__(Backend.TENSOR_FLOW, model_type, labels)

    def _create_model(self):
        if self.model_type == ModelType.MOBILE_NET_V2:
            self.application = applications.mobilenet_v2
            self.model = self.application.MobileNetV2(
                weights=self.weights_path + 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
            self.mean = [127.5, 127.5, 127.5]
            self.std = [127.5, 127.5, 127.5]
        elif self.model_type == ModelType.RES_NET50:
            self.application = applications.resnet
            self.model = self.application.ResNet50(
                weights=self.weights_path + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
            self.kwargs = {'flip_axis': -1}
            self.mean = [103.939, 116.779, 123.68]
        elif self.model_type == ModelType.INCEPTION_V3:
            self.application = applications.inception_v3
            self.model = self.application.InceptionV3(
                weights=self.weights_path + 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
            self.mean = [127.5, 127.5, 127.5]
            self.std = [127.5, 127.5, 127.5]
            self.image_size = 299
        else:
            raise Exception('unknown model')

    def predict(self, x, batch_size: int):
        return self.model.predict(x, batch_size, verbose=True)

    def preprocess(self, x):
        return self.application.preprocess_input(x)

    # noinspection PyTypeChecker
    def convert_mobile(self, calibration_data=None):
        path = DATA_ROOT + self.backend.value + '/' + self.model_type.value + '/'
        os.makedirs(path, exist_ok=True)

        lite_model = tf.lite.TFLiteConverter.from_keras_model(self.model).convert()
        with open(path + self.model_type.value + '.tflite', 'wb') as file:
            file.write(lite_model)
            file.close()

        def generate_data():
            for data in calibration_data:
                data = self.preprocess(data)
                yield [data[np.newaxis]]

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = generate_data
        quant = converter.convert()
        with open(path + self.model_type.value + '-quant.tflite', 'wb') as file:
            file.write(quant)
            file.close()


class TorchModel(BaseModel):
    def __init__(self, model_type: ModelType, labels):
        self.model: Module = None
        super().__init__(Backend.PY_TORCH, model_type, labels)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.kwargs = {'axis': -3}

    def _create_model(self):
        if self.model_type == ModelType.MOBILE_NET_V2:
            self.model = models.quantization.mobilenet_v2()
            self.model.load_state_dict(torch.load(self.weights_path + 'mobilenet_v2-b0353104.pth'))
        elif self.model_type == ModelType.RES_NET50:
            self.model = models.quantization.resnet50()
            self.model.load_state_dict(torch.load(self.weights_path + 'resnet50-19c8e357.pth'))
        elif self.model_type == ModelType.INCEPTION_V3:
            self.model = models.quantization.inception_v3()
            self.model.load_state_dict(torch.load(self.weights_path + 'inception_v3_google-1a9a5a14.pth'))
            self.image_size = 299
        else:
            raise Exception('unknown model')
        self.model = cuda(self.model)
        self.model.eval()

    def predict(self, x, batch_size: int):
        results = np.zeros((len(x), 1000), np.float32)
        total = int(np.ceil(len(x) / batch_size))
        for i in progress(total):
            start = i * batch_size
            end = start + batch_size
            if i == total - 1:
                end = len(x)
            results[start:end] = self.__predict(x[start:end])
        return results

    def __predict(self, x):
        with torch.no_grad():
            return cpu(self.model(cuda(torch.from_numpy(x)))).numpy()

    def preprocess(self, x):
        return applications.imagenet_utils.preprocess_input(channel_first(x), data_format='channels_first',
                                                            mode='torch')

    # noinspection PyTypeChecker
    def convert_mobile(self, calibration_data=None):
        path = DATA_ROOT + self.backend.value + '/' + self.model_type.value + '/'
        os.makedirs(path, exist_ok=True)
        script = torch.jit.script(self.model)
        torch.jit.save(script, path + self.model_type.value + '.pt')

        backup = copy.deepcopy(self.model)
        cpu(self.model)

        self.model.fuse_model()

        self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self.model, inplace=True)

        # Calibrate with the training set
        calibration_data = self.preprocess(calibration_data)
        self.predict(calibration_data, 64)

        # Convert to quantized model
        quant_model = torch.quantization.convert(self.model)
        self.model = backup
        script = torch.jit.script(quant_model)
        torch.jit.save(script, path + self.model_type.value + '-quant.pt')
