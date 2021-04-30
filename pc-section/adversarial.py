import numpy as np
# from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import mobilenet_v2
import tensorflow as tf
import os
import cv2

from utilities import DATA_ROOT, SELECTED, BATCH_SIZE

print('loading started')
x_val = np.load(DATA_ROOT + SELECTED + 'x_val.npy')
y_val = np.load(DATA_ROOT + SELECTED + 'y_val.npy')
selected_indices = np.load(DATA_ROOT + SELECTED + 'indices.npy')
labels = np.load(DATA_ROOT + 'labels.npy')
labels_keras = np.array(sorted(labels, key=lambda x: x[0]))
y_val_keras = np.argsort(np.argsort(labels[:, 0]))[y_val]
print('loading finished')

# model = resnet.ResNet50()
model = mobilenet_v2.MobileNetV2()

# x_processed = resnet.preprocess_input(x_val)
x_processed = mobilenet_v2.preprocess_input(x_val)
indices = np.argmax(model.predict(x_processed, BATCH_SIZE), 1) == y_val_keras
x_val = x_val[indices]
y_val = y_val[indices]
y_val_keras = y_val_keras[indices]
x_processed = x_processed[indices]
selected_indices = selected_indices[indices]
print(len(y_val_keras), 'were classified correctly')

loss_function = tf.keras.losses.CategoricalCrossentropy()


def calc_per(x, y):
    with tf.GradientTape() as gr_tape:
        gr_tape.watch(x)
        pred = model(x)
        loss = loss_function(y, pred)
    return tf.sign(gr_tape.gradient(loss, x)).numpy()


def create_perturbation(x, y):
    y_hot = tf.one_hot(y, 1000, dtype=np.int8)
    x = tf.convert_to_tensor(x)
    last = 0
    results = []
    for i in range(0, len(x) + 1, BATCH_SIZE):
        if i != 0:
            results.append(calc_per(x[last:i], y_hot[last:i]))
            last = i
    if last != len(x):
        results.append(calc_per(x[last:], y_hot[last:]))
    return np.vstack(results)


perturbation = create_perturbation(x_processed, y_val_keras)

for eps in [0.01]:
    path = DATA_ROOT + 'mobilenet/fgsm-' + str(eps) + '/'
    os.makedirs(path, exist_ok=True)
    print('fgsm for eps:%.3f' % eps)
    distorted_images = (np.clip((x_processed + eps * perturbation) * .5 + .5, 0, 1) * 255).astype(np.uint8)
    del x_processed
    for i in range(len(distorted_images)):
        if i % 500 == 0:
            print('%d images saved.' % i)
        image = distorted_images[i, :, :, ::-1]
        cv2.imwrite(path + ('%04d - ' % i) + str(selected_indices[i] + 1) + '.png', image)
    indices = np.argmax(model.predict(mobilenet_v2.preprocess_input(distorted_images), verbose=1),
                        axis=1) == y_val_keras
    np.save(path + 'pred_result.npy', np.vstack((selected_indices, indices)).transpose())
    np.save(path + 'y_val.npy', y_val)
    print('accuracy for eps %.3f is: %f' % (eps, np.sum(indices) / len(y_val_keras)))
