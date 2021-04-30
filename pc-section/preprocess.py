import os
import numpy as np
from scipy import io
import cv2
from utilities import DATA_ROOT, set_image_size, get_image_size
import utilities

IMAGE_FOLDER = 'images/'
INFO_FOLDER = 'info/'
META = 'meta.mat'
GROUND = 'ILSVRC2012_validation_ground_truth.txt'
IMAGE_PATH = DATA_ROOT + IMAGE_FOLDER
META_PATH = DATA_ROOT + INFO_FOLDER + META
GROUND_PATH = DATA_ROOT + INFO_FOLDER + GROUND


def convert_image(image: str):
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    size = get_image_size()
    image_size = np.array(img.shape[:2])
    ratio = max(size, 256) / image_size.min()
    image_size = np.rint(image_size * ratio).astype(np.int)
    img = cv2.resize(img, tuple(np.flip(image_size)), interpolation=cv2.INTER_CUBIC)
    y, x = image_size // 2 - size // 2
    return img[y:y + size, x:x + size, :]


if __name__ == '__main__':
    files = os.listdir(IMAGE_PATH)
    files.sort()
    files = [IMAGE_PATH + image for image in files]

    # max_inputs = 3000
    max_inputs = len(files)

    for size in [224, 299]:
        set_image_size(size)
        utilities.IMAGE_SIZE = size
        x_val = np.zeros((1000, get_image_size(), get_image_size(), 3), dtype=np.uint8)
        print('start conversion for size:', size)
        for i in utilities.progress(max_inputs):
            x_val[i % 1000] = convert_image(files[i])
            if (i + 1) % 1000 == 0:
                np.save(DATA_ROOT + ('x_val_%d_%d.npy' % (size, i + 1)), x_val)

        print('conversion completed')

    synsets = io.loadmat(META_PATH)['synsets']
    labels = np.array(list(map(lambda synset: (synset[1][0], synset[2][0]), synsets[:1000, 0])))
    np.save(DATA_ROOT + 'labels.npy', labels)

    ground = open(GROUND_PATH)
    y_val = [int(line.strip()) - 1 for line in ground]
    ground.close()
    np.save(DATA_ROOT + 'y_val.npy', y_val)
    print('all done')
