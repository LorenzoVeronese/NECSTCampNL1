# SOURCES:
# https://www.youtube.com/watch?v=0kiroPnV1tM
# https://www.youtube.com/watch?v=cUHPL_dk17E
# https://www.youtube.com/watch?v=RaswBvMnFxk


import tensorflow as tf
import os
import random
import numpy as np


from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import nibabel


seed = 42
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256 
IMG_CHANNELS = 1

TRAIN_PATH = r"C:\Users\loren\OneDrive - Politecnico di Milano\Desktop\Lorenzo\Università\NECSTCamp\Progetto\Segmentation\Data"
LABELS_PATH = r"C:\Users\loren\OneDrive - Politecnico di Milano\Desktop\Lorenzo\Università\NECSTCamp\Progetto\Segmentation\Labels"
TEST_PATH = r"C:\Users\loren\OneDrive - Politecnico di Milano\Desktop\Lorenzo\Università\NECSTCamp\Progetto\Segmentation\Data\volumes 100-139"


def extractImages(path, start, end):
    """
    PARAM
        path: location of directory which contains files to read
        start: index of the first image to read
        end: index of the last image to read
    RETURN
        a list of numpy arrays, each one representing an image in the path directory
        named from start to end
    """
    
    images = []
    if(path == LABELS_PATH): # it's in the LABELS_PATH directory
        prefix = 'labels-'
    else:
        prefix = 'volume-' # it's in the TRAIN_PATH or TEST_PATH directory
    
    tempPath = path
    for i in range(start, end):
        tempPath = os.path.join(path, prefix + str(i) + '.nii.gz')
        epiImg = nibabel.load(tempPath)
        epiImg = epiImg.get_fdata()
        epiImg = epiImg[:, :, 70]
        # print(len(epiImg.shape)) this is the number of channels (es. 2 => one channel)
        # print(epiImg[100][100])
        # print(type(epiImg[100][100]))
        if(prefix == 'labels-'):
            for i in range(0, len(epiImg)):
                for j in range(0, len(epiImg[i])):
                    if (int(epiImg[j][i]) + 1) != 5:
                        epiImg[j][i] = 0.0
        epiImg = resize(epiImg, (IMG_HEIGHT, IMG_WIDTH))
        images.append(epiImg)

    return images



def sanityCheck(trainOriginals, trainLabels, preds_train_t):
    ix = random.randint(0, len(preds_train_t))
    imshow(trainOriginals[ix])
    plt.show()
    imshow(np.squeeze(trainLabels[ix]))
    plt.show()
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()



def uNet():

    #Contraction path
    # 256x256
    c1 = tf.keras.layers.Conv2D(
        16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(256, 256, 1))(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c1)

    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    # 128x128
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p1)

    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c2)

    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    # 64x64
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c3)

    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    # 32x32
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c4)

    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    # 16x16
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    # 32x32
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c6)

    # 64x64
    u7 = tf.keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c7)

    # 128x128
    u8 = tf.keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c8)

    # 256x256
    u9 = tf.keras.layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return outputs



#Prepare dataset
# Train set
# images
print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
print('Loading images...')

trainOriginals = []
trainOriginals1 = np.array(extractImages(os.path.join(TRAIN_PATH, 'volumes 0-49'), 0, 50))
trainOriginals2 = np.array(extractImages(os.path.join(TRAIN_PATH, 'volumes 50-99'), 50, 100))
for img in trainOriginals1:
    trainOriginals.append(img)
for img in trainOriginals2:
    trainOriginals.append(img)
trainOriginals = np.array(trainOriginals)
print('trainOriginals done')
# labels
trainLabels = np.array(extractImages(LABELS_PATH, 0, 100))  # 2 to change to 49
print('trainLabels done')
# Test set
# images
testOriginals = np.array(extractImages(os.path.join(TRAIN_PATH, 'volumes 100-139'), 100, 140))
print('testOriginals done')
# labels
testLabels = np.array(extractImages(LABELS_PATH, 100, 140))
print('testLabels done')



#Build the model
print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
print('Building the model...')
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

outputs = uNet()

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()



#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'model_for_nuclei.h5', verbose=1, save_best_only=True
    )

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

results = model.fit(trainOriginals, trainLabels, validation_split=0.1,
                    batch_size=16, epochs=25, callbacks=callbacks)

####################################

idx = random.randint(0, len(trainOriginals))


preds_train = model.predict(trainOriginals[:int(trainOriginals.shape[0]*0.9)], verbose=1)
preds_val = model.predict(trainOriginals[int(trainOriginals.shape[0]*0.9):], verbose=1)
preds_test = model.predict(testOriginals, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Perform some sanity check on some random training samples
print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_')
print('Sanity check: ')
# 1
sanityCheck(trainOriginals, trainLabels, preds_train_t)
'''
ix = random.randint(0, len(preds_train_t))
imshow(trainOriginals[ix])
plt.show()
imshow(np.squeeze(trainLabels[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()'''
# 2
sanityCheck(trainOriginals, trainLabels, preds_train_t)
# 3
sanityCheck(trainOriginals, trainLabels, preds_train_t)
# 4
sanityCheck(trainOriginals, trainLabels, preds_train_t)
# 5
sanityCheck(trainOriginals, trainLabels, preds_train_t)
