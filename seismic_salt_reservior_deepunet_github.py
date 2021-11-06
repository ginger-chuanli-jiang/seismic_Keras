
# coding: utf-8

# In[1]:


from __future__ import print_function

import os
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from random import randint
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm_notebook
from copy import deepcopy
import scipy as sp
import random
from skimage.transform import PiecewiseAffineTransform, AffineTransform, warp
from skimage import io
import tensorflow as tf
from scipy import misc
from scipy import ndimage
from scipy.ndimage import map_coordinates
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# In[ ]:


img_size_ori = 101
img_size_target = 96
    
train_df = pd.read_csv("./train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
    
#original images + masks
imgs = [np.array(load_img("./train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[:4000].index)]
msks = [np.array(load_img("./train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df[:4000].index)]

def piecewise_affine(image):
    rows, cols = image.shape[0], image.shape[1]
    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 20)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    dst_rows = src[:, 1] - np.cos(np.linspace(0, 3 * np.pi, src.shape[0])) * 20
    dst_cols = src[:, 0]
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out = warp(image, tform)
    return out

#(6)piecewise_affine
print('piecewise_affine starts')
imgs_new = [piecewise_affine(x) for x in imgs]
msks_new = [piecewise_affine(x) for x in msks] 
print('piecewise_affine ends')

#(3)fliplr
imgs_new2 = [np.fliplr(x) for x in imgs]
msks_new2 = [np.fliplr(x) for x in msks]
    
# Option 2: recreate ori+new+new2 images + masks
train_df_new = deepcopy(train_df)
train_df_new['z'] = train_df['z']
train_df_new.index = train_df.index + '_ds'

train_df_new2 = deepcopy(train_df)
train_df_new2['z'] = train_df['z']
train_df_new2.index = train_df.index + '_ds2'

train_df = train_df.append(train_df_new)
train_df = train_df.append(train_df_new2)
    
#option 2: ori + new + new2
train_df["images"] = imgs + imgs_new + imgs_new2
train_df["masks"] = msks + msks_new + msks_new2
    
train_df_images = train_df.images
def remove_black_old(x):
    black_ids = []
    for i in range(x.shape[0]):
        if np.count_nonzero(x[i]) < 1:
            black_ids.append(i)           
    return black_ids
black_ids_old = remove_black_old(train_df_images)
train_df = train_df.drop(train_df.index[black_ids_old])

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=1337)


# In[ ]:


upconv = True
n_classes = 1
n_filters_start = 16
n_filters = n_filters_start
inputs = Input((96, 96, 1))
droprate=0.25
growth_factor = 2

#inputs = BatchNormalization()(inputs)
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#pool1 = Dropout(droprate)(pool1)
    
n_filters *= growth_factor
pool1 = BatchNormalization()(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
pool2 = Dropout(0.47)(pool2)

n_filters *= growth_factor
pool2 = BatchNormalization()(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
pool3 = Dropout(0.64)(pool3)

n_filters *= growth_factor
pool3 = BatchNormalization()(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
pool4_1 = Dropout(0.71)(pool4_1)

n_filters *= growth_factor
pool4_1 = BatchNormalization()(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
pool4_2 = Dropout(0.56)(pool4_2)

n_filters *= growth_factor
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

n_filters //= growth_factor
if upconv:
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
else:
    up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
up6_1 = BatchNormalization()(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
conv6_1 = Dropout(0.39)(conv6_1)

n_filters //= growth_factor
if upconv:
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
else:
    up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
up6_2 = BatchNormalization()(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
conv6_2 = Dropout(0.98)(conv6_2)

n_filters //= growth_factor
if upconv:
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
else:
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
up7 = BatchNormalization()(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
conv7 = Dropout(0.29)(conv7)

n_filters //= growth_factor
if upconv:
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
else:
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
up8 = BatchNormalization()(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
conv8 = Dropout(0.34)(conv8)

n_filters //= growth_factor
if upconv:
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
else:
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=conv10)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["acc"])
    
early_stopping = EarlyStopping(monitor='val_acc', mode = 'max',patience=20, verbose=0)
model_checkpoint = ModelCheckpoint("./keras.model",monitor='val_acc', 
                                   mode = 'max', save_best_only=True, verbose=0)    
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max', factor=0.2, patience=5, min_lr=0.00001, verbose=0)
epochs = 200
batch_size = 32
tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks = [early_stopping, model_checkpoint, reduce_lr])
score, acc = model.evaluate(x_valid, y_valid, verbose=0)
print('Test accuracy:', acc)


# In[ ]:


model = load_model("./keras.model")


# In[ ]:


preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])


# # Score the model and do a threshold optimization by the best IoU.

# In[ ]:


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


# In[ ]:


thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])


# In[ ]:


threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]


# # Submission
# Load, predict and submit the test image predictions.

# In[ ]:


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


# In[ ]:


train_df = pd.read_csv("./train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]


# In[ ]:


x_test = np.array([upsample(np.array(load_img("./test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


# In[ ]:


preds_test = model.predict(x_test)
pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rlez_mask']
sub.to_csv('submission.csv')

