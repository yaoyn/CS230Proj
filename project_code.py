import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import itertools
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras import regularizers

from keras import backend as K

import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import model_run 


# from: https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
# load_type: subset means only portion based on subset_frac is loaded
#          : resize means shrinking the images
# train, val and test frac is based on the amount of loaded data.
path = '/home/ubuntu/'
folder = 'skin-cancer-mnist-ham10000'

load_type = 'full_resize' # 'subset', 'subset_resize, 'full', 'full_resize'
subset_frac = 0.1
set_frac = np.array([0.7, 0.15, 0.15])
num_classes = 7

data, cat_labels, img_h, img_w = model_run.load_data(path, folder, load_type, subset_frac, set_frac)
print(cat_labels)
K.tensorflow_backend._get_available_gpus()


epochs = 75

#Model 1
base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(rate = 0.3)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model_1 = Model(inputs = base_model.input, outputs = predictions)
batch_size = 128
model_name = "model_1"
print("model creation completed")
class_weight = np.array([1,1,1,1,1,1,1])
model, history = model_run.train_model(model_name, data, epochs, batch_size, class_weight, model_1)
print("model training done")

print("start fitting")
eval_model = model_run.evaluate_model(model, data)
print("fitting ended")

print("post processing started")
cfm_train, cfm_val, cfm_test = model_run.compute_cfm(model, data)
model_run.plot_all(model_name, history, cfm_train, cfm_val, cfm_test, cat_labels)
recall_i, recall_total = model_run.compute_recall(cfm_train, cfm_val, cfm_test, cat_labels)
np.savez(model_name+"_recall", recall_i = recall_i, recall_total = recall_total, eval_model = eval_model)

#Model 2
base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(rate = 0.3)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model_1 = Model(inputs = base_model.input, outputs = predictions)
batch_size = 128
model_name = "model_2"
print("model creation completed")
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(data['y_train_o']), y=data['y_train_o'])
model, history = model_run.train_model(model_name, data, epochs, batch_size, class_weight, model_1)
print("model training done")

print("start fitting")
eval_model = model_run.evaluate_model(model, data)
print("fitting ended")

print("post processing started")
cfm_train, cfm_val, cfm_test = model_run.compute_cfm(model, data)
model_run.plot_all(model_name, history, cfm_train, cfm_val, cfm_test, cat_labels)
recall_i, recall_total = model_run.compute_recall(cfm_train, cfm_val, cfm_test, cat_labels)
np.savez(model_name+"_recall", recall_i = recall_i, recall_total = recall_total, eval_model = eval_model)
print("post processing completed")
print("post processing completed")

#Model 3-5
base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(rate = 0.3)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model_1 = Model(inputs = base_model.input, outputs = predictions)
batch_size = 128
weight_multiplier = np.array([1.5, 2, 3])
model_name_set = ["model_3", "model_4", "model_5"]

i = 0
for model_name in model_name_set:
    print("running " + model_name)
    print("model creation completed")
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(data['y_train_o']), y=data['y_train_o'])
    class_weight[0] = class_weight[0]*weight_multiplier[i]
    class_weight[1] = class_weight[1]*weight_multiplier[i]
    class_weight[5] = class_weight[5]*weight_multiplier[i]
    model, history = model_run.train_model(model_name, data, epochs, batch_size, class_weight, model_1)
    print("model training done")

    print("start fitting")
    eval_model = model_run.evaluate_model(model, data)
    print("fitting ended")

    print("post processing started")
    cfm_train, cfm_val, cfm_test = model_run.compute_cfm(model, data)
    model_run.plot_all(model_name, history, cfm_train, cfm_val, cfm_test, cat_labels)
    recall_i, recall_total = model_run.compute_recall(cfm_train, cfm_val, cfm_test, cat_labels)
    np.savez(model_name+"_recall", recall_i = recall_i, recall_total = recall_total, eval_model = eval_model)
    print("post processing completed")
    print("post processing completed")
    i += 1
    
#Model 6-8
base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(rate = 0.3)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model_1 = Model(inputs = base_model.input, outputs = predictions)
batch_size = 128

weight_multiplier = np.array([1.5, 2, 3])
model_name_set = ["model_6", "model_7", "model_8"]

i = 0
for model_name in model_name_set:
    print("running " + model_name)
    print("model creation completed")
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(data['y_train_o']), y=data['y_train_o'])
    class_weight[0] = class_weight[0]*weight_multiplier[i]*3
    class_weight[1] = class_weight[1]*weight_multiplier[i]
    class_weight[5] = class_weight[5]*weight_multiplier[i]
    model, history = model_run.train_model(model_name, data, epochs, batch_size, class_weight, model_1)
    print("model training done")

    print("start fitting")
    eval_model = model_run.evaluate_model(model, data)
    print("fitting ended")

    print("post processing started")
    cfm_train, cfm_val, cfm_test = model_run.compute_cfm(model, data)
    model_run.plot_all(model_name, history, cfm_train, cfm_val, cfm_test, cat_labels)
    recall_i, recall_total = model_run.compute_recall(cfm_train, cfm_val, cfm_test, cat_labels)
    np.savez(model_name+"_recall", recall_i = recall_i, recall_total = recall_total, eval_model = eval_model)
    print("post processing completed")
    print("post processing completed")
    i += 1
