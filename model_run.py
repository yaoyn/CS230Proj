# Adapted from: https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
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
K.tensorflow_backend._get_available_gpus()


def load_data(path, folder, load_type, subset_frac, set_frac):
    
    base_dir = os.path.join(path, folder)
    train_frac = set_frac[0]
    val_frac = set_frac[1]
    test_frac = set_frac[2]


    if (load_type == 'subset' or load_type == 'full'):
        img_h = 450
        img_w = 600
    elif (load_type == 'subset_resize' or load_type == 'full_resize'):
        img_h = 75
        img_w = 100

    # Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_dir, '*', '*.jpg'))}

    # This dictionary is useful for displaying more human-friendly labels later on
    lesion_type_dict = {
      'nv': 'Melanocytic nevi',
      'mel': 'Melanoma',
      'bkl': 'Benign keratosis-like lesions ',
      'bcc': 'Basal cell carcinoma',
      'akiec': 'Actinic keratoses',
      'vasc': 'Vascular lesions',
      'df': 'Dermatofibroma'
    }

    lesion_fullname_dict = {
      'Melanocytic nevi': 'nv',
      'Melanoma' : 'mel',
      'Benign keratosis-like lesions ': 'bkl',
      'Basal cell carcinoma': 'bcc',
      'Actinic keratoses': 'akiec',
      'Vascular lesions': 'vasc',
      'Dermatofibroma': 'df' 
    }


    # from: https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
    #Step 3: reading and processing data
    csv_filename = 'HAM10000_metadata.csv'
    df_curr = pd.read_csv(os.path.join(base_dir, csv_filename))

    # Creating New Columns for better readability
    df_curr['path'] = df_curr['image_id'].map(imageid_path_dict.get)
    df_curr['cell_type'] = df_curr['dx'].map(lesion_type_dict.get) 
    df_curr['cell_type_idx'] = pd.Categorical(df_curr['cell_type']).codes
    df_curr.head()
    dtype = pd.CategoricalDtype(np.unique(df_curr['cell_type'].values), ordered=True)
    cat_labels = np.unique(pd.Categorical.from_codes(codes=df_curr['cell_type_idx'], dtype= dtype))
    for i in range(len(cat_labels)):
        cat_labels[i] = lesion_fullname_dict[cat_labels[i]]
    #Step 4: data cleaning
    df_curr.isnull().sum()
    df_curr['age'].fillna((df_curr['age'].mean()), inplace=True)
    df_curr.isnull().sum()
    #Step 6: loading and resizing of images
    if (load_type == 'subset'):
        print('subset')
        df_curr = df_curr.sample(frac=subset_frac)
        m_curr = df_curr.shape[0] # percentage of total
        df_curr['image'] = df_curr['path'].map(lambda x: np.asarray(Image.open(x)))
    elif (load_type == 'subset_resize'):
        print('subset_resize')
        df_curr = df_curr.sample(frac=subset_frac)
        m_curr = df_curr.shape[0] # percentage of total
        df_curr['image'] = df_curr['path'].map(lambda x: np.asarray(Image.open(x).resize((img_w,img_h))))
    elif (load_type == 'full'):
        print("full")
        df_curr = df_curr
        m_curr = df_curr.shape[0] # percentage of total
        df_curr['image'] = df_curr['path'].map(lambda x: np.asarray(Image.open(x)))
    elif (load_type == 'full_resize'):
        print("full_resize")
        m_curr = df_curr.shape[0] # percentage of total
        df_curr['image'] = df_curr['path'].map(lambda x: np.asarray(Image.open(x).resize((img_w,img_h))))

    df_X = df_curr.drop(columns=['cell_type_idx'], axis=1)
    df_Y = df_curr['cell_type_idx']
    #Step 7: train test split
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(df_X, df_Y, test_size=test_frac, random_state=1234)
    #Step 8: normalization
    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())
    x_train = x_train / 255
    x_test = x_test / 255
    # Step 9: label encoding
    # Perform one-hot encoding on the labels
    y_train = to_categorical(y_train_o, num_classes = 7)
    y_test = to_categorical(y_test_o, num_classes = 7)
    #Step 10: splitting training and validation split
    test_frac_train = val_frac / train_frac
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = test_frac_train, random_state = 2)
    #Step 11: model building, CNN
    input_shape = (img_h, img_w, 3)
    

    data_dict = {
      "x_train": x_train,
      "y_train": y_train,
      "x_validate": x_validate,
      "y_validate": y_validate,
      "x_test": x_test,
      "y_test": y_test,
      "y_train_o": y_train_o}
    
    return data_dict, cat_labels, img_h, img_w

def train_model(model_name, data, epochs, batch_size, class_weight, model):

    x_train = data['x_train']
    y_train = data['y_train']
    x_validate = data['x_validate']
    y_validate = data['y_validate']

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                              patience=3, 
                                              verbose=1, 
                                              factor=0.5, 
                                              min_lr=0.00001)

    datagen = ImageDataGenerator(
              featurewise_center=False,  # set input mean to 0 over the dataset
              samplewise_center=False,  # set each sample mean to 0
              featurewise_std_normalization=False,  # divide inputs by std of the dataset
              samplewise_std_normalization=False,  # divide each input by its std
              zca_whitening=False,  # apply ZCA whitening
              rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
              zoom_range = 0.1, # Randomly zoom image 
              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
              horizontal_flip=True,  # randomly flip images
              vertical_flip=True)  # randomly flip images

    datagen.fit(x_train)


    # class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_o), y=y_train_o)
    print("ddd")
    d_class_weights = dict(enumerate(class_weight))
    print(d_class_weights)
    #Step 13: fitting the model
    history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                                  epochs = epochs, validation_data = (x_validate,y_validate),
                                  verbose = 1, steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
                                  callbacks=[learning_rate_reduction], class_weight=d_class_weights) 
                                  
    
    model_name = model_name + ".h5"
    model.save(model_name)
    return model, history


def evaluate_model(model, data):
    x_train = data['x_train']
    y_train = data['y_train']
    x_validate = data['x_validate']
    y_validate = data['y_validate']
    x_test = data['x_test']
    y_test = data['y_test']

    loss_t, accuracy_t = model.evaluate(x_train, y_train, verbose=1)
    loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    result = np.array([[loss_t, accuracy_t], [loss_v, accuracy_v], [loss, accuracy]])
    return result

def compute_cfm(model, data):
    x_train = data['x_train']
    y_train = data['y_train']
    x_validate = data['x_validate']
    y_validate = data['y_validate']
    x_test = data['x_test']
    y_test = data['y_test']
    
    Y_pred = model.predict(x_train)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_train,axis = 1) 
    confusion_mtx_train = confusion_matrix(Y_true, Y_pred_classes)

 
    Y_pred = model.predict(x_validate)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_validate,axis = 1) 
    confusion_mtx_val = confusion_matrix(Y_true, Y_pred_classes)

    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_test,axis = 1) 
    confusion_mtx_test = confusion_matrix(Y_true, Y_pred_classes)
    
    return confusion_mtx_train, confusion_mtx_val, confusion_mtx_test

#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_name, model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(model_name + "_hist.eps")


# Function to plot confusion matrix    
def plot_confusion_matrix(model_name, plot_type, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_name + "_" + plot_type + ".eps")


def calculate_individual_recall(cm, cat_labels):
    recall = np.zeros(cat_labels.shape)
    for i in range(len(cat_labels)):
        denom = np.sum(cm[i,:])
        if (denom == 0):
            denom = 1
        recall[i] = cm[i,i] / denom
    return recall

def calculate_cancer_recall(cm, cat_labels):
    numerator = np.zeros(2)
    denom = np.zeros(2)
    for i in range(len(cat_labels)):
        if (cat_labels[i] == 'mel' or cat_labels[i] == 'bcc' or cat_labels[i] == 'akiec'):
            numerator[0] += cm[i,i] 
            denom[0] += np.sum(cm[i,:])
        else:
            numerator[1] += cm[i,i]
            denom[1] += np.sum(cm[i,:])
    if (denom[0] == 0):
        denom[0] = 1
    if (denom[1] == 0):
        denom[1] = 1
    recall_total = np.divide(numerator, denom)
    return recall_total

def compute_recall(cfm_train, cfm_val, cfm_test, cat_labels):
    recall_i_train = calculate_individual_recall(cfm_train, cat_labels)
    recall_total_train = calculate_cancer_recall(cfm_train, cat_labels)

    recall_i_train = recall_i_train.reshape((1, len(recall_i_train)))
    recall_total_train = recall_total_train.reshape((1, len(recall_total_train)))                                       
    
    recall_i_val = calculate_individual_recall(cfm_val, cat_labels)
    recall_total_val = calculate_cancer_recall(cfm_val, cat_labels)
    recall_i_val = recall_i_val.reshape((1, len(recall_i_val)))
    recall_total_val = recall_total_val.reshape((1, len(recall_total_val)))      


    recall_i_test = calculate_individual_recall(cfm_test, cat_labels)
    recall_total_test = calculate_cancer_recall(cfm_test, cat_labels)
    recall_i_test = recall_i_test.reshape((1, len(recall_i_test)))
    recall_total_test = recall_total_test.reshape((1, len(recall_total_test)))
    
    recall_comb = np.concatenate([recall_i_train, recall_i_val, recall_i_test], axis = 0)
    
    recall_total_comb = np.concatenate([recall_total_train, recall_total_val, recall_total_test], axis = 0)
    return recall_comb, recall_total_comb

def plot_all(model_name, history, cfm_train, cfm_val, cfm_test, cat_labels):
    plot_model_history(model_name, history)
    plt.close()
    plot_confusion_matrix(model_name, "train", cfm_train, classes = cat_labels) 
    plt.close()
    plot_confusion_matrix(model_name, "val", cfm_val, classes = cat_labels) 
    plt.close()
    plot_confusion_matrix(model_name, "test", cfm_test, classes = cat_labels) 

if __name__ == "__main__":
    base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate = 0.3)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    class_weight = np.array([1,1,1,1,1,1,1])
    model = Model(inputs = base_model.input, outputs = predictions)
    model = train_model(epoch, batch_size, model, history)

