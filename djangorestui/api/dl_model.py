#os and system level 
import os
import json
import sys
import threading

# preprocessing 
import librosa
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

# modeling imports
from keras.models import Sequential
# modeling imports - layers
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization,Convolution2D, MaxPooling2D,Bidirectional, LSTM,SimpleRNN
from keras.layers import Convolution2D, MaxPooling2D
# modeling imports - optimizer
from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#utils
from sklearn.model_selection import  train_test_split
import keras.backend as K
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 

def get_parameters():
    """Take the input parameters from the command line."""
    with open(sys.argv[1]) as f :
            config_file = json.load(f)
    return config_file

input_params = get_parameters()
print(input_params)

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall. 

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def loadModel(model_file_name, weights_file_name = None):
    """Load keras model from disk."""
    if weights_file_name is None:
        weights_file_name = model_file_name
    # load json and create model
    json_file = open('./api/dl_models/{}.json'.format(model_file_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./api/dl_models/{}.h5".format(weights_file_name))
    print(loaded_model)
    print("Loaded model {} from disk".format(model_file_name))
    return loaded_model

def get_predicted_classes(model_obj, val_data, test_data= None):
    val_preds = model_obj.predict_classes(val_data)
    if test_data is not None:
        test_preds = model_obj.predict_classes(test_data)
        return (val_preds, test_preds)
    return val_preds

def get_predicted_class_prob(model_obj, val_data, test_data= None):
    val_preds = pd.DataFrame(model_obj.predict_proba(val_data))#.apply(lambda x : x.max(), axis=1)
    if test_data is not None:
        test_preds = pd.DataFrame(model_obj.predict_proba(test_data))#.apply(lambda x : x.max(), axis=1)
        return (val_preds, test_preds)
    return val_preds

# def print_metrics(model_obj, train_x, train_y, test_x, test_y):
#     print(model_obj.metrics_names)
#     print(model_obj.evaluate(train_x, train_y))
#     print(model_obj.evaluate(test_x, test_y))
#     train_preds = model_obj.predict_classes(train_x)
#     test_preds = model_obj.predict_classes(test_x)
#     target_names = lb.classes_
#     return (pd.DataFrame(classification_report(y_pred=test_preds,
#                           y_true=test_y.argmax(axis=1),
#                           target_names = target_names,
#                          output_dict = True
#                   )))

def extract_features(y,sr):
    """This function extracts the features from songs, 
    We are extracting the Chroma features, spectral centroid, spectral bandwidth, spectral roll off, zero crossing rate and the mfccc's"""
    chroma_freq = librosa.feature.chroma_stft(y=y, sr=sr)
    specral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20)[0:,0:1500]
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr,n_mfcc=39).T,axis = 0)

    chroma_freq_arr = np.mean(np.ndarray.flatten(chroma_freq))
    spectral_centroid_arr = np.mean(np.ndarray.flatten(specral_centroid))
    spectral_bandwidth_arr = np.mean(np.ndarray.flatten(spectral_bandwidth))
    spectral_rolloff_arr = np.mean(np.ndarray.flatten(spectral_rolloff))
    zero_crossing_rate_arr = np.mean(np.ndarray.flatten(zero_crossing_rate))
    mfcc_arr = np.ndarray.flatten(mfcc)
    temp = np.array([chroma_freq_arr,spectral_centroid_arr,spectral_bandwidth_arr,spectral_rolloff_arr,zero_crossing_rate_arr])
    all_features =  np.concatenate([temp,mfcc_arr])
    return all_features


def load_song(song_path,song_name):
    data_path = '{}/{}'.format(song_path,song_name)
    print(data_path)
    y,sr = librosa.load(data_path)
    return (y,sr)

def create_model_basic_nn():
    """Define and return a model"""
    model_name = 'Basic_NN'
    print(model_name)
    model_basic_nn = Sequential()
    input_shape = 44
    num_labels = 10
    model_basic_nn.add(Dense(256, input_shape=(input_shape,)))
    model_basic_nn.add(Activation('relu'))
    #model_basic_nn.add(Dropout(0.5))

    model_basic_nn.add(Dense(128))
    model_basic_nn.add(Activation('relu'))
    #model_basic_nn.add(Dropout(0.5))

    model_basic_nn.add(Dense(num_labels))
    model_basic_nn.add(Activation('softmax'))
    model_basic_nn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc',recall])

    model_basic_nn = loadModel('model_basic_nn')
    return model_basic_nn

model_basic_nn = create_model_basic_nn()

def create_model_rnn():
    """Define and return the bidi lstm model"""
    model_name = 'Basic_RNN'
    print(model_name)
    input_shape = 44
    num_labels = 10
    
    model_basic_rnn = Sequential()
    model_basic_rnn.add(SimpleRNN(256 , input_shape=(1, 44)))
    model_basic_rnn.add(Dense(256))
    model_basic_rnn.add(Activation('relu'))
    model_basic_rnn.add(Dropout(0.5))

    model_basic_rnn.add(Dense(256))
    model_basic_rnn.add(Activation('relu'))
    model_basic_rnn.add(Dropout(0.5))

    model_basic_rnn.add(Dense(num_labels))
    model_basic_rnn.add(Activation('softmax'))

    model_basic_rnn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc',recall])
    return model_basic_rnn

# model_rnn = create_model_rnn()

# loading scaler, model and predicting
def return_prediction():
    """Run the model and return the prediction"""
    with open('./api/dl_models/scaler.pkl', 'rb') as pickle_file:
        scaler = pickle.load(pickle_file,encoding='latin-1')

    with open('./api/dl_models/label_encoder.pkl', 'rb') as label_encoder_file:
        lb = pickle.load(label_encoder_file,encoding='utf-8')
    print(input_params)
    if input_params:
        song_path = '/home/jarvis/work/GEM2/djangorestui/api/media'
        song_name = input_params['song_name']
        y,sr = load_song(song_path, song_name)
        # print(y)
        try:
            X = extract_features(y,sr).reshape(1,-1)
            X = scaler.transform(np.array(X, dtype = float))
            print(X.shape)
            # print(str(lb.classes_[0],'utf-8'))
            col_names = [str(i,'utf-8') for i in lb.classes_]
            # X = X.reshape(1,1,44)
            print(col_names)
            # pred = model_basic_nn.predict_proba(X)
            # pred_df = pd.DataFrame(pred,columns=col_names)
            # pred_df.to_csv("api/predictions.csv",index = False)
            # pred_class = model_basic_nn.predict_classes(X)
            # print("pred_class",pred_class)
            pred_proba = model_basic_nn.predict_proba(X)
            print("pred_proba",pred_proba)
            print("sort",np.argsort(pred_proba)[0][::-1][0:3])
            ind = np.argsort(pred_proba)[0][::-1][0:3]
            top3_classes = OrderedDict()
            print("top3_classes",top3_classes)
            for i in ind:
                top3_classes[str(lb.classes_[i],'utf-8')] =  str(pred_proba[0][i])

            print("top3_classes",top3_classes)
            with open('top3_classes.json', 'w') as outfile:
                json.dump(top3_classes, outfile)
            print("Done")
            # c = str(lb.classes_[pred_class[0]],'utf-8')
            # print("c",c)
            # with open('pred_class.txt', 'w') as f:
            #     if len(c) != 0:
            #         f.write(c)
        except:        
            print ("Could not predict")
            with open('pred_class.txt', 'w') as f:
                f.write("Could not predict")

return_prediction()