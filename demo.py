import time
import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Merge, Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D
import numpy as np
from keras import backend as K




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')

spacy.load('en_vectors_web_lg')

   
# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'

# Chagne the value of verbose to 0 to avoid printing the progress statements
verbose = 1

#GET WEIGHTS



def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

def load_model_legacy(model, weight_path):
    ''' this function is used because the weights in this model
    were trained with legacy keras. New keras does not support loading these weights '''

    import h5py
    f = h5py.File(weight_path, mode='r')
    flattened_layers = model.layers

    nb_layers = f.attrs['nb_layers']

    for k in range(nb_layers):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if not weights: continue
        if len(weights[0].shape) >2: 
            # swap conv axes
            # note np.rollaxis does not work with HDF5 Dataset array
            weights[0] = np.swapaxes(weights[0],0,3)
            weights[0] = np.swapaxes(weights[0],0,2)
            weights[0] = np.swapaxes(weights[0],1,2)
        flattened_layers[k].set_weights(weights)

    f.close()

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    
    if weights_path:
        # model.load_weights(weights_path)
        load_model_legacy(model, weights_path)

    #Remove the last two layers to get the 4096D activations
    model = pop(model)
    model = pop(model)
        

    return model

image_model = VGG_16(CNN_weights_file_name)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
image_model.compile(optimizer=sgd, loss='categorical_crossentropy')


#load Weights
def VQA_MODEL():
    image_feature_size          = 4096
    word_feature_size           = 300
    number_of_LSTM              = 3
    number_of_hidden_units_LSTM = 512
    max_length_questions        = 30
    number_of_dense_layers      = 3
    number_of_hidden_units      = 1024
    activation_function         = 'tanh'
    dropout_pct                 = 0.5


    # Image model
    model_image = Sequential()
    model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

    # Language Model
    model_language = Sequential()
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
    model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

    # combined model
    model = Sequential()
    model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))

    for _ in range(number_of_dense_layers):
        model.add(Dense(number_of_hidden_units, kernel_initializer='uniform'))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_pct))

    model.add(Dense(1000))
    model.add(Activation('softmax'))

    return model

vqa_model = VQA_MODEL()
vqa_model.load_weights(VQA_weights_file_name)
vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# def get_image_model():
#     ''' Takes the CNN weights file, and returns the VGG model update 
#     with the weights. Requires the file VGG.py inside models/CNN '''
    

#     # this is standard VGG 16 without the last two layers
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     # one may experiment with "adam" optimizer, but the loss function for
#     # this kind of task is pretty standard
#     image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
#     return image_model

def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))


    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = image_model.predict(im)[0]
    return image_features

# def get_VQA_model():
#     ''' Given the VQA model and its weights, compiles and returns the model '''


    

#     vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     return vqa_model

def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    # word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    ''' accepts command line arguments for image file and the question and 
    builds the image model (VGG) and the VQA model (LSTM and MLP) 
    prints the top 5 response along with the probability of each '''
    
    image_file_name ="C:\\Users\\hp\\Pictures\\AirBrush_20160810163851.jpg"
    question = "what is the race of this guy?"
    

    
    if verbose : print("\n\n\nLoading image features ...")
    image_features = get_image_features(image_file_name, CNN_weights_file_name)

    if verbose : print("Loading question features ...")
    question_features = get_question_features(question)

    if verbose : print("Loading VQA Model ...")
    


    if verbose : print("\n\n\nPredicting result ...") 
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would 
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print("{} % {}!".format(round(y_output[0,label]*100,2), labelencoder.inverse_transform([label])[0]))
        #print (str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
