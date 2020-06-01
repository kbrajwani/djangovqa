import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .models import NSAI
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import NSAISerializer
from django.http import Http404
from tensorflow.python.keras.backend import set_session
import spacy
from django.conf import settings
from sklearn.externals import joblib
import cv2

# Create your views here.
def index(request):
    return render(request, 'index.html')


def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    print("."+image_file_name)
    im = cv2.resize(cv2.imread("."+image_file_name), (224, 224))


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
    with settings.GRAPH1.as_default():
            set_session(settings.SESS)            
            image_features[0,:] = settings.IMAGE_MODEL.predict(im)[0]


    return image_features


def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    # word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    
    tokens = settings.WORD_EMBEDDINGS(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

def results(request):
    if request.method == 'POST':
        print(request.FILES)

        if request.FILES:
            photo = request.FILES["photo"]
        else:
            photo = None

        question = request.POST["question"]
        print(question)
        print(photo)

        
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, photo)
        image_file_name = default_storage.url(file_name_2)
        print(image_file_name)
            
        image_features = get_image_features(image_file_name)

        question_features = get_question_features(question)

        with settings.GRAPH1.as_default():
            set_session(settings.SESS)  
            y_output = settings.VQA_MODEL.predict([question_features, image_features])
        y_sort_index = np.argsort(y_output)

        labelencoder =  settings.LABELENCODER
        for label in reversed(y_sort_index[0,-5:]):
            print("{} % {}!".format(round(y_output[0,label]*100,2), labelencoder.inverse_transform([label])[0]))


        NSAI(photo=photo, question=question).save()

        context = {
            "question": question
        }

        # context = serializers.serialize('json',context)
        print(context)

        return JsonResponse(context)

    print("No cont")

    return render(request, 'index.html')


def main():
    ''' accepts command line arguments for image file and the question and
    builds the image model (VGG) and the VQA model (LSTM and MLP)
    prints the top 5 response along with the probability of each '''

    image_file_name = "C:\\Users\\pc\\Downloads\\brian.jpg"
    question = "what is the race of this guy?"

    if settings.verbose: print("\n\n\nLoading image features ...")
    image_features = settings.get_image_features(image_file_name, settings.CNN_weights_file_name)

    if settings.verbose: print("Loading question features ...")
    question_features = settings.get_question_features(question)

    if settings.verbose: print("Loading VQA Model ...")
    vqa_model = settings.get_VQA_model()

    if settings.verbose: print("\n\n\nPredicting result ...")
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    labelencoder = joblib.load(settings.label_encoder_file_name)
    for label in reversed(y_sort_index[0, -5:]):
        print("{} % {}!".format(round(y_output[0, label] * 100, 2), labelencoder.inverse_transform([label])[0]))
        # print (str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))
