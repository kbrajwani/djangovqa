import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .models import NSAI

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import NSAISerializer
from django.http import Http404

from django.conf import settings
from sklearn.externals import joblib


# Create your views here.
def index(request):
    return render(request, 'index.html')


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

        """
        MODEL HERE

        image_file_name = photo
        question = question

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




        
        MODEL END
        """

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
