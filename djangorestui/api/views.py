from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpRequest , HttpResponse,JsonResponse, HttpResponseRedirect
from .getDatabaseTables import getdataBaseTable
from rest_framework import generics
from .serializers import *
from .models import LoadSongs
import os
import json
import subprocess
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# from .dl_model import *
# from urllib import urllib
# from django.shortcuts import get_list_or_404, get_object_or_404
def make_names(f_name):
    f_name = "_".join(f_name.split(" "))
    return f_name

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        for i in os.listdir('../'):
          if i == filename:
            src = './djangorestui/api/media/' + i
            dst = './djangorestui/api/media/' + make_names(i)
            os.rename(src,dst)

        return render(request, 'main/sample_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'main/sample_upload.html')

def uploaded_files(request):
    file_names = []
    for i in os.listdir('api/media/'):
      if os.path.isfile('api/media/{}'.format(i)):
        file_names.append(i)
    file_m_time = {}
    # for i in os.listdir('api/media'):
    #     file_m_time[i] = os.path.getmtime('api/media/{}'.format(i))

    return render(
                request = request,
                template_name ='main/uploadedsongs.html',
                context = {
                "songs": file_names})

def blank_page(request):
  if(request.GET.get('run_model')):
    return render(
                request = request,
                template_name ='main/blank.html')

def classification_code_page(request):
    return render(
                request = request,
                template_name ='main/Soundclassification-Chaithanya-large-data-final.html')

def style_transfer_code_page(request):
    return render(
                request = request,
                template_name ='main/Style_Transfer.html')

def music_mix_code_page(request):
      return render(
                request = request,
                template_name ='main/Music_Generation_test_TimeDistributed1-Classical.html')
def augmentation_code_page(request):
      return render(
                request = request,
                template_name ='main/music+data+augmentation.html')

def generated_music_page(request):
    file_names = []
    for i in os.listdir('api/media/generated_music/'):
      if os.path.isfile('api/media/generated_music/{}'.format(i)):
        file_names.append(i)
    return render(
                request = request,
                template_name ='main/generatedmusic.html',
                context = {
                "songs": file_names})


def select_song(request):
  if(request.GET.get('select_song')):
    models = []
    for i in os.listdir('./api/dl_models'):
        if '.json' in i:
            models.append(i)
    return render(
                        request = request,
                        template_name ='main/selectsong.html',
                        context = {"songs":request.GET.get('textbox_1') ,
                        # "models":models
                        })
  

# def uploaded_songs(request):
#   if (request.GET.get('media_button')):
#       return render(
#                   request = request,
#                   template_name ='main/uploadedsongs.html',
#                   context = {
#                   "songs": LoadSongs.objects.all().order_by('id').reverse()}) # display in reverse order
    

class LoadSongsView(generics.ListCreateAPIView):
    """This class is used to run the simulation for a post request"""
    queryset = LoadSongs.objects.all()
    serializer_class = SongsInputSerializer

    def perform_create(self, serializer):
        """Save the post data when creating a new simulation."""
        # this does not return a response
        serializer.save() # save to api_simulationruns table

def load_song(song_path,song_name):
    data_path = '{}/{}'.format(song_path,song_name)
    print(data_path)
    y,sr = librosa.load(data_path)
    return (y,sr)

def run_model(request):
  if(request.GET.get('run_model')):
    model_name = request.GET.get('model_box')#.split(" ")[0]
    song_name = request.GET.get('song_box')#.split(" ")
    table = {"model_name" : model_name,
    "song_name" : song_name}
    print("table",table)
    with open('model_input.json', 'w') as outfile:
        json.dump(table, outfile)
    subprocess.run('python ./api/dl_model.py ./model_input.json', shell = True)
    # preds_df = pd.read_csv('./api/predictions.csv')
    with open('../djangorestui/top3_classes.json') as f:
      c  = json.load(f)
    k = list(c.keys())

    v = list(c.values())
    return render(request = request,
        template_name ='main/predictions.html',
        context = {
        "song_name": song_name,
        "class_1" : k[0],
        "prob_class_1" : v[0],
        "class_2" : k[1],
        "prob_class_2" : v[1],
        "class_3" : k[2],
        "prob_class_3" : v[2],
        })

def return_plot(request):
  if(request.GET.get('waveplot')) or (request.GET.get('spectogram')):
    song_path = "code/djangorestui/api/media/"
    song_name = request.GET.get('song_box')
    y,sr = load_song(song_path, song_name)
    print(y)
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.figure(figsize=(14, 5),)
    librosa.display.waveplot(y, sr=sr,color='r',)
    fig.savefig('img/{}_waveplot.png'.format(genre))

    return render(request = request,
          template_name ='main/plots.html',
          context = {
        "plot_name": request.GET.get('song_box'),
        })