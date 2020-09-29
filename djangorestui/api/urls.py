from django.conf.urls import url, include
from rest_framework.urlpatterns import format_suffix_patterns
from .views import *
from django.urls import path
from . import views
from django.contrib import admin
from django.conf.urls.static import static
from django.http import HttpRequest , HttpResponse,JsonResponse
from django.views.generic.base import RedirectView
from django.conf.urls.static import static

# http://127.0.0.1:8000/uploadsong/
app_name = 'api'
# point urls to specific views
urlpatterns = [
    path("", views.simple_upload, name = "homepage"),
    url("^uploadsongs/$", LoadSongsView.as_view(), name = "upload"),
    url("^uploadedsongs/$", views.uploaded_files),
    url("^classificationcode/$", views.classification_code_page),
    url("^styletransfercode/$", views.style_transfer_code_page),
    url("^musicmix/$", views.music_mix_code_page),
    url("^uploadedsongs/selectsong/$", views.select_song),
    url("^uploadedsongs/runmodel/$", views.run_model),
    url("^uploadedsongs/selectsong/runmodel/$", views.run_model),
    # url("^uploadedsongs/selectsong/$", views.run_model),
    url("^uploadedsongs/selectsong/selectplot/$", views.return_plot),
    url("^generatedmusic/$", views.generated_music_page),
    # url("^uploadedsongs/playlist/$", views.playAudioFile, name = "playsong"),
    # url("^media/",views.return_songs,name="goToPage"),
    # url("^media/$",views.return_songs_on_click,name="goToPage"),
    # url("^runmodel/$",views.blank_page,name="select_button"),
    # url('^simulationmainpage/$', views.simulationMainPage, name = 'sim'),
    # url('^simulationmainpage/simulation/$', SimulationView.as_view(), name = 'sim'), # to create/ post the json
    # url('^simulationmainpage/simulation/(?P<pk>[0-9]+)/$',SimulationDetailsView.as_view(), name="simdetails"), # to edit/delete the json
    # url('^simulationmainpage/actionUrl/$',views.select_simulation,name='goToPage'),
    # url('^simulationmainpage/actionUrl/runningSimulation/$',views.run_simulation,name='run'),
    # url('^simulationmainpage/runSimulation/$',SimulationRunView.as_view(),name='api_ping'),

    # url('^academicCertiMainPage/$',views.acadCertiPage,name='acad'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# urlpatterns = format_suffix_patterns(urlpatterns)

# admin.site.index_template = 'admin/my_custom_index.html'
# admin.autodiscover()