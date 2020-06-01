from django.urls import path,include
from home import views

from django.views.decorators.csrf import csrf_exempt

from rest_framework.urlpatterns import  format_suffix_patterns

urlpatterns = [
    path('',views.index,name='index'),
    path('results',csrf_exempt(views.results),name='results'),
]

urlpatterns = format_suffix_patterns(urlpatterns)