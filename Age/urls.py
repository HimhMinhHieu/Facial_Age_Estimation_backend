from django.urls import path
from . import views

app_name='estimation'
urlpatterns = [
    path("result", views.result, name='landpage'),
    path("estimate", views.estimate, name='estimation'),
]
