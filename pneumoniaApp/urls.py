from django.urls import path
from . import views

urlpatterns = [
    path('',views.Welcome, name=""),
    path('result',views.result, name="result"),
    path('history',views.history, name="history")
]