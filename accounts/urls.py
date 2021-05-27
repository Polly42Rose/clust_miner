from django.urls import path

from .views import SignUpView, handleSignUp

urlpatterns = [
    path('signup/', handleSignUp, name='signup'),
]