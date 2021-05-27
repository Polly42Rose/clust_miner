from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name="home"),
    path('auth', views.auth, name="auth"),
    path('about', views.about, name="about"),
    path('add-log', views.add_log, name="add-log"),
    path(r'^log/(?P<pk>\d+)$', views.LogDetailView.as_view(), name='log-detail'),
    path(r'^delete/(?P<pk>\d+)/$', views.delete_log, name='delete_view'),
    path('run/(?P<pk>\d+)/$', views.run, name='log'),
    path(r'run/(?P<pk>\d+)/download$', views.download, name="model"),
]
#Add Django site authentication urls (for login, logout, password management)
urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/', include('accounts.urls')),
]