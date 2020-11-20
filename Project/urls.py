
from django.contrib import admin
from django.urls import path
from WordPredict.views import index,predict

from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$',index,name="index"),
    url(r'^posts/(?P<slug>.+?)/$',predict,name="predict")

]


