from django.shortcuts import render
from django.http import JsonResponse
from django.template.defaulttags import csrf_token
from django.views.decorators.csrf import ensure_csrf_cookie

import WordPredict.pmodel as PModel
# Create your views here.


def index(request):

    return render(request,'homepage.html')

def predict(request,slug):
    print(slug)

    predicition = PModel.score_output(slug, fact = 0.4)
    print(predicition)
    data = {}
    return JsonResponse(predicition,safe=False)