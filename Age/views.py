from django.shortcuts import redirect, HttpResponse
from Age.models import AgeEst
from Age.serializers import AgeSer
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

import os
from Facial_Age_estimation_PyTorch import inference
# Create your views here.
def win(request):
    return HttpResponse('ok')

@api_view(['GET'])
def result(request):
    return HttpResponse('ok')

@csrf_exempt
def estimate(request):
    upload_path=os.listdir('upload/img_upload/')
    if request.method == 'POST':
    
        image = request.FILES.get('image')
        print(image)
        if os.listdir('upload/img_upload'):
            count = 1
            AgeEst.objects.all().delete()
            for index,f in enumerate(upload_path):
                if (f==upload_path[0]):
                    continue
                elif (f[index] == upload_path[count]):
                    print(f'upload:{upload_path[count]}')
                    print(f'index:{f[index]}')
                    os.remove(f'D:/AI/final/2/web/FAE-Res50/backend/upload/img_upload/{f[index]}')
                count = count + 1
        else:
            os.mkdir('img_upload')
        new= AgeEst.objects.create(image=image)
        new.save()
        result=inference.inference(f'upload/{new.image}')
        result=result[1]
        AgeEst.objects.filter(image=new.image).update(result=result)
        return redirect('http://localhost:3000/result')
    if request.method == 'GET':
        data=list(AgeEst.objects.all())
        data=AgeSer(data, many=True)
        data=data.data[0]
        print(data)
        return JsonResponse(data, safe=False)