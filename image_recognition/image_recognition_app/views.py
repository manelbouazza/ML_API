from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseServerError
import cv2
#import tensorflow as tf
from tensorflow import keras
from keras import models
#from tensorflow.keras.models import load_model
import os
import numpy as np

class_names = ['Plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_form(request):
    return render(request, 'image_recognition/predict_form.html')

@csrf_exempt
def predict_image(request):
    
    try:
        
        mdl = keras.models.load_model(r"C:\Users\Mon-PC\Documents\soap_projects\image_recognition\image_recognition_app\models\image_classifier.model")
        
        if request.method == 'POST':
            image_file = request.FILES.get('image')

            if image_file:
                img = cv2.imdecode(np.frombuffer(image_file.read(), dtype=np.uint8), 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8)

                prediction = mdl.predict(np.array([img]) / 255.0)  # Normalize to [0, 1]
                index = np.argmax(prediction)
                result = {'prediction': class_names[index]}
                return JsonResponse(result)
            else:
                return JsonResponse({'error': 'No image provided'})
        else:
            return JsonResponse({'error': 'Invalid request method'})
    except Exception as e:
        print(e)  
        return HttpResponseServerError(f"Internal Server Error: {str(e)}")
