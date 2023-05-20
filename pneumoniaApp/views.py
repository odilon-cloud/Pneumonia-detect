from django.conf import settings
from django.core.files.storage import FileSystemStorage
#import tensorflow
import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
# Create your views here.
import os
from django.shortcuts import render,redirect
from django.template import loader
#from .form import ImageForm 
from .models import Image 
import joblib



# def Welcome(request):
#         if request.method == "POST": 
#             form=ImageForm(data=request.POST,files=request.FILES) 
#             if form.is_valid(): 
#                 form.save() 
#                 obj=form.instance 
#                 return render(request,"index.html",{"obj":obj}) 
#         else: 
#             form=ImageForm() 
#             img=Image.objects.all() 
#         return render(request,"index.html",{"img":img,"form":form}) 
#         #return render(request, 'index.html')
def Welcome(request):

    return render(request, 'index.html')

def result(request):
    if request.method == 'POST':
        patient_id = request.POST.get('patientId')
        description = request.POST.get('description')
        photo = request.FILES.get('photo')

        # saving the uplaoded photo
        fs = FileSystemStorage(location='D:/saturdayai/pneumoniaML/pneumoniaApp/testedpic')
        filename = fs.save(photo.name, photo)
        photo_url = fs.path(filename)
            #predicting
        # image_width = 150
        # image_height = 150
        # img = tf.keras.preprocessing.image.load_img(photo_url, target_size=(image_width, image_height))
        # img = tf.keras.preprocessing.image.img_to_array(img)
        # img = np.expand_dims(img, axis=0)
        # prediction = model.predict(img)
        # print(prediction)
        # img_size = 150

        # img = cv2.imread(photo_url, cv2.IMREAD_GRAYSCALE)
        # resized_img = cv2.resize(img, (img_size, img_size))
        # normalized_img = resized_img / 255
        # input_img = np.reshape(normalized_img, (1, img_size, img_size, 1))

        # # Make prediction for the single image
        # prediction = model.predict_classes(input_img)
        # predicted_label = labels[prediction[0][0]]

        img_size = 150

        img = cv2.imread(photo_url, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, (img_size, img_size))
        normalized_img = resized_img / 255
        input_img = np.reshape(normalized_img, (1, img_size, img_size, 1))
        print(input_img.shape)
        model = joblib.load('D:\saturdayai\pneumoniaML\pneumoniaApp\savedModel\model.joblib')
        # Make prediction for the single image
        prediction = model.predict(input_img)
        predicted_class = np.argmax(prediction)
        predicted_label = labels[predicted_class]

        #print("Predicted Label:", predicted_label)
    context = {
            'patient_id': patient_id,
            'description': description,
            'photo_url': photo_url,
            'result': predicted_label,
    }
    return render(request,'result.html', context)


def history(request):
    return render(request,'history.html')
