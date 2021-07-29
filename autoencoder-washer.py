import os 
import numpy as np

from sklearn import metrics
from tensorflow import keras
from keras.preprocessing import image

from flask import Flask
from flask import render_template
from flask import request

from flask import Flask, redirect, url_for, request
app = Flask(__name__)

# PREDICT THE MEAN SQAURED ERROR BETWEEN INPUT IMAGE AND PREDICTION
def predImages(imgDir):
    res = {}
    for filename in os.listdir(imgDir):
        if filename.endswith(".jpg"):
            img = image.load_img(imgDir+filename, target_size=(256, 256))
            img  = image.img_to_array(img)
            pred = model_AE.predict(img.reshape(-1, 256, 256, 3))
            score = np.sqrt(metrics.mean_squared_error(pred.flatten(), img.reshape(-1, 256, 256, 3).flatten()))
            res[filename] = score
    return res

# Detect ANOMALIES by COMPARING RECONSTRUCION ERROR and THRESHOLD 
def checkAnomaly(pred,threshold=0):
    anomalies = 0
    for val in pred.values():
        if val > threshold:
            anomalies+=1
    return anomalies

# Print in Python Console
def print_res(threshold):
    print("Anamolies in training Data / False Negatives:\t{0} out of {1}"
      .format(checkAnomaly(train_pred, threshold), len(train_pred)))
    print("Anamolies in Sure / True Positives:\t{0} out of {1}"
      .format(checkAnomaly(sure_pred, threshold), len(sure_pred)))
    print("Anamolies in Sabi / True Positives:\t{0} out of {1}"
      .format(checkAnomaly(sabi_pred, threshold), len(sabi_pred)))
    print("Anamolies in Kizu / True Positives:\t{0} out of {1}"
      .format(checkAnomaly(kizu_pred, threshold), len(kizu_pred)))

@app.route("/test", methods=["POST", 'GET'])
def test():
    if request.method == 'POST':
        threshold = float(request.form["name_of_slider"])
        print("IF{0}",format(threshold))
        print_res(threshold)
        
        train_anomalies, size_train_pred = checkAnomaly(train_pred, threshold), len(train_pred)
        sure_anomalies, size_sure_pred = checkAnomaly(sure_pred, threshold), len(sure_pred)
        sabi_anomalies, size_sabi_pred = checkAnomaly(sabi_pred, threshold), len(sabi_pred)
        kizu_anomalies, size_kizu_pred = checkAnomaly(kizu_pred, threshold), len(kizu_pred)
        
        return render_template('slider.html', 
                               slider_min = slider_min, 
                               slider_max = slider_max, 
                               slider_val = threshold,
                               train_anomalies = train_anomalies,
                               size_train_pred = size_train_pred,
                               sure_anomalies = sure_anomalies,
                               size_sure_pred = size_sure_pred,
                               sabi_anomalies = sabi_anomalies,
                               size_sabi_pred = size_sabi_pred,
                               kizu_anomalies = kizu_anomalies,
                               size_kizu_pred = size_kizu_pred
                              )
    else:
        print("ELSE")
        threshold = (slider_min+slider_max+1)/2
        
        train_anomalies, size_train_pred = checkAnomaly(train_pred, threshold), len(train_pred)
        sure_anomalies, size_sure_pred = checkAnomaly(sure_pred, threshold), len(sure_pred)
        sabi_anomalies, size_sabi_pred = checkAnomaly(sabi_pred, threshold), len(sabi_pred)
        kizu_anomalies, size_kizu_pred = checkAnomaly(kizu_pred, threshold), len(kizu_pred)
        
        return render_template('slider.html', 
                               slider_min = slider_min, 
                               slider_max = slider_max, 
                               slider_val = threshold,
                               train_anomalies = train_anomalies,
                               size_train_pred = size_train_pred,
                               sure_anomalies = sure_anomalies,
                               size_sure_pred = size_sure_pred,
                               sabi_anomalies = sabi_anomalies,
                               size_sabi_pred = size_sabi_pred,
                               kizu_anomalies = kizu_anomalies,
                               size_kizu_pred = size_kizu_pred
                              )
if __name__ == '__main__':
    # LOAD TRAINED MODEL AND PREDICT THE VALUES
    model_AE = keras.models.load_model('model_AE.h5')
    train_pred = predImages(imgDir="./input_data/da_opencv/pos/")
    sure_pred = predImages(imgDir="./input_data/washer_ng/sure/")
    sabi_pred = predImages(imgDir="./input_data/washer_ng/sabi/")
    kizu_pred = predImages(imgDir="./input_data/washer_ng/kizu/")
    slider_max = max(max(train_pred.values()), max(sure_pred.values()), max(sabi_pred.values()), max(kizu_pred.values()))
    slider_min = min(min(train_pred.values()), min(sure_pred.values()), min(sabi_pred.values()), min(kizu_pred.values()))
    
    app.run()