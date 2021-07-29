# washer-anomaly-detection
* Model is trained on Non-anamolous washer_ok images for anomaly detection using autoencoders. 
* OpenCV and sklearn Data augmentation is used to increase training data (washer_ok images). Refer：[DA_filters.ipynb](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/DA_filters.ipynb)と[Train Data](/https://github.com/pranavnijampurkar33/washer-anomaly-detection/tree/main/input_data/da_opencv/pos)
* Refer [autoencoder-washer.ipynb](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/autoencoder-washer.ipynb) for model training details
* Run the [autoencoder-washer.py](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/autoencoder-washer.py) code to load trained model, adjust the threshold to get optimum results
* Please follow below instructions to run [autoencoder-washer.py](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/autoencoder-washer.py)

**日本語版**
* モデルは、オートエンコーダーを使用した異常検出のために、クラスのwasher_ok画像でトレーニングされています。
* OpenCVおよびsklearnデータ拡張は、トレーニングデータ（washer_ok画像）を増やすために使用されます。**参照：** [DA_filters.ipynb](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/DA_filters.ipynb)と[Train Data](/https://github.com/pranavnijampurkar33/washer-anomaly-detection/tree/main/input_data/da_opencv/pos)
* モデルトレーニングの詳細については、[autoencoder-washer.ipynb](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/autoencoder-washer.ipynb)を参照してください。
* [autoencoder-washer.py](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/autoencoder-washer.py)コードを実行してトレーニング済みモデルを読み込み、Thresholdを調整して最適な結果を取得します。
* コードを実行するには、以下の手順を参照してください。

**Files with extention .ipynb can be seen in github with their results**
**拡張子が.ipynbのファイルは、結果とともにgithubで確認できます**


## Flask App to load trained model and Adjust the Threshold
(トレーニング済みモデルをロードしてしきい値を調整するFlaskアプリ)

Make sure you download all the files in the repository
リポジトリ内のすべてのファイルをダウンロードしてください

下記のコードを実行して「　http://localhost:5000/test　 」をブラウザで開けてください。

**Required Python libraries** os, numpy, scikit-learn, tensorflow(v2+), keras, Flask
**必要なPythonライブラリ** os, numpy, scikit-learn, tensorflow(v2+), keras, Flask

Installation commands:

    pip install numpy
    pip install Keras
    pip install scikit-learn
    pip install tensorflow
    pip install Flask

Run code using 
    
    python autoencoder-washer.py

### Result: 
![result_ex.png](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/imgs/result_ex.png)

### Model Summary:
![model_summary.png](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/imgs/model_summary.png)

### Training-Validation Loss for 300 epochs
![train_val_loss.png](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/imgs/train_val_loss.png)

### Code Flow
![Code_flow.png](https://github.com/pranavnijampurkar33/washer-anomaly-detection/blob/main/imgs/Code_flow.png)

### Other approaches taken
* Binary Classification using 3 layer CNN 
* Binary Classification using AlexNet
