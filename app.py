from flask import Flask, send_file

import urllib.request
import joblib
import numpy as np
import librosa
import boto3 #python에서 aws s3로 접근하기 위한 라이브러리
import tempfile

app = Flask(__name__)

#ubuntu 서버의 myprofile에 미리 값들 저장해놓은 상태
session = boto3.Session(profile_name='myprofile')
s3=session.client('s3')

#모델 불러오기
model=joblib.load('./AVO-ai/model/new_model.pkl')

def predict_sound(audio_file):
  
  labels = ['crying', 'shouting']

  y, sr = librosa.load(audio_file, mono=True, duration=1)

  #mfcc값 추출
  mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=100)

  buffer = []

  for element in mfcc:
    buffer.append(np.mean(element))

  x_test = np.array([buffer])
  y_predict = model.predict(x_test)

  #결과값이 0이면 crying, 1이면 shouting으로 매핑
  label = labels[y_predict[0]]

  return label

@app.route("/")
def modeltest():
    
    bucket='avo-data'
    key='s37_0.wav'

    response=s3.get_object(Bucket=bucket, Key=key)
    wav_data=response['Body'].read()

    #wav 파일을 임시 파일로 저장하고
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(wav_data)

    #임시 파일의 경로 얻기
    soundfile=temp_file.name

    #이건 잘 불러와졌는지 확인하기 위한 print문. wav의 경우 bytes로 나오면 잘 나오는 것이다
    print(type(wav_data))

    res=predict_sound(soundfile)

    return res

@app.route("/test")
def home():
    return "EC2 Flask"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
