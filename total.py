from __future__ import print_function
from flask import Flask, send_file

import urllib.request
import joblib
import numpy as np
import librosa
import boto3
import tempfile
import time
import json
import requests

app = Flask(__name__)

#aws-cli 사용해서 myprofile로 AWS KEY들이 저장되어있는 상태
session = boto3.Session(profile_name='myprofile')
s3=session.client('s3')
model=joblib.load('./AVO-ai/model/final_model.pkl')

def read_json(file_path):
        with open(file_path, 'r') as file:
            data=json.load(file)
        return data

def process_transcript(transcript):
        if transcript == "엄마" or transcript == "아빠":
            return "calling"
        else:
            return "nothing"

def aws_transcribe():
    transcribe = boto3.client('transcribe', region_name='ap-northeast-2')
    timestamp=str(int(time.time()))
    job_name="avoCalling"+timestamp
    job_uri="s3://avo-data/record1.wav"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='ko-KR'
        )
    while True:
        status=transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(5)

    transcription_result_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    output_file_path="output.json"

    response=requests.get(transcription_result_uri)


    if response.status_code ==200:
        with open(output_file_path, "wb") as output_file:
            output_file.write(response.content)
        print("json 결과 다운로드 완료")
        data=read_json("./output.json")
        result=data["results"]["transcripts"][0]["transcript"]
        end=process_transcript(result)
        print(end)
        return end
    else:
        print("json 결과 다운로드 실패")


def predict_sound(audio_file):
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    labels = ['crying', 'shouting']
    y, sr = librosa.load(audio_file, mono=True, duration=2)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    padded_mfcc = pad2d(mfcc, 500)

    buffer = []

    for element in padded_mfcc:
        buffer.append(np.mean(element))
    x_test = np.array([buffer])

    y_predict = model.predict(x_test)
    label = labels[y_predict[0]]
    print(label)
    y_predict = model.predict_proba(x_test)
    confidence = y_predict[0][y_predict[0].argmax()]

    print(confidence)
    if confidence < 0.9:
        label = "nothing"

    return label

def send(body):
    data = {
        "token": "hi",
        "title": "아이가 울고있어요",
        "body": body
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post("http://52.78.239.63:8080/fcm", data=json.dumps(data), headers=headers)
    print(response.text)


@app.route("/classify", methods=['GET'])
def modeltest():

    res=aws_transcribe()
    if res == "nothing":
        bucket='avo-data'
        #key='s37_0.wav'
        key='record1.wav'

        response=s3.get_object(Bucket=bucket, Key=key)
        wav_data=response['Body'].read()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(wav_data)

        soundfile=temp_file.name
        print(type(wav_data))

        res=predict_sound(soundfile)

    send(res)
    return res


@app.route("/test")
def home():
    return "EC2 Flask"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
