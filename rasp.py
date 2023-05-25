import boto3
import tempfile
import subprocess
import time

session = boto3.Session(profile_name='myprofile')
s3=session.client('s3')

def upload_to_s3(bucket_name, file_path):
    try:
        file_name = file_path.split('/')[-1]
        s3.upload_file(file_path, bucket_name, f"{file_name}")
        print("WAV file 업로드 성공")
    except Exception as e:
        print("WAV file 업로드 중 오류 발생: ", str(e))


def record_audio(duration, num):
    for i in range(num):
        file_name=f"record1.wav"
        #raspberry 재연결하면 마이크 포트번호 바뀔 때도 있으니 arecord -l로 확인해주고 수정해주는 단계 먼저 해줘야함!
        command=f"arecord -D plughw:3,0 -d 2 -f cd {file_name}"
        subprocess.run(command, shell=True)
        time.sleep(duration)
        upload_to_s3('avo-data', f"./{file_name}")


record_audio(2,2)
