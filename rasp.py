import boto3
import tempfile
import subprocess
import time

session = boto3.Session(profile_name='myprofile')
s3=session.client('s3')

def upload_to_s3(bucket_name, file_path):
    try:
        file_name = file_path.split('/')[-1]
        s3.upload_file(file_path, bucket_name, f"records/{file_name}")
        print("WAV file 업로드 성공")
    except Exception as e:
        print("WAV file 업로드 중 오류 발생: ", str(e))


def record_audio(duration, num):
    for i in range(num):
        file_name=f"record{i+1}.wav"
        command=f"arecord -D plughw:3,0 -d 2 -f cd {file_name}"
        subprocess.run(command, shell=True)
        time.sleep(duration)
        upload_to_s3('avo-data', f"./{file_name}")


record_audio(2,2)
