from parameters import par
from sagemaker import get_execution_role
import boto3
import pickle
import io
from urllib.parse import urlparse
import tempfile
import numpy as np
from PIL import Image
import cv2
from smart_open import smart_open
from s3fs.core import S3FileSystem

# Configure S3 Bucket Access
role = get_execution_role()

conn = boto3.client('s3')
s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket(par.bucket)
s3fs = S3FileSystem()

def read_img(img_location):
    try:
        obj = bucket.Object(img_location)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)
        return Image.open(file_stream)
    except:
        print("couldn't find image at" + str(img_location))
        raise TypeError("errrorr")


def imread_cv2(img_location):
    try:
        obj = bucket.Object(img_location)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)
        return cv2.imread(file_stream)
    except:
        return 0

def to_s3_npy(data: np.array, s3_uri: str):
    # s3_uri looks like f"s3://{BUCKET_NAME}/{KEY}"
    bytes_ = io.BytesIO()
    np.save(bytes_, data, allow_pickle=True)
    bytes_.seek(0)
    parsed_s3 = urlparse(s3_uri)
    conn.upload_fileobj(Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:])
    return True

def load_npy(path):
    with s3fs.open(path) as s3file:
         return np.load(s3file)
        
def load_file(path):
    with s3fs.open(path) as s3file:
         return open(s3file, 'r')

def list_contents(location):
    return conn.list_objects(Bucket=par.bucket, Prefix=location)['Contents']
    
def test_read():
    # Test access by printing out directory contents
    contents = conn.list_objects(Bucket=par.bucket, Prefix='sequences/')['Contents']
    for f in contents:
        print(f['Key'])
