import tensorflow as tf
from flask import Flask, request
import cv2
import numpy as np
import subprocess
from deployed_pro import *

app = Flask(__name__)


@app.route('/')
def home_endpoint():
    return 'Covid Detection Model'


@app.route('/predict')
def run_mod():
    subprocess.run('python deployed_pro.py', shell=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
