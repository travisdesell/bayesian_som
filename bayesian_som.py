import numpy as np
import requests, gzip, os, hashlib

import cv2

import sys
from helper_directory.helper import helper_function

print(sys.argv)

for i in range(len(sys.argv)):
    print(f"the {i}th argument is '{sys.argv[i]}'")

if len(sys.argv) != 2:
    print(f"ERROR: usage is python3 {sys.argv[0]} <number>")

target_number = int(sys.argv[1])

print(f"running bayesian SOM for number {target_number}")

helper_function(target_number)

exit(1)

#np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(edgeitems=30, linewidth=100000)


path = './data/mnist'
if not os.path.exists(path):
    os.makedirs(path)

#fetch data
path='./data/mnist'
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"Y_test.shape: {Y_test.shape}")

for i in range(0, 10):
    print(X[i])
    print(f"image was a: {Y[i]}")
    cv2.imshow("X[i]", X[i])
    cv2.waitKey()
