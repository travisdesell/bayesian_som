#good tutorial on setting up venv with packages 
#https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
import numpy as np
import requests, gzip, os, hashlib

import cv2

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

#goal here is to create a sampling dristribution for each pixel, and due to the central limit theorem these distributions will #converge on normal.  Then using the 50th percentile of each sampling distribution to estimate the mean, we create a mean
#vector and calculate the covariance matrix between each pixels sampling distribution.  We then use this mean vector and 
#covariance matrix to define a multivariate normal distribution to sample from, and reconstruct the image matrix of 28X28 #pixels
#NOTE
#take notice of the iterations parameter in the mean_cov() function, it is currently set at 100 to save time and compute power
#iterations of 10000 took my macbook pro with i7 and 32GB RAM 45 minutes to run.  
#FURTHER PLANS
#Instead of a single sample being taken from the mulivariate distribution, many samples should be taken and then averaged to 
#see if the image becomes clearer

def list_of_lists(n):
    """n = number of empty lists you want in list"""
    lofl=[]
    for i in range(n):
        lofl.append([]);
    return lofl

def pixel_list_creator(X):
    """takes all the images and creates 1d arrays of each pixel"""
    pixel_list=list_of_lists(784)
    for i in range(len(X)):
        x=np.reshape(X[i],-1)
        for j in range(len(x)):
            y=x[j]
            pixel_list[j].append(y)
    pixel_list=[np.array(x) for x in pixel_list]
    return pixel_list

def sampling_dist(arr,iterations):
    """iterations: how many samples you want to draw for sampling distribution.
       if you increase the number of samples you decrease variance of distribution."""
    """takes an array and bootstraps a sampling distribution."""
    """returns sampling distribution.  Iterations"""
    distribution=[]
    #change range for higher fidelity sampling distribution, but takes more compute
    for i in range(iterations):
        sample = np.random.choice(arr,size=len(arr),replace=True)
        xbar = np.mean(sample)
        distribution.append(xbar)
    distribution = np.array(distribution)
    return distribution

def sample_mean(arr):
    """returns 50th percentile of sampling distribution"""
    percent_50th=np.percentile(arr,50)
    return percent_50th

def mean_vector(arr_list):
    """takes a list of pixel arrays and returns the mean vector"""
    """for multivariate normal of sampling distributions for each pixel"""
    mu = []
    for i in range(len(arr_list)):
        percentile = sample_mean(arr_list[i])
        mu.append(percentile)
    mu=np.array(mu)
    return mu

def cov_matrix(arr_list):
    """returns the covariance matrix between pixel sampling dristributions"""
    arr_list = [np.sort(x) for x in arr_list]
    x = np.vstack([arr_list])
    cov=np.cov(x)
    return cov

def mean_cov(X,iterations):
    """Iterations: see sampling_dist() function above for details"""
    """Returns mean vector and covariance matrix for bootstrpped multivariate normal"""
    pixel_list = pixel_list_creator(X)
    sampling_dist_list = [sampling_dist(x,iterations) for x in pixel_list]
    mu_vector = mean_vector(sampling_dist_list)
    cov = cov_matrix(sampling_dist_list)
    return mu_vector, cov

def multivariate_sample(mu,covariance):
    """extracts random sample from multivariate distribution defined by specific mean vector"""
    """and covariance matrix."""
    x=np.random.multivariate_normal(mu,covariance)
    x=np.rint(x)
    x=x.astype(int)
    x=np.reshape(x,(28,28))
    return x

test=mean_cov(X,100)
mu=test[0]
covariance=test[1]
final=multivariate_sample(mu, covariance)
print(final)