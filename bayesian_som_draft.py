#good tutorial on setting up venv with packages 
#https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
import numpy as np
import requests, gzip, os, hashlib

import cv2
import sys

import matplotlib.pyplot as plt

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

def list_of_lists(n):
    """n = number of empty lists you want in list"""
    lofl=[]
    for i in range(n):
        lofl.append([]);
    return lofl

#Filter into bins for each label  
X_list=list_of_lists(10)
for i in range(len(X)):
    if Y[i]==0:
        X_list[0].append(X[i])
    if Y[i]==1:
        X_list[1].append(X[i])
    if Y[i]==2:
        X_list[2].append(X[i])
    if Y[i]==3:
        X_list[3].append(X[i])
    if Y[i]==4:
        X_list[4].append(X[i])
    if Y[i]==5:
        X_list[5].append(X[i])
    if Y[i]==6:
        X_list[6].append(X[i])
    if Y[i]==7:
        X_list[7].append(X[i])
    if Y[i]==8:
        X_list[8].append(X[i])
    if Y[i]==9:
        X_list[9].append(X[i])
        
#these are the list of labels given at the command line
label_list=sys.argv[1:len(sys.argv)]

#goal here is to create a sampling dristribution for each pixel, and due to the central limit theorem these distributions will #converge on normal.  Then using the 50th percentile of each sampling distribution to estimate the mean, we create a mean
#vector and calculate the covariance matrix between each pixels sampling distribution.  We then use this mean vector and 
#covariance matrix to define a multivariate normal distribution to sample from, and reconstruct the image matrix of 28X28 #pixels
#NOTE
#take notice of the iterations parameter in the mean_cov() function, it is currently set at 100 to save time and compute power
#iterations of 10000 took my macbook pro with i7 and 32GB RAM 45 minutes to run.  
#FURTHER PLANS
#Instead of a single sample being taken from the mulivariate distribution, many samples should be taken and then averaged to 
#see if the image becomes clearer

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

def mean_vector_simple(arr_list):
    mu=[]
    for i in range(len(arr_list)):
        mu.append(np.mean(arr_list[i]))
    mu=np.array(mu)
    return mu

def cov_matrix(arr_list):
    """returns the covariance matrix between pixel sampling dristributions"""
   # arr_list = [np.sort(x) for x in arr_list]
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

def final_xbar(mu, covariance, sample_size):
    """repeatedly samples defined multivariate normal distribution and provides mean of all samples"""
    test=multivariate_sample(mu,covariance)
    for i in range(sample_size):
        x = multivariate_sample(mu,covariance)
        test=test+x
    test=test/(sample_size+1)
    test=np.rint(test)
    test=test.astype(int)
    return test


def create_image_list(label_list):
    """this will create a list of lists of n_images for each label provided on the command line"""
    image_list=[]
    for i in range(len(label_list)):
        label=label_list[i]
        data=X_list[int(label)]
        test=mean_cov(data,100)
        mu=test[0]
        covariance=test[1]
        for j in range(10):
            image=multivariate_sample(mu,covariance)
            image_list.append(image)
    return image_list

def create_simple_image_list(label_list):
    image_list=[]
    for i in range(len(label_list)):
        label=label_list[i]
        data=X_list[int(label)]
        pixel_list=pixel_list_creator(data)
        mu=mean_vector_simple(pixel_list)
        covariance=cov_matrix(pixel_list)
        for j in range(10):
            image=multivariate_sample(mu,covariance)
            image_list.append(image)
    return image_list    

def create_image_LISTS(label_list):
    """this will create a list of lists of n_images for each label provided on the command line"""
    image_lists=list_of_lists(len(label_list))
    for i in range(len(label_list)):
        label=label_list[i]
        data=X_list[int(label)]
        test=mean_cov(data,100)
        mu=test[0]
        covariance=test[1]
        for j in range(100):
            image=multivariate_sample(mu,covariance)
            image_lists[i].append(image)
    return image_lists

def create_simple_image_LISTS(label_list):
    image_lists=list_of_lists(len(label_list))
    for i in range(len(label_list)):
        label=label_list[i]
        data=X_list[int(label)]
        pixel_list=pixel_list_creator(data)
        mu=mean_vector_simple(pixel_list)
        covariance=cov_matrix(pixel_list)
        for j in range(100):
            image=multivariate_sample(mu,covariance)
            image_lists[i].append(image)
    return image_lists
    

def display_save_labels(label_list):
    """displays a grid of 2X5 images for each label in image_list"""
    """will also save png file bayesian_som.png in pwd for better viewing"""
    image_list=create_image_list(label_list)
    num_row = 2 * len(label_list)
    num_col = 5 
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(image_list)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image_list[i], cmap='gray')
    plt.tight_layout()
    plt.savefig("bayesian_som")
    plt.show()

def display_save_labels_simple(label_list):
    image_list=create_simple_image_list(label_list)
    num_row = 2 * len(label_list)
    num_col = 5 
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(image_list)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image_list[i], cmap='gray')
    plt.tight_layout()
    plt.savefig("bayesian_som_simple")
    plt.show()   

def  display_save_difference(label_list=[x for x in range(10)]): 
    """this function will create first create multivariate distributions for each label"""
    """on the label list.  It will then sample each distribution 100 times and build a list"""
    """of every pairwise difference for a given label.  It will then take the average"""
    """for each pixel and then print out the overall difference between images in matrix form"""
   
    #this will give you a list of lists, one for each label, each list containing 100 samples from labels multivariate normal dist 
    image_lists = create_image_LISTS(label_list)
    difference_lists=list_of_lists(len(label_list))
    for i in range(len(label_list)):
        for j in range(len(image_lists[i])-1):
            for k in range(j+1,len(image_lists[i])):
                difference=image_lists[i][k] - image_lists[i][j]
                difference=np.abs(difference)
                difference_lists[i].append(difference)
    difference_means=[]
    for i in range(len(difference_lists)):
        pixel_means=[]
        test=pixel_list_creator(difference_lists[i])
        for j in range(len(test)):
            x=np.mean(np.array(test[j]))
            pixel_means.append(x)
        pixel_means=np.rint(pixel_means)
        pixel_means=pixel_means.astype(int)
        pixel_means=np.reshape(pixel_means,(28,28))
        difference_means.append(pixel_means) 
    num_row = 2
    num_col = 5# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(difference_means)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(difference_means[i], cmap='gray')
    plt.tight_layout()
    plt.savefig("bayesian_som_differences")
    plt.show()        

def  display_save_simple_difference(label_list=[x for x in range(10)]): 
    image_lists = create_simple_image_LISTS(label_list)
    difference_lists=list_of_lists(len(label_list))
    for i in range(len(label_list)):
        for j in range(len(image_lists[i])-1):
            for k in range(j+1,len(image_lists[i])):
                difference=image_lists[i][k] - image_lists[i][j]
                difference=np.abs(difference)
                difference_lists[i].append(difference)
    difference_means=[]
    for i in range(len(difference_lists)):
        pixel_means=[]
        test=pixel_list_creator(difference_lists[i])
        for j in range(len(test)):
            x=np.mean(np.array(test[j]))
            pixel_means.append(x)
        pixel_means=np.rint(pixel_means)
        pixel_means=pixel_means.astype(int)
        pixel_means=np.reshape(pixel_means,(28,28))
        difference_means.append(pixel_means) 
    num_row = 2 
    num_col = 5# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(difference_means)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(difference_means[i], cmap='gray')
    plt.tight_layout()
    plt.savefig("bayesian_som_simple_differences")
    plt.show() 

def running_covariance(label_list):
    #first loop through label list
    for label in label_list:
        image_list=X_list[int(label)]
        #this will help limit pictures to 100ish
        modulo=int(len(image)list/100) 
        #initiate pixel list with the first image
        running_pixel_list=pixel_list_creator([image_list[0]])
        #make pixel list for each image and concat to running_pixel_list
        for image in image_list:
            image_for_print=[]
            pixel_list=pixel_list_creator([image])
            running_pixel_list = np.concatenate((running_pixel_list,pixel_list), axis=1)
            #this condition will only calculate cov and mean so that about
            #100 picks are printed out in the end
            if (len(running_pixel_list[0]) % modulo == 0)):
                
      #figure out how to limit printouts 
                
                
            
        
        
    

