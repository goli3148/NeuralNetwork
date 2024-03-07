import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os 
import imageio

def get_data(
    mean_list: np.ndarray,
    cov_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    mean_list: 
        a [2, 2] numpy array where [0][0] and [0][1] are the x and y coordinates
        for the mean of first class, and [1][0] and [1][1] are x, y coordiantes 
        for the mean of second class

    cov_list: 
        a [2, 2, 2] numpy array where [0, :, :] is the covariance matrix of the first class
        and [1, :, :] is the covariance matrix of the second class
    
    Returns: 
        X: a [N, 2] numpy array where N is the number of samples and 
           X[i, :] is the coordinates for the i-th sample
        d: 
            a [N] numpy array where d[i] is the class label for i-th sample         
    """

    X = []
    d = []
    for c in range(2):
        X.append(np.random.multivariate_normal(mean_list[c], cov_list[c], 50))
        if c == 0: 
            d.append((np.ones(X[c].shape[0], dtype=np.int32) * -1))
        else:
            d.append((np.ones(X[c].shape[0], dtype=np.int32) * 1))

    X = np.vstack(X)
    d = np.hstack(d)
    
    return X, d

def plot_all(X: np.ndarray, d: np.ndarray, W: np.ndarray, 
            epoch: int, path: str):
    """
    X: a [N, 2] numpy array where N is the number of samples and 
           X[i, :] is the coordinates for the i-th sample
    d: 
        a [N] numpy array where d[i] is the class label for i-th sample   
    
    W: a [3, 1] numpy array containing weights of the perceptron model
    epoch: epoch number
    path: the path to save the plot 
    """
    make_dir(f'{path}')
    graph(my_formula, W, [-10, 10])
    
    plot_data(X.T, d)

    plt.xlim(np.min(X.T[:, 0])-1, np.max(X.T[:, 0])+1)
    plt.ylim(np.min(X.T[:, 1])-1, np.max(X.T[:, 1])+1)
    
    plt.legend(['0', '1'], loc='upper right')
    plt.text(0, 0, "epoch=%d" % epoch)
    plt.savefig(f'{path}/epoch=%d.jpg' % epoch)
    plt.clf()

def save_gif(num_frames, path_of_frames, name):
    make_dir(f'{path_of_frames}')
    frames = []
    for t in range(num_frames):
        image = imageio.imread(f'{path_of_frames}/epoch=%d.jpg' % t)
        frames.append(image)

    imageio.mimsave(f'{path_of_frames}/{name}.gif',  # output gif
                    frames,  # array of input frames
                    fps=2,  # optional: frames per second
                    loop=1)
   
def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

def my_formula(W, x):
    return (-W[0]/W[1])*x + (-W[2]/W[1])

def graph(formula, W, x_range, c='black'):  
    x = np.linspace(start=x_range[0],stop=x_range[1],num=x_range[1]-x_range[0]+1)
    y = formula(W, x)  
    plt.plot(x, y, c=c)  

def show_data(X, d):
    """
    X: [N, 2] array of samples to plot 
       X[i, 0] and X[i, 1] corresponds to x and y coordiante of the i-th sample
    d: [N, ] label of samples
    """
    plt.scatter(X[:, 0], X[:, 1], c=d, s=12)
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_data(X, d):
    plt.scatter(X[:, 0], X[:, 1], c=d, s=8)
    plt.axis('equal')
    # plt.show()



