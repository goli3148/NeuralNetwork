import numpy as np 
import matplotlib.pyplot as plt 

def load_data(): 
    train_loaded = np.load(f'train_data.npz')
    test_loaded = np.load(f'test_data.npz')
    
    train_x, train_y = train_loaded['x'].T, train_loaded['y'] 
    test_x, test_y = test_loaded['x'].T, test_loaded['y'] 

    # Standarize data
    train_x = train_x / 255.
    test_x = test_x / 255.

    # Shuffle data (no effect for full batch method)
    train_indices = np.random.permutation(train_x.shape[1])
    train_x = train_x[:, train_indices]
    train_y = train_y[train_indices]

    test_indices = np.random.permutation(test_x.shape[1])
    test_x = test_x[:, test_indices]
    test_y = test_y[test_indices]

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = load_data()
print('train data matrix: ', train_x.shape)
print('test data matrix: ', test_x.shape)

print('number of data samples for each class (train): ', np.unique(train_y, return_counts=True))
print('number of data samples for each class (test): ', np.unique(test_y, return_counts=True))

# Plot some of the images 
def reshape_images(img):
    """
    reshapes the given matrix from [12288, ] to [64, 64, 3]
    """
    return np.reshape(img, (64, 64, 3))

def show_img(X, title: str = ''):
    plt.imshow(X)
    plt.axis('off')
    plt.title(f'{title}')
    plt.show()

index = 0
show_img(reshape_images(train_x[:, index]), str(train_y[index]))
index = 1003
show_img(reshape_images(train_x[:, index]), str(train_y[index]))