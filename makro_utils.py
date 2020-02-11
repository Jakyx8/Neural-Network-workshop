import struct as st
import numpy as np
import math

def load_MNIST():
    #Load dataset and transform them into input vectors
    train_file = open('/home/alobar/jupyter/Makro/Data/MNIST dataset/train-images-idx3-ubyte', 'rb')
    train_file_labels = open('/home/alobar/jupyter/Makro/Data/MNIST dataset/train-labels-idx1-ubyte', 'rb')
    test_file = open('/home/alobar/jupyter/Makro/Data/MNIST dataset/t10k-images-idx3-ubyte', 'rb')
    test_file_labels = open('/home/alobar/jupyter/Makro/Data/MNIST dataset/t10k-labels-idx1-ubyte', 'rb')

    #Vectorizing training images
    train_file.seek(0)
    magic_training = st.unpack('>4B',train_file.read(4))
    img_num = st.unpack('>I',train_file.read(4))[0]
    rows = st.unpack('>I',train_file.read(4))[0]
    columns = st.unpack('>I',train_file.read(4))[0]
    x_train = np.zeros((img_num,rows,columns))
    train_file.seek(16)
    x_train = np.asarray(st.unpack('>' + 'B'*img_num*rows*columns, train_file.read(img_num*rows*columns))).reshape(img_num, rows, columns)

    #Vectorizing training labels
    train_file_labels.seek(0)
    magic_training_label = st.unpack('>4B', train_file_labels.read(4))
    num_labels = st.unpack('>I', train_file_labels.read(4))[0]
    y_train = np.zeros((num_labels, 1))
    y_train = np.asarray(st.unpack('>' + 'B'*num_labels, train_file_labels.read(num_labels)))

    #Vectorizing test set images
    test_file.seek(0)
    magic_training = st.unpack('>4B',test_file.read(4))
    img_num = st.unpack('>I',test_file.read(4))[0]
    rows = st.unpack('>I',test_file.read(4))[0]
    columns = st.unpack('>I',test_file.read(4))[0]
    x_test = np.zeros((img_num,rows,columns))
    test_file.seek(16)
    x_test = np.asarray(st.unpack('>' + 'B'*img_num*rows*columns, test_file.read(img_num*rows*columns))).reshape(img_num, rows, columns)

    #Vectorizing test set labels
    test_file_labels .seek(0)
    magic_training_label = st.unpack('>4B', test_file_labels .read(4))
    num_labels = st.unpack('>I', test_file_labels .read(4))[0]
    y_test = np.zeros((num_labels, 1))
    y_test = np.asarray(st.unpack('>' + 'B'*num_labels, test_file_labels.read(num_labels)))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    return x_train, y_train, x_test, y_test


def mini_batches(x_train, y_train, mini_batch_size):
    
    m = x_train.shape[0]
    num_of_complete_mini_batches = math.floor(m/(mini_batch_size))
    mini_batches = []
    
    for k in range(0, num_of_complete_mini_batches):
        mini_batch_x = x_train[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_y= y_train[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
   
    if m % mini_batch_size != 0:
        mini_batch_x = train_x[:, num_of_complete_mini_batches * mini_batch_size : m]
        mini_batch_y = train_y[:, num_of_complete_mini_batches * mini_batch_size : m]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

def mini_batches_x_only(x_train, mini_batch_size):
    
    m = x_train.shape[0]
    num_of_complete_mini_batches = math.floor(m/(mini_batch_size))
    mini_batches = []
    
    for k in range(0, num_of_complete_mini_batches):
        mini_batch_x = x_train[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :]
        mini_batch = (mini_batch_x)
        mini_batches.append(mini_batch)
   
    if m % mini_batch_size != 0:
        mini_batch_x = x_train[num_of_complete_mini_batches * mini_batch_size : m, :, :]
        mini_batch = (mini_batch_x)
        mini_batches.append(mini_batch)
    
    return mini_batches
    