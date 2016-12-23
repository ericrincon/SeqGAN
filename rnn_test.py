import numpy as np
from model import Generator

from tflearn.datasets import imdb
from tflearn.data_utils import pad_sequences, to_categorical

def main():
    (X_train, y_train), (X_test, y_test), _ = imdb.load_data()
    X_train = np.array(pad_sequences(X_train, maxlen=100))
    print X_train.shape
    X_test = np.array(pad_sequences(X_test, maxlen=100))

    vocab_size = X_train.max() + 1
    print 'vocab size: {}'.format(vocab_size)
    y_train = X_train.copy()
    y_test = np.array(y_test)
    cnn = Generator(10, vocab_size, 50, 100)
    cnn.train(X_train, y_train, 5)

if __name__ == '__main__':
    main()