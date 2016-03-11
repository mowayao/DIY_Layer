from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers.core import Activation,Dense,Layer,Reshape,Flatten
from keras import backend as K
import h5py
import numpy as np 

class SpatialSoftmax(Layer):
    
    
    @property
    def output_shape(self):
        return (self.input_shape[0],self.input_shape[1]*2)


    def get_output(self,train):
        x = self.get_input()
        x = x-K.max(x)
        exp_x = K.exp(x)
        all_sum = K.sum(exp_x,axis=(2,3),keepdims=True)
        COL = K.sum(exp_x,axis=3,keepdims=True)
        ROW = K.sum(exp_x,axis=2,keepdims=True)

        prob_col = K.mean(COL/all_sum,axis=3)
        prob_row = K.mean(ROW/all_sum,axis=2)

        loc = K.variable(range(109))

        ret1 = K.dot(prob_row,loc)

        ret2 = K.dot(prob_col,loc)
        #ret2_sq = K.squeeze(ret2_dot,axis=3)
        #ret2 = K.mean(ret2_dot,axis=2)

        return K.concatenate((ret1,ret2),axis=1)





def build_model():

    model = Sequential()
    model.add(Convolution2D(64,7,7,subsample=(2,2),input_shape=(4,240,240)))

    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5))

    model.add(Activation('relu'))
    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))
    model.add(SpatialSoftmax())
    model.add(Flatten())
    model.add(Dense(3600))
    
    #model.add(Reshape((60,60)))
    model.compile(optimizer='sgd',loss='mse')


    return model


def save_model():

    pass

def train_model():


    #model = build_model()
    F = h5py.File('train.h5','r')
    train_x = F['train'][:]/255
    train_y = F['train_o'][:]/255
    F.flush()
    print train_x.shape
    print train_y.shape
    model = build_model()

    F = h5py.File('test.h5','r')
    test_x = F['test'][:]/255
    test_y = F['test_o'][:]
    F.flush()
    F.close()
    print "....fitting"
    model.fit(train_x,train_y,batch_size=1, nb_epoch=1000, verbose=1, shuffle=False)

def test(X,model):

    y = model.predict_on_batch(X)





train_model()
test
