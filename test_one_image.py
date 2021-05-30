import numpy as np, pandas as pd, os, gc
import matplotlib.pyplot as plt, time
from PIL import Image 
import warnings
import csv
import cv2

warnings.filterwarnings("ignore")
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
import os


from tensorflow.keras import backend as K

def rle2maskResize(rle):
    """
        Convert run length encoding to mask
        """
    if (pd.isnull(rle))|(rle==''):
        return np.zeros((128,800) ,dtype=np.uint8)
    
    height= 256
    width = 1600
    mask= np.zeros( width*height ,dtype=np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    
    return mask.reshape( (height,width), order='F' )[::2,::2]

def mask2pad(mask, pad=2):
    """
        Enlarge Mask to include more space around the defect
        """
    w = mask.shape[1]
    h = mask.shape[0]
    
    # MASK UP
    for k in range(1,pad,2):
        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK DOWN
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK LEFT
    for k in range(1,pad,2):
        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)
        mask = np.logical_or(mask,temp)
    # MASK RIGHT
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)
        mask = np.logical_or(mask,temp)

    return mask

def mask2contour(mask, width=3):
    """
        Convert mask to its contour
        """
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)

def dice_coef(y_true, y_pred, smooth=1):
    """
        Compute Dice Coefficient
        """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, preprocess=None, info={}):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = 'test_images/'
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index): 
        X = np.empty((self.batch_size,128,800,3),dtype=np.float32)
        y = np.empty((self.batch_size,128,800,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((800,128))
            if self.subset == 'train': 
                for j in range(4):
                    y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])
        if self.preprocess!=None: X = self.preprocess(X)
        if self.subset == 'train': return X, y
        else: return X

# PREDICT 1 BATCH TEST DATASET
def launch_test(image):
    model = load_model('UNET.h5',custom_objects={'dice_coef':dice_coef})
        #with open('sample_submission.csv', 'w', newline='') as csvfile:
        #csvfile.write('ImageId, EncodedPixels, ClassId\n')
        #csvfile.write("{}, 1 409600, 0".format(image))
    test = pd.read_csv('sample_submission.csv')
    

    #test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
    
    #test_batches = DataGenerator(test.iloc[::4],subset='test',batch_size=32)
    test_preds = model.evaluate_generator(test_dataloader)
    #predict_generator(test_batches,steps=1,verbose=1)
    return test_preds

#!/usr/bin/python
import sys
if __name__ == "__main__":
    if len (sys.argv) > 1:
        test_image = sys.argv[1]
        model = load_model('UNET.h5',custom_objects={'dice_coef':dice_coef})

        # PREDICT 1 BATCH TEST DATASET
        test = pd.read_csv('sample_submission.csv')
        # test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])
        test_batches = DataGenerator(test[:1],subset='test',batch_size=1)
        test_preds = model.predict_generator(test_batches,steps=1,verbose=1)

        X = np.empty((1,128,800,3),dtype=np.float32)
        X[0,] = Image.open('test_images/' + sys.argv[1]).resize((800,128))
        y_pred = model.predict(X)

        msk = test_preds[0][:, :, 0]
        
        plt.imshow(msk)
        plt.show()
