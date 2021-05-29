import numpy as np, pandas as pd, os, gc
import matplotlib.pyplot as plt, time
from PIL import Image 
import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
model = load_model('./unet',custom_objects={'dice_coef':dice_coef})

import keras

class DataGenerator(keras.utils.Sequence):
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
            self.data_path = path + 'test_images/'
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
    test = pd.read_csv(path + 'sample_submission.csv')
    test_batches = DataGenerator(test.iloc[::4],subset='test',batch_size=256)
    test_preds = model.predict_generator(test_batches,steps=1,verbose=1)
    return

#!/usr/bin/python
import sys
if __name__ == "__main__":
    if len (sys.argv) > 1:
        test_image = "./" + sys.argv[1]
        pred = launch_test(test_image)
        for i,batch in enumerate(pred):
        plt.figure(figsize=(14,50)) #20,18
        for k in range(16):
            plt.subplot(16,1,k+1)
            img = batch[0][k,]
            img = Image.fromarray(img.astype('uint8'))
            img = np.array(img)
            extra = '  has defect'
                for j in range(4):
                    msk = batch[1][k,:,:,j]
                    msk = mask2pad(msk,pad=3)
                    msk = mask2contour(msk,width=2)
                    if np.sum(msk)!=0: extra += ' '+str(j+1)
                    if j==0: # yellow
                        img[msk==1,0] = 235
                        img[msk==1,1] = 235
                    elif j==1: img[msk==1,1] = 210 # green
                    elif j==2: img[msk==1,2] = 255 # blue
                    elif j==3: # magenta
                        img[msk==1,0] = 255
                        img[msk==1,2] = 255
                plt.title(filenames[16*i+k]+extra)
                plt.axis('off')
            plt.imshow(img)
        plt.subplots_adjust(wspace=0.05)
        plt.show()



