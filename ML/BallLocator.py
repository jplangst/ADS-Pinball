import cv2 as cv
import numpy as np
import time

import tensorflow as tf

print("CNN Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#tf.config.run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, Permute, Activation

#Define the neural network
def get_unet_conv_part(n_ch, inp, chPos):
    conv = Conv2D(n_ch, (3, 3), activation='relu', padding='same',data_format=chPos)(inp)
    conv = Dropout(0.2)(conv)
    conv = Conv2D(n_ch, (3, 3), activation='relu', padding='same',data_format=chPos)(conv)
    return conv

def get_unet_pool_part(n_ch, inp, chPos):
    conv = get_unet_conv_part(n_ch, inp, chPos)
    pool = MaxPooling2D((2, 2),data_format=chPos)(conv)
    return conv, pool

def get_unet_upsize_part(n_ch, inpConv, uConv, chPos):
    up = UpSampling2D(size=(2, 2),data_format=chPos)(inpConv)
    if (chPos == 'channels_first'):
        up = concatenate([uConv,up],axis=1)
    else:
        up = concatenate([uConv,up],axis=3)
    conv = get_unet_conv_part(n_ch, up, chPos)
    return conv
    
def get_unet(n_inpCh, n_ch, chFactor, n_lvls, n_outCh, patch_height,patch_width, chPos):
    """ chPos can be 'channels_first' or 'channels_last' """
    convStack = []
    chStack = []
    if (chPos == 'channels_first'):
        inputs = Input(shape=(n_inpCh,patch_height,patch_width), name='InputLayer')
    else:
        inputs = Input(shape=(patch_height,patch_width,n_inpCh), name='InputLayer')
    pool = inputs
    ch = n_ch
    for i in range(n_lvls):
        iCh = int(ch)
        conv, pool = get_unet_pool_part(iCh, pool, chPos)
        convStack.append(conv)
        chStack.append(iCh)
        ch *= chFactor

    iCh = int(ch)
    conv = Conv2D(iCh, (3, 3), activation='relu', padding='same',data_format=chPos)(pool)
    conv = Dropout(0.2)(conv)
    conv = Conv2D(iCh, (3, 3), activation='relu', padding='same',data_format=chPos)(conv)

    for i in range(n_lvls):
        ii = n_lvls-1-i
        convL = convStack[ii]
        iCh = chStack[ii]
        conv = get_unet_upsize_part(iCh, conv, convL, chPos)

    conv = Conv2D(n_outCh*2, (1, 1), activation='relu',padding='same',data_format=chPos)(conv)
    if (chPos == 'channels_first'):
        conv = Reshape((n_outCh*2,patch_height*patch_width))(conv)
        conv = Permute((2,1))(conv)
    conv = Reshape((patch_height*patch_width*n_outCh, 2))(conv)
    conv = Activation('softmax', name='OutputLayer')(conv)

    model = Model(inputs=inputs, outputs=conv)

    loss = tf.keras.losses.CategoricalCrossentropy()
    opt = 'Adam'
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(2)])

    return model, inputs, conv

#This class is responsible for detecting the pinball ball from the images being provided
class BallLocator():
    def __init__(self,target=None, name=None):
        super(BallLocator,self).__init__()
        
        self.target = target
        self.name = name
        self.processing = True

        nInpCh = 2  # The N last greyscale images.
        nOutCh = nInpCh - 1  # The N last masks.

        channels = 20
        chFact = 1.0
        lvls = 5

        self.imgW = 512
        self.imgH = 256
        chPos = 'channels_last'

        self.tfSession = None

        self.pinballUNet, self.inputNode, self.outputNode = get_unet(nInpCh, channels, chFact, lvls, nOutCh, self.imgH, self.imgW, chPos)
        self.pinballUNet.summary()

        input_shape = (self.imgH,self.imgW,2)
        self.frameHistory = np.zeros(input_shape)

    def loadWeights(self):
        # Read weights from json
        import json
        jsonFn = "ML/weights_DL_2_5_20_1.0_1.json"

        with open(jsonFn,'r',encoding = 'utf-8') as fp:
            jw = json.load(fp)
        weights = []
        for i in range(len(jw)):
            weights.append(np.array(jw[i], dtype=np.float32))

        self.pinballUNet.set_weights(weights)   

    def setSession(self, tfSession):
        self.tfSession = tfSession

    def mask2pos(self,res):
        w = res.shape[1]
        h = res.shape[0]
        for i in range(5): #5tries
            nz = np.nonzero(res)
            if len(nz[1]) < 1:
                return -1,-1
            posX = nz[1].max()
            posY = np.nonzero(np.expand_dims(res[:,posX],0))[1].min()
            seed = (posX,posY)
            mask = np.zeros((h+2,w+2),np.uint8)

            floodflags = 4
            floodflags |= cv.FLOODFILL_MASK_ONLY
            floodflags |= (255 << 8)

            num, im3, mask, rect = cv.floodFill(res, mask, seed, 255, (10,)*3, (10,)*3, floodflags)

            if((num > 80) and (num < 220)):
                posX = rect[0]+rect[2]/2
                posY = rect[1]+rect[3]/2
                return posX/w, posY/h 
            
            res = res & (255 - mask[1:-1, 1:-1])
        return -1,-1 # No balls found in X tries

    def im2pos(self, im):
        w = im.shape[1]
        h = im.shape[0]

        im = im / 255.0
        im2 = np.expand_dims(im,0)

        pred = self.tfSession.run(self.outputNode, feed_dict={self.inputNode: im2})

        mskVec = np.argmax(pred[0],axis=1)
        mskImg = np.reshape(mskVec, (h,w))*255
        res = np.array(mskImg,dtype=np.uint8)
        #cv.imshow('Segmentation', res)
        return self.mask2pos(res)

    def update_ball_location_states(self, new_frame):
        # move the oldest state to the end of array and replace with new state
        self.frameHistory = np.roll(self.frameHistory, -1, axis=2)
        self.frameHistory[:,:,1] = new_frame

    # Resize the input frame to the size expected by the CNN and make convert it to greyscale 
    def processFrame(self, frame):
        frame = cv.resize(frame,(self.imgW,self.imgH))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.update_ball_location_states(frame)

    # Attempts to locate the ball. 
    # Returns the X and Y coordinate of the ball.
    # Returns -1, -1 if no shapes resembling the ball was found.
    # Returns -1, -2 if no ball was located in the shapes found. 
    def locate_ball(self, latestFrame):
        ### Preprocess the new frame and update the frame history ###
        self.processFrame(latestFrame)

        ### Get the input frames from the frame history and attempt to locate the ball ###
        inputframes = self.frameHistory

        #with sharedData.cnnTfGraph.as_default(), sharedData.cnnTfSession.as_default():
        bX, bY = self.im2pos(inputframes)
        return [bX, bY]