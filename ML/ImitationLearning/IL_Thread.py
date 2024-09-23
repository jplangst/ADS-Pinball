import threading
import time
import numpy as np
import math
import tensorflow as tf
import h5py
import uuid

import Shared.sharedData as sharedData
from ML.BallLocator import BallLocator
from ML.PinballController import PinballController

# The ILThread is responsible for recording frames and user inputs 
class ILRecordingThread(threading.Thread):
    def __init__(self, target=None, name=None):
        super(ILRecordingThread,self).__init__()
        self.target = target
        self.name = name
        self.recording = True
        self.pinballController = PinballController()

        # Data recording related
        self.episode = 0
        # Setup the HDF5 file for recording the data
        self.playAreaFrame_shape = (515, 970)  # 2D frame shape 
        self.playAreaFrame_chunkSize = (self.chunkSize, self.playAreaFrame_shape[0], self.playAreaFrame_shape[1])  # Write data in chunks of 10 frames
        self.screenFrame_shape = (510, 206)  # 2D frame shape 
        self.screenFrame_chunkSize = (self.chunkSize, self.screenFrame_shape[0], self.screenFrame_shape[1])  # Write data in chunks of 10 frames
        self.compression_level = 5     # Moderate GZIP compression

        # Pre allocate the chunks for recording the data
        self.playAreaDataChunk = np.zeros((self.chunkSize, *self.playAreaFrame_shape))
        self.screenDataChunk = np.zeros((self.chunkSize, *self.screenFrame_shape))
        self.userInputsDataChunk = np.zeros((self.chunkSize,))
        self.ballLocationDataChunk = np.zeros((self.chunkSize,2))

        # data chunks thich will be used for recording the data
        self.playAreaDataChunk = None
        self.screenDataChunk = None
        self.userInputsDataChunk = None
        self.ballLocationDataChunk = None
        self.currentChunkIndex = 0
        self.chunkSize = 10

        # CNN related 
        self.locatedConsecutiveFrames = 0
        config = tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'GPU': True})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.cnnGraph = tf.compat.v1.Graph()
        self.cnnSession = tf.compat.v1.Session(graph=self.cnnGraph, config=config)

        with self.cnnGraph.as_default(), self.cnnSession.as_default(), tf.compat.v1.variable_scope('BallLocator'):
            self.ballLocator = BallLocator()
            init = tf.compat.v1.global_variables_initializer()
            self.cnnSession.run(init)
            self.ballLocator.loadWeights()
            self.ballLocator.setSession(self.cnnSession)

            #This is to init the model so we don't have to wait for that later
            self.ballLocator.im2pos(np.zeros((self.ballLocator.imgH,self.ballLocator.imgW,2)))

    def run(self):
        sharedData.recordingIL = True  

        # Continously record data until stopped
        while sharedData.recordingIL:          
            buttonPressState = self.pinballController.readInputPins()

            # Check if we should start recording 
            if not self.recording and sharedData.gameOver and buttonPressState==4:
                self.recording = True
                sharedData.gameOver = False
            else: #Otherwise wait until start has been pressed
                time.sleep(0.000001)
                continue
            
            # Save one file per game
            uuid = uuid.uuid4() #Create a unique identifier for the file
            filename = "episode_"+self.episode+"_data_"+uuid+".h5"
            try:
                self.hf = h5py.File(filename, "a")  # 'a' mode means append if the file exists, or create a new one
                # Access existing datasets or create new ones if needed
                if "playArea_frame" not in self.hf:
                    self.hf.create_dataset("playArea_frame", shape=(0, *self.playAreaFrame_shape), maxshape=(None, *self.playAreaFrame_shape), dtype='float64',
                                        compression="gzip", compression_opts=self.compression_level, chunks=self.playAreaFrame_chunkSize)
                if "screen_frame" not in self.hf:
                    self.hf.create_dataset("playArea_frame", shape=(0, *self.screenFrame_shape), maxshape=(None, *self.screenFrame_shape), dtype='float64',
                                        compression="gzip", compression_opts=self.compression_level, chunks=self.screenFrame_chunkSize)
                if "ball_location" not in self.hf:
                    self.hf.create_dataset("ball_location", shape=(0,2), maxshape=(None,2), dtype='int32',
                                        compression="gzip", compression_opts=self.compression_level, chunks=(10,))
                if "user_inputs" not in self.hf:
                    self.hf.create_dataset("user_inputs", shape=(0,), maxshape=(None,), dtype='int32',
                                        compression="gzip", compression_opts=self.compression_level, chunks=(10,2))

                while self.recording:
                    ## This will stop the recording of data when a full game has been played, e.g. all balls have been lost
                    if sharedData.gameOver: 
                        self.recording = False
                        self.episode += 1

                    ## Check for button presses
                    buttonsPressedState = self.pinballController.readInputPins()
                    episode_reward = 0
                    episode_string = "RECORDING DATA"

                    #Check for frames to process
                    while(not sharedData.ILRecordingQueue.empty()):
                        ### Grab the current frame ###
                        playAreaFrame, screenFrame = sharedData.ILRecordingQueue.get_nowait()
                        ### Attempt to detect the ball in the frame ###
                        locateBall, ballLocation = self.locateBall(playAreaFrame)

                        ### Send information to visualization thread ###
                        if(not sharedData.pinballVisQueue.full()):
                            sharedData.pinballVisQueue.put([playAreaFrame, ballLocation, episode_reward, buttonsPressedState, episode_string, self.episode])
                        else:
                            print("Queue full")

                        ### Record the data to file
                        self.recordData(playAreaFrame, screenFrame, ballLocation, buttonsPressedState)
                        
                    time.sleep(0.000001)
            finally:
                # Ensure the HDF5 file is closed properly, even if an error occurs
                self.hf.close()
        print("IL recording thread stopped")
        return
        
    
    def stopRecording(self):
        print("Stopping IL recording thread")       
        sharedData.recordingIL = False
        self.recording = False

    def locateBall(self, frame):
            ### CNN ### - NOTE it is expected that the frame has been calibrated and cropped to the play area             
            with self.cnnGraph.as_default(), self.cnnSession.as_default(), tf.compat.v1.variable_scope('BallLocator'):
                ballLocation = self.ballLocator.locate_ball(frame)   

            ballDetected = not (math.isclose(ballLocation[0],-1) and math.isclose(ballLocation[1],-1))
            # If the ball was successfully located, store the last position
            # The ball must have been tracked for the last 3 frames. This is to reduce impact of noisy detections. 
            # TODO could also try smoothing instead by keeping a position and then adding a part of the new position to it. 
            if ballDetected: 
                self.locatedConsecutiveFrames += 1
                if self.locatedConsecutiveFrames >= 3:
                    sharedData.lastValidBallLocation = ballLocation
            else:
                self.locatedConsecutiveFrames = 0

            return ballDetected, ballLocation
    
    def recordData(self, playAreaFrame, screenFrame, ballLocation, buttonsPressedState):
        ### Add data to the chunk
        self.playAreaDataChunk[self.currentChunkIndex] = playAreaFrame
        self.screenDataChunk[self.currentChunkIndex] = screenFrame
        self.userInputsDataChunk[self.currentChunkIndex] = buttonsPressedState
        self.ballLocationDataChunk[self.currentChunkIndex] = ballLocation
        self.currentChunkIndex += 1

        ### When the chunk is full, write it to the file and reset counter
        if self.currentChunkIndex >= self.chunkSize:
            playArea_dataset = self.hf["playArea_frame"]
            screen_dataset = self.hf["screen_frame"]
            userInputs_dataset = self.hf["user_inputs"]
            ballLocation_dataset = self.hf["ball_location"]
            
            # Resize datasets to accommodate new data
            current_playArea_size = playArea_dataset.shape[0]
            current_screen_size = screen_dataset.shape[0]
            current_userInputs_size = userInputs_dataset.shape[0]
            current_ballLocation_size = ballLocation_dataset.shape[0]

            # Record the data
            playArea_dataset.resize(current_playArea_size + self.playAreaDataChunk.shape[0], axis=0)
            screen_dataset.resize(current_screen_size + self.screenDataChunk.shape[0], axis=0)
            userInputs_dataset.resize(current_userInputs_size + self.userInputsDataChunk.shape[0], axis=0)
            ballLocation_dataset.resize(current_ballLocation_size + self.ballLocationDataChunk.shape[0], axis=0)

            # Append new data
            playArea_dataset[current_playArea_size:] = self.playAreaDataChunk
            screen_dataset[current_screen_size:] = self.screenDataChunk
            userInputs_dataset[current_userInputs_size:] = self.userInputsDataChunk
            ballLocation_dataset[current_ballLocation_size:] = self.ballLocationDataChunk

            # Flush to save to disk
            self.hf.flush()
            self.currentChunkIndex = 0