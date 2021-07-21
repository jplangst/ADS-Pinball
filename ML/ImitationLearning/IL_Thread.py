import threading
import time
import numpy as np
import math

import Shared.sharedData as sharedData
from ML.PinballController import PinballController

## TODO 
# 1. Reuse the game over detection to stop recording data

# The ILThread is responsible for recording frames and user inputs 
class ILRecordingThread(threading.Thread):
    def __init__(self, target=None, name=None):
        super(ILRecordingThread,self).__init__()
        self.target = target
        self.name = name
        self.recording = True
        self.pinballController = PinballController()

    def run(self):
        sharedData.recordingIL = True

        while self.recording:
            ## This will stop the recording of data when a full game has been played, e.g. all balls have been lost
            if sharedData.gameOver: ### TODO start recording again
                self.recording = False

            ## Check for button presses
            buttonsPressedState = self.pinballController.readInputPins()
            ballLocation = [-1,-1]
            episode_reward = 0
            episode_string = "RECORDING"
            episode = 0

            ## Grab the current frame

            ## TODO when we actually record we probably want the raw frame image, the ball location, the current score            

            #Check for frames to process
            while(not sharedData.ILRecordingQueue.empty()):
                frame, _ = sharedData.ILRecordingQueue.get_nowait()

                ### Send information to visualization thread ###
                if(not sharedData.pinballVisQueue.full()):
                    sharedData.pinballVisQueue.put([frame, ballLocation, episode_reward, buttonsPressedState, episode_string, episode])
                else:
                    print("Queue full")

            time.sleep(0.000001)
        sharedData.recordingIL = False
        print("IL recording thread stopped")
        return
    
    def stopRecording(self):
        print("Stopping ML thread")       
        self.recording = False


