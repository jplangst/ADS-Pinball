import threading
import cv2 
import time
import numpy as np

import Shared.sharedData as sharedData

class EpisodeRecorderThread(threading.Thread):
    def __init__(self,target=None, name=None, recordingFolder='.'):
        super(EpisodeRecorderThread,self).__init__()
        self.target = target
        self.name = name
        self.processing = True
        self.recordingFolder = recordingFolder

        # The out stream used when recording videos
        self.out = None 

        return

    def run(self):
        while self.processing:
            #Check for frames to process
            if not sharedData.recordEpisodeFramesQueue.empty():
                episodeFrame, state, episode = sharedData.recordEpisodeFramesQueue.get_nowait()
                self.recordFrame(episodeFrame, state, episode)
            time.sleep(0.001)
        self.cleanup()
        print("OCR thread stopped")
        return
    
    def stopProcessing(self):
        print("Stopping OCR thread")
        self.processing = False

    def cleanup(self):
        if not self.out == None:
            self.out.release()
            self.out = None 

    # state - 0 to record a new video, 1 to add frames to the video, 
    #   2 to stop recording and close the reource
    def recordFrame(self, frame, state, episode, FPS=30):
        
        #if frame != None:
            #frame = np.rot90(frame, k=-1)

        #Create a new video recording resource and save the frame
        if state == 0:                   
            height, width, _ = frame.shape
            filename = 'episode'+str(episode)+'.avi'      
            fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
            self.out = cv2.VideoWriter(self.recordingFolder+filename, fourcc, FPS, (width,  height))
            self.out.write(frame)
        #End video recording and close resource
        elif state == 2 and self.out != None:        
            self.out.release()
            self.out = None 
        #Record the frame
        else: 
            if self.out != None:
                self.out.write(frame)