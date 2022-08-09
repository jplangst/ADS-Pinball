# To restart the camera daemon in case it has stopped working. Use in terminal
# sudo systemctl restart nvargus-daemon  

import cv2 
import time

import Shared.sharedData as sharedData
from PinballUtils import *
from ML.MlThread import MLThread
from Visualization.Visualization import VisualizationThread
from Visualization.EpisodeRecorderThread import EpisodeRecorderThread

# TODO check reward for pushing start. Might need to tune. 
# TODO convert limiter to class so we can have one instance per button 
# TODO try without lstm and increase dense part instead 

class PinballMain():
    def __init__(self, episode=0) -> None:
        self.startMLThread = True

        ### Jetson AGX Configuration ###
        set_power_mode(0) #0 is maximum performance
        set_jetson_clocks() #This line overclocks the clock speeds
        set_jetson_fan(255) #The fan speed, ranging from 0-255 where 255 is max

        ### Threads ### 
        if self.startMLThread:
            self.mlThread = MLThread(name='ML Thread', modelRestorePath="Pinball_PPO_LSTM/PinballMachine/20220619-162415/model.ckpt-2000")
        self.visThread = VisualizationThread(name='Visualisation Thread')
        self.recordEpisodeThread = EpisodeRecorderThread(name='Episode Recorder Thread',recordingFolder='episodeRecordings/')
        #self.ocrThread = OCRThread(name='ocrThread') 

        ### State management ###
        self.episode = episode
        self.episodeState = 2 #0=new episode, 1=same episode, 2=end episode
        #self.newEpisodeRecording = False 

        ### Input Video Processing ### 
        self.videoCap = None

        ### For the camera unit attached to the AGX unit ###
        #Slice the observation frame so it only contains the play area
        self.inputFrameSize = (1080,1920)
        self.playAreaXSlice = (70, self.inputFrameSize[0]-500)
        self.playAreaYSlice = (450, self.inputFrameSize[1]-450)

        # Attempt to load the camera calibration data 
        calibrationFile = 'CameraCalibration/calibrationData.json'
        self.calibrationLoaded, self.calData, self.cameraMatrix, self.newCameraMatrix, \
            self.roi, self.dist = loadCameraCalibration(calibrationFile)        

        # Setup the GStreamer video capture
        self.pipeline = "nvarguscamerasrc sensor-id=0 sensor-mode=0 ! video/x-raw(memory:NVMM), \
            width=(int)1920, height=(int)1080, format=(string)NV12 \
            ! nvvidconv ! video/x-raw, \
            format=(string)BGRx ! videoconvert ! video/x-raw, \
            format=(string)BGR ! appsink"

    def start_threads(self): 
        if self.startMLThread:      
            self.mlThread.start()       
        self.visThread.start()
        self.recordEpisodeThread.start()
        #self.ocrThread.start()

    def stop_threads(self):
        if self.startMLThread:
            self.mlThread.stopProcessing()
            self.mlThread.join()
        self.visThread.stopProcessing()
        self.visThread.join()
        self.recordEpisodeThread.stopProcessing()
        self.recordEpisodeThread.join()
        #self.ocrThread.stopProcessing()
        #self.ocrThread.join()

    def cleanup(self):
        self.stop_threads()
        if not self.videoCap == None:
            self.videoCap.release()  
    
    # Check for game over by counting the amount of bright pixels in the image
    # If the game is over the shared data flag: gameOver is set to True 
    def checkForGameOver(self, frame):
        thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        ret, thresh = cv2.threshold(thresh, 254, 255, cv2.THRESH_BINARY)
        nmbBright = cv2.countNonZero(thresh)
        #cv2.imshow('Game Over bright pixels', thresh)
        #print(nmbBright)

        #Game over if the ball was last seen close to the bottom of the screen and the number of bright pixels are below the threshold
        if((sharedData.lastValidBallLocation[0] > 0.8) 
            and nmbBright < sharedData.gameOverBrightPixelThreshold):
            print("Game over detected")
            sharedData.lastValidBallLocation = [-1,-1]
            return True 
        return False

    def rotate_frame(self, frame, degrees):
        frame_center = tuple(np.array(frame.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(frame_center, degrees, 1.0)
        result = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def processFrame(self, frame):
        if self.calibrationLoaded:
            frame = perspectiveCorrect(frame, self.cameraMatrix, self.newCameraMatrix, self.dist, self.roi)
        frame = self.rotate_frame(frame, -188)

        #Slice the video frame so it only contains the play area
        frame = frame[self.playAreaXSlice[0]:self.playAreaXSlice[1], self.playAreaYSlice[0]:self.playAreaYSlice[1]]
        return frame

    def initEpisode(self):
        self.episode += 1
        self.episodeState = 0 #New episode
        sharedData.gameOver = False

    def endEpisode(self):
        self.episodeState = 2 #End of episode
        sharedData.gameOver = True

    def startAIPinball(self):
        ### Using the camera attached to the AGX unit ###
        self.videoCap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        # To visualise when the ML thread is not running
        if not self.startMLThread:
            sharedData.RLTraining = True

        if not self.videoCap.isOpened():
            return "EXITING - Failed to open the video capture stream"

        self.start_threads()

        #Start reading frames from the video source
        sharedData.readingVideoFrames = True
        errorReadingFramesCounter = 0
        errorReadingFramesThreshold = 100
        while(sharedData.readingVideoFrames): 
            ret, frame = self.videoCap.read()

            # Check if we read the frame correctly
            if not ret:
                errorReadingFramesCounter += 1
                print("Error reading frame from camera")

                if errorReadingFramesCounter < errorReadingFramesThreshold:
                    continue
                else:
                    print("Exiting due to camera issues")
                    break

            frame = self.processFrame(frame)

            #Just visualise the frames while the RL trains
            if(sharedData.RLTraining):            
                if(not sharedData.pinballVisQueue.full()):
                    sharedData.pinballVisQueue.put([frame, [-1,-1], sharedData.currentEpisodeReward, 0, str(self.episode)+ " Steps: " + str(sharedData.episodeStep) + " Training", self.episode])
                continue

            #Init the episode
            if self.episodeState == 2:
                self.initEpisode()
                terminal = False
                sharedData.MLFramesQueue.put([frame, self.episode, self.episodeState, terminal])
                self.episodeState = 1
                continue

            #Check for game over
            if not sharedData.gameOver:
                gameOver = self.checkForGameOver(frame)
                if gameOver:
                    self.endEpisode()
                    time.sleep(0.2)

            #Add the play area frame to the queue when the previous has been processed
            if(not sharedData.MLFramesQueue.full()):
                sharedData.MLFramesQueue.put([frame, self.episode, self.episodeState, gameOver])

            time.sleep(0.000001)
            
        self.cleanup()

        return "Finished cleanly"

pinballObject = PinballMain(2002)
result = pinballObject.startAIPinball()
print(result)