# To restart the camera daemon in case it has stopped working. Use in terminal
# sudo systemctl restart nvargus-daemon  
import cv2 
import time
import datetime

import Shared.sharedData as sharedData
from PinballUtils import *
from ML.MlThread import MLThread
from Visualization.Visualization import VisualizationThread
from Visualization.EpisodeRecorderThread import EpisodeRecorderThread

#TODO Fix the action distribution histogram 
#TODO add a maximum episode / step size for the experience replay buffer. When full remove old episodes or steps to make space for new ones. 

class PinballMain():
    def __init__(self, episode=0, modelRestorePath=None, checkpointsFolder=None) -> None:
        self.startMLThread = True

        ### Jetson AGX Configuration ###
        set_power_mode(0) #0 is maximum performance
        set_jetson_clocks() #This line overclocks the clock speeds
        set_jetson_fan(255) #The fan speed, ranging from 0-255 where 255 is max

        ### Threads ### 
        if self.startMLThread:
            self.mlThread = MLThread(name='ML Thread', modelRestorePath=modelRestorePath, checkpointDir=checkpointsFolder)
        self.visThread = VisualizationThread(name='Visualisation Thread')
        self.recordEpisodeThread = EpisodeRecorderThread(name='Episode Recorder Thread',recordingFolder='episodeRecordings/')
        #self.ocrThread = OCRThread(name='ocrThread') 

        ### State management ###
        self.pauseProgram = False 
        self.pauseStartDatetime = datetime.datetime.now()
        self.episode = episode
        self.firstEpisode = True
        self.startEpisode = self.episode
        self.episodeState = 2 #0=new episode, 1=same episode, 2=end episode

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
    
    # Check if the number of bright pixels are lower than a given threshold.
    # Returns true if it is. False otherwise. 
    def checkImageBrightness(self, frame, threshold):
        thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        ret, thresh = cv2.threshold(thresh, 254, 255, cv2.THRESH_BINARY)
        nmbBright = cv2.countNonZero(thresh)
        if sharedData.debugBrightnessCheck:
            cv2.imshow('Bright pixels', thresh)
            print(nmbBright)
        return (nmbBright < threshold, nmbBright)

    def checkIfWeekdayAndLunch(self):
        e = datetime.datetime.now()
        # Check if weekend. Weekday in int. Monday=0, Sunday=6
        if(e.weekday() < 5):
            #If not check if itis lunch time
            if(e.hour >= 10 and e.hour < 14):
                #Pause as we are in lunch time
                return True
        return False
                
    # Checks for gameover. Game over is detected if the ball was last seen 
    # below a threshold and the lights in the pinball machine are out.
    def checkForGameOver(self, frame):
        result, _ = self.checkImageBrightness(frame, sharedData.gameOverBrightPixelThreshold)

        #Game over if the ball was last seen close to the bottom of the screen and the number of bright pixels are below the threshold
        if(result and sharedData.lastValidBallLocation[0] > 0.8): 
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

    def initEpisode(self, frame):
        self.episode += 1
        self.episodeState = 0 #New episode
        sharedData.gameOver = False
        terminal = False
        sharedData.MLFramesQueue.put([frame, self.episode, self.episodeState, terminal])
        self.episodeState = 1

    def endEpisode(self):
        self.episodeState = 2 #End of episode
        sharedData.gameOver = True

    # Checks the state of the camera stream
    def checkCameraFeed(self, ret):
        # Check if we read the frame correctly
        if not ret:
            sharedData.errorReadingFramesCounter += 1
            print("Error reading frame from camera")
            if sharedData.errorReadingFramesCounter < sharedData.errorReadingFramesThreshold:
                return 2
            else:
                print("Exiting due to camera issues")
                return 0
        return 1

    def startAIPinball(self):
        ### Attempt tp open the camera feed
        self.videoCap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.videoCap.isOpened():
            return "EXITING - Failed to open the video capture stream"
        
        # To visualise when the ML thread is not running
        if not self.startMLThread:
            sharedData.RLTraining = True

        self.start_threads()

        #Start reading frames from the video source
        sharedData.readingVideoFrames = True
        while(sharedData.readingVideoFrames): 
            ret, frame = self.videoCap.read()

            # Check if the camera feed is ok
            camResult = self.checkCameraFeed(ret)
            if camResult == 0:
                break 
            elif camResult == 2:
                continue 
            
            frame = self.processFrame(frame)

            #Just visualise the frames while the RL trains
            if(sharedData.RLTraining or self.pauseProgram):            
                if(not sharedData.pinballVisQueue.full()):
                    sharedData.pinballVisQueue.put([frame, [-1,-1], sharedData.currentEpisodeReward, 0, str(self.episode)+ " Steps: " + str(sharedData.episodeStep) + " Training", self.episode])
                
                if self.pauseProgram:
                    currTime = datetime.datetime.now()
                    timeDiff = currTime-self.pauseStartDatetime
                    if timeDiff.total_seconds() > 300:
                        self.pauseProgram = False
                        #self.firstEpisode = True
                continue

            #Init the episode if light conditions are good
            if self.episodeState == 2:
                bBrightEnough, nmbBright = self.checkImageBrightness(frame, sharedData.brightnessThreshold)
                bLunchTime = self.checkIfWeekdayAndLunch()
                bOkToPlay = bBrightEnough and not bLunchTime

                if self.firstEpisode or bOkToPlay:
                    self.firstEpisode = False
                    self.initEpisode(frame)    
                else:
                    #If not we clear output actions and wait for 5 minutes before checking again
                    self.mlThread.RL_Controller.pinballController.clearActions()

                    if not bBrightEnough:
                        print("Too dark to play, waiting 5 minutes. NmbBright: ", nmbBright)
                    elif bLunchTime:
                        print("Taking a lunch break :)")
                    
                    self.pauseProgram = True
                    self.pauseStartDatetime = datetime.datetime.now()
                continue

            #Check for game over
            if (not sharedData.gameOver and self.checkForGameOver(frame)) or sharedData.bGameplayStagnated:    
                self.endEpisode()
                time.sleep(0.2) 

            #Add the play area frame to the queue when the previous has been processed
            if(not sharedData.MLFramesQueue.full()):
                sharedData.MLFramesQueue.put([frame, self.episode, self.episodeState, sharedData.gameOver])

            time.sleep(0.000001)

        self.cleanup()
        return "Finished cleanly"

# Attempt to load trainig progress state
ret, episode, modelCheckpointFilepath, checkpointsFolder = loadTrainingStateFile()
pinballObject = None
if ret:
    pinballObject = PinballMain(episode, modelCheckpointFilepath, checkpointsFolder)
else:
    pinballObject = PinballMain()

# Start the training process
result = pinballObject.startAIPinball()
print(result)