# To restart the camera daemon in case it has stopped working. Use in terminal
# sudo systemctl restart nvargus-daemon  

import cv2 
import time

import Shared.sharedData as sharedData
from Shared.sharedData import PinballMode
from PinballUtils import *
from ML.MlThread import MLThread
from ML.ImitationLearning.IL_Thread import ILRecordingThread
from Visualization.Visualization import VisualizationThread
from Visualization.EpisodeRecorderThread import EpisodeRecorderThread

#TODO Think about adding a time delay after the ball has disappeared before pressing start gives a point. 
#TODO Fix the action distribution histogram 

class PinballMain():
    def __init__(self, episode=0, pinballMode=PinballMode.RECORDING,modelRestorePath=None, checkpointsFolder=None) -> None:
        self.pinballMode = pinballMode

        ### Jetson AGX Configuration ###
        set_power_mode(0) #0 is maximum performance
        set_jetson_clocks() #This line overclocks the clock speeds
        set_jetson_fan(255) #The fan speed, ranging from 0-255 where 255 is max

        ### Threads ### 
        if self.pinballMode == PinballMode.TRAINING or self.pinballMode == PinballMode.PLAYING: 
            self.mlThread = MLThread(name='ML Thread', modelRestorePath=modelRestorePath, checkpointDir=checkpointsFolder)
        if self.pinballMode == PinballMode.RECORDING:
            self.ilRecordingThread = ILRecordingThread(name='IL Recording Thread')
        self.visThread = VisualizationThread(name='Visualisation Thread')
        self.recordEpisodeThread = EpisodeRecorderThread(name='Episode Recorder Thread',recordingFolder='episodeRecordings/')
        #self.ocrThread = OCRThread(name='ocrThread') 

        ### State management ###
        self.episode = episode
        self.startEpisode = self.episode
        self.episodeState = 2 #0=new episode, 1=same episode, 2=end episode

        ### Input Video Processing ### 
        self.videoCap = None

        ### For the camera unit attached to the AGX unit ###
        self.inputFrameSize = (1080,1920)
        #Slice the observation frame so it only contains the play area
        self.playAreaXSlice = (70, self.inputFrameSize[0]-500)
        self.playAreaYSlice = (450, self.inputFrameSize[1]-450)
        #Slice the observation frame so it only contains the digital screen 
        # TODO find the correct values for this slice
        self.screenAreaXSlice = (70, self.inputFrameSize[0]-500)
        self.screenAreaYSlice = (450, self.inputFrameSize[1]-450)

        # Attempt to load the camera calibration data for the play area
        calibrationFile = 'CameraCalibration/playArea_calibrationData.json'
        self.calibrationLoaded, self.calData, self.cameraMatrix, self.newCameraMatrix, \
            self.roi, self.dist = loadCameraCalibration(calibrationFile)        

        # Attempt to load the camera calibration data for the digital display
        calibrationFile = 'CameraCalibration/ocr_calibrationData.json'
        self.ocr_calibrationLoaded, self.ocr_calData, self.ocr_cameraMatrix, self.ocr_newCameraMatrix, \
            self.ocr_roi, self.ocr_dist = loadCameraCalibration(calibrationFile)  

        # Setup the GStreamer video capture
        self.pipeline = "nvarguscamerasrc sensor-id=0 sensor-mode=0 ! video/x-raw(memory:NVMM), \
            width=(int)1920, height=(int)1080, format=(string)NV12 \
            ! nvvidconv ! video/x-raw, \
            format=(string)BGRx ! videoconvert ! video/x-raw, \
            format=(string)BGR ! appsink"

    def start_threads(self): 
        if self.pinballMode == PinballMode.TRAINING or self.pinballMode == PinballMode.PLAYING:     
            self.mlThread.start()   
        if self.pinballMode == PinballMode.RECORDING:
            self.ilRecordingThread.start()
        self.visThread.start()
        self.recordEpisodeThread.start()
        #self.ocrThread.start()

    def stop_threads(self):
        if self.pinballMode == PinballMode.TRAINING or self.pinballMode == PinballMode.PLAYING: 
            self.mlThread.stopProcessing()
            self.mlThread.join()
        if self.pinballMode == PinballMode.RECORDING:
            self.ilRecordingThread.stopRecording()
            self.ilRecordingThread.join()
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
            playAreaFrame = perspectiveCorrect(frame, self.cameraMatrix, self.newCameraMatrix, self.dist, self.roi)
        playAreaFrame = self.rotate_frame(playAreaFrame, -188)
        #Slice the video frame so it only contains the play area
        playAreaFrame = playAreaFrame[self.playAreaXSlice[0]:self.playAreaXSlice[1], self.playAreaYSlice[0]:self.playAreaYSlice[1]]

        # TODO verify slice area and rotation for the display screen
        if self.ocr_calibrationLoaded:
            screenFrame = perspectiveCorrect(frame, self.ocr_cameraMatrix, self.ocr_newCameraMatrix, self.ocr_dist, self.ocr_roi)
        screenFrame = self.rotate_frame(screenFrame, -188)
        #Slice the video frame so it only contains the play area
        playAreaFrame = playAreaFrame[self.screenAreaXSlice[0]:self.screenAreaXSlice[1], self.screenAreaYSlice[0]:self.screenAreaYSlice[1]]

        return playAreaFrame, screenFrame

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
            
            playAreaFrame, screenFrame = self.processFrame(frame)

            ## TODO finish this mode
            if self.pinballMode == PinballMode.PLAYING: #Same as training but we don't perform any more training.
                ## Record frames and send to the agent
                ## Perform selected decisions

                ##TODO this is temp until this mode is implemented 
                ## TODO use this mode to test camera lag. Perhaps it is due to adding a frame while the current frame is being processed. Perhaps it is better to add the frame only after the current one is fininshed processing!
                if(not sharedData.pinballVisQueue.full()):
                        sharedData.pinballVisQueue.put([playAreaFrame, [-1,-1], sharedData.currentEpisodeReward, 0, 
                            str(self.episode)+ " Steps: " + str(sharedData.episodeStep) + " Training", self.episode])
                continue

            ## TODO finish this mode
            elif self.pinballMode == PinballMode.RECORDING:
                ## TODO start recording when start button first pressed, unless already recording. Can check this in the IL thread. Might as well.

                #Check for game over
                if not sharedData.gameOver and self.checkForGameOver(playAreaFrame):    
                    sharedData.gameOver = True
                    time.sleep(0.2)

                ## Grab the current frame and add it to the IL Recording Queue
                if(not sharedData.ILRecordingQueue.full()):
                    sharedData.ILRecordingQueue.put([playAreaFrame, screenFrame])

            ## If we are doing reinforcement learning
            elif self.pinballMode == PinballMode.TRAINING:
                #Just visualise the frames while the RL trains
                if(sharedData.RLTraining):            
                    if(not sharedData.pinballVisQueue.full()):
                        sharedData.pinballVisQueue.put([playAreaFrame, [-1,-1], sharedData.currentEpisodeReward, 0, 
                            str(self.episode)+ " Steps: " + str(sharedData.episodeStep) + " Training", self.episode])
                    continue

                #Init the episode if light conditions are good
                if self.episodeState == 2:
                    result, nmbBright = self.checkImageBrightness(playAreaFrame, sharedData.brightnessThreshold)
                    if self.startEpisode==self.episode or result:
                        self.initEpisode(playAreaFrame)    
                    else:
                        #If not we clear output actions and wait for 5 minutes before checking again
                        print("Too dark to play, waiting 5 minutes. NmbBright: ", nmbBright)
                        self.mlThread.RL_Controller.pinballController.clearActions()
                        time.sleep(300)
                    continue

                #Check for game over
                if not sharedData.gameOver and self.checkForGameOver(playAreaFrame):    
                    self.endEpisode()
                    time.sleep(0.2)

                #Add the play area frame to the queue when the previous has been processed
                if(not sharedData.MLFramesQueue.full()):
                    sharedData.MLFramesQueue.put([playAreaFrame, self.episode, self.episodeState, sharedData.gameOver])

            time.sleep(0.000001)

        self.cleanup()
        return "Finished cleanly"

# Attempt to load trainig progress state
loadTrainingState = True
pinballMode = PinballMode.PLAYING #RECORDING, PLAYING, TRAINING

pinballObject = None
if loadTrainingState:
    ret, episode, modelCheckpointFilepath, checkpointsFolder = loadTrainingStateFile()
    if ret:
        pinballObject = PinballMain(episode, pinballMode, modelCheckpointFilepath, checkpointsFolder)
else:
    episode = 0
    pinballObject = PinballMain(episode, pinballMode)

# Start the training process
result = pinballObject.startAIPinball()
print(result)