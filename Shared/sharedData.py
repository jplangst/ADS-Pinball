import queue

### Pinball machine mode
from enum import Enum
class PinballMode(Enum):
    PLAYING = 1
    TRAINING = 2
    RECORDING = 3

### Main state ###
# Set to false to end the main loop
readingVideoFrames = False

# Game over detection parameters 
gameOverBallLocationThreshold = 0.8
gameOverBrightPixelThreshold = 100

# Suitable lighting conditions detection parameters
brightnessThreshold = 10000

# Camera stream parameters
errorReadingFramesCounter = 0
errorReadingFramesThreshold = 100 

# Reward function parameters
CenterOfTriggersPoint = (0.94,0.5)
ballToCenterDistanceThreshold = 0.15
goodTriggerThreshold = 0.80 # Considering the X component
goodTriggerBallTravelDistanceThreshold = 0.15 # Considering the X component

### ML ###
# This queue holds the frame that will be processed by the ML scripts
MLFramesQueue = queue.Queue(1) 
rlProcessingFrame = False
performingML = False
RLTraining = False

### Episode Recording ###
#This queue holds the frames that we want to record to video
recordEpisodeFramesQueue = queue.Queue(1)
recordingEpisodeFrame = False 

### IL Recording Thread
ILRecordingQueue = queue.Queue(1)
ilEpisodeFrame = False #TODO Not sure if need yet

### OCR ###
#This queue holds the score as found via OCR. The OCR thread will insert into this queue.
ocrScoreQueue = queue.Queue(1) 
#This queue holds the frames that should be OCR processed. The main thread will insert frames at some interval and the OCR thread will dequeue to process.
ocrFramesQueue = queue.Queue(1) 
#Will hold the current score as determined by the OCR
sharedOCRScore = 0
#Will hold the difference between the last ocr score updates
ocrScoreDifference = 0
#A flag that indicates if game over was detected by OCR or by counting bright pixels. Used to terminate an episode
gameOver = False
performingOCR = False

### Visualisation ###
#This queue holds the data needed for the pinball visualisation module
pinballVisQueue = queue.Queue(1)
currentEpisodeReward = 0
episodeStep = 0
currentAction = 0
currentReward = 0
episodeRecordingInterval = 100 ##TODO set this back to reasonable interval when actually trainnig. e.g. 25

### Ball locator ###
ballFramesQueue = queue.Queue(1)
lastValidBallLocation = [-1,-1]

### Record JSON file related
stateFileFilepath = "./trainingState.json"

### Debug flags ###
debugBrightnessCheck = False