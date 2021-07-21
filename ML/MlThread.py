import threading
import time
import numpy as np
import math
import tensorflow as tf
# We are using the compatability mode for TF1 in order to use the RL code. 
# As long as we use this deprecation warnings will occur and fill the output.
# If we upgrade the RL part to use TF2 as well we can remove this.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import Shared.sharedData as sharedData
from ML.BallLocator import BallLocator
from ML.RL import RL_Controller

# The MLThread is responsible for taking frames from the shared data ML queue and 
# processing it with the CNN to locate the ball. The location of the ball is fed 
# to the RL model for decision making. The decision made is performed through the JetsonAGX controller. 

class MLThread(threading.Thread):
    def __init__(self, target=None, name=None, modelRestorePath=None, checkpointDir=None):
        super(MLThread,self).__init__()
        self.target = target
        self.name = name
        self.processing = True
        self.locatedConsecutiveFrames = 0

        # RL related. NOTE Could move this into the RL class instead. Similar to the CNN if desired
        input_frames_RL = 4 #The size of the history of past ball locations we will utilise
        nmb_input_features_RL = 2 # (X,Y) coordinates
        self.input_shape_RL = (input_frames_RL,nmb_input_features_RL)
        self.ballLocationHistory = np.zeros(self.input_shape_RL)
        self.RL_Controller = RL_Controller(BATCH=256, MODEL_RESTORE_PATH=modelRestorePath, CHECKPOINT_DIR=checkpointDir)

        # CNN related 
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

    def update_ball_location_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.ballLocationHistory = np.roll(self.ballLocationHistory, -1, axis=0)
        self.ballLocationHistory[-1] = new_state

    def run(self):
        sharedData.performingML = True

        while self.processing:
            if sharedData.gameOver:
                self.RL_Controller.pinballController.clearActions()

            #Check for frames to process
            while(not sharedData.MLFramesQueue.empty()):
                frame, episode, episodeState, terminal = sharedData.MLFramesQueue.get_nowait()

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

                ### Update ball location history and get the updated stack of locations##      
                self.update_ball_location_states(ballLocation)
                inputBallLocations = self.ballLocationHistory
                inputBallLocationsFlattened = inputBallLocations.flatten()
                
                ### RL ###              
                episode_reward, action_performed = self.RL_Controller.step_environment(inputBallLocationsFlattened, terminal, episodeState, episode)
                sharedData.currentEpisodeReward = episode_reward
                # TODO currently the training is done as part of the step_environment function. 
                # Can consider splitting it so it is easier to control from here.
                
                # Train the RL agent for X epochs if N episodes played

                ### Send information to visualization thread ###
                if(not sharedData.pinballVisQueue.full()):
                    sharedData.pinballVisQueue.put([frame, ballLocation, episode_reward, action_performed, episode, episode])

            time.sleep(0.000001)
        self.RL_Controller.pinballController.clearActions()
        sharedData.performingML = False
        print("ML thread stopped")
        return
    
    def stopProcessing(self):
        print("Stopping ML thread")       
        self.processing = False