import tensorflow.compat.v1 as tf

# We are using the compatability mode for TF1 in order to use the RL code. 
# As long as we use this deprecation warnings will occur and fill the output.
# If we upgrade the RL part to use TF2 as well we can remove this.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datetime import datetime
import numpy as np
import math
import os

import Shared.sharedData as sharedData
from ML.MlUtils import RunningStats, discount, add_histogram
from ML.PPO_AI_Pinball import PPO
from ML.PinballController import PinballController

class RL_Controller(object):
    def __init__(self, OUTPUT_RESULTS_DIR='.', MODEL_RESTORE_PATH=None, GAMMA=0.99, LAMBDA=0.95, BATCH=8192):
        ENVIRONMENT = 'PinballMachine'

        TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "Pinball_PPO_LSTM", ENVIRONMENT, TIMESTAMP)

        self.pinballController = PinballController()

        self.ppo = PPO(self.SUMMARY_DIR, gpu=True, state_dimension=np.array([8]), action_dimension=np.array([5]))

        if MODEL_RESTORE_PATH is not None:
            self.ppo.restore_model(MODEL_RESTORE_PATH)

        self.t, self.terminal = 0, False
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v, self.buffer_terminal = [], [], [], [], []
        self.experience, self.batch_rewards = [], []
        self.rolling_r = RunningStats()

        self.goodTrigger = False 
        self.lastGoodYBallLoc = 0
        self.ballVisibleSteps = 0

        ### Need this to pass information ###
        self.a = 0
        self.v = 0
        self.t = 0

        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA 
        self.BATCH = BATCH 

        self.init_training()

        # Flag which indicates when the agent is learning from past experiences
        self.training = False
        return

    #Should be called at the beginning of every episode
    def init_training(self):
        # Zero the LSTM state
        self.lstm_state = self.ppo.sess.run([self.ppo.pi_eval_i_state, self.ppo.vf_eval_i_state])
        self.ep_r = 0
        self.ep_t = 0
        self.ep_a = []
        self.ballVisibleSteps = 0
        self.s = None
        self.graph_summary = None
        return

    #Should be called at the end of an episode to update the PPO policy using collected experiences
    def update_ppo(self):  
        self.training = True

        # Normalise rewards
        rewards = np.array(self.buffer_r)
        rewards = np.clip(rewards / self.rolling_r.std, -10, 10)
        self.batch_rewards = self.batch_rewards + self.buffer_r

        v_final = [self.v * (1 - 1)]  # v = 0 if terminal, otherwise use the predicted v
        values = np.array(self.buffer_v + v_final)
        terminals = np.array(self.buffer_terminal + [1])

        # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
        delta = rewards + self.GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
        advantage = discount(delta, self.GAMMA * self.LAMBDA, terminals)
        returns = advantage + np.array(self.buffer_v)
        # Per episode normalisation of advantages
        # advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

        # TODO there is an issue here. Investigate the original code to find out what the shape is supposed to look like!!!!
        #bs = np.reshape(self.buffer_s, (len(self.buffer_s),) + self.ppo.s_dim)
        bs = np.vstack(self.buffer_s)
        ba = np.vstack(self.buffer_a)
        br = np.vstack(returns)
        badv = np.vstack(advantage)

        #bs, ba, br, badv = np.reshape(self.buffer_s, (len(self.buffer_s),) + self.ppo.s_dim), np.vstack(self.buffer_a), \
        #                np.vstack(returns), np.vstack(advantage)
        self.experience.append([bs, ba, br, badv])

        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v, self.buffer_terminal = [], [], [], [], []

        # Update ppo
        if self.t >= self.BATCH:
            # Per batch normalisation of advantages
            advs = np.concatenate(list(zip(*self.experience))[3])
            for x in self.experience:
                x[3] = (x[3] - np.mean(advs)) / np.maximum(np.std(advs), 1e-6)

            # Update rolling reward stats
            self.rolling_r.update(np.array(self.batch_rewards))

            print("Training using %i episodes and %i steps..." % (len(self.experience), self.t))
            self.graph_summary = self.ppo.update(self.experience)
            self.t, self.experience, self.batch_rewards = 0, [], []

        self.training = False       

    def calculate_reward(self, currLocation, prevLocation, controllerPenalty, terminal):
        # If the state is terminal the reward is zero as there are no possible future rewards
        if(terminal):
            return 0

        #Constant penalty to encourage actions. This will also penalise actions that did not make a big impact.
        r = -0.005

        # If the controller's trigger limiter script kicked in 
        r -= controllerPenalty

        # Check if the ball is visible
        ballVisible = not (math.isclose(currLocation[0],-1) and math.isclose(currLocation[1],-1))
        
        #If the ball is not visible return the r as is
        if not ballVisible:
            if self.a == 4:
                r += 0.005
            return r
        elif ballVisible and self.a == 4:
            r -= 0.01

        self.ballVisibleSteps += 1

        # If the good trigger flag has been set check how the ball moved since last frame
        if self.goodTrigger:
            # If the ball's location has moved higher up and the X distance moved is bigger than the threshold 
            # It is rewarded and the good trigger flag is reset
            if(sharedData.lastValidBallLocation[0] < self.lastGoodYBallLoc and self.lastGoodYBallLoc-sharedData.lastValidBallLocation[0] 
                 > sharedData.goodTriggerBallTravelDistanceThreshold): #currLocation[0] old one
                r += 3
                self.goodTrigger = False 
            # If the ball's location is lower than before it was not a good trigger and the flag is reset
            elif(sharedData.lastValidBallLocation[0]  > prevLocation[0]):
                self.goodTrigger = False
        #If we are not already evaluating a good trigger, check if the good trigger flag should be set
        elif currLocation[0] > sharedData.goodTriggerThreshold:
            #If a flipper was triggered the location of the ball at the time of trigger is recorded and the good trigger flag is set.
            if(self.a == 1 or self.a == 2 or self.a == 3):
                self.lastGoodYBallLoc = currLocation[0]
                self.goodTrigger = True
        return r

    def calculate_reward_old(self, lastLocation, controllerPenalty, terminal):
        #Constant penalty to encourage actions
        r = -0.1

        # Deduct the penalty from the pinballcontroller to discourage unwanted behviour such as trigger spamming
        r -= controllerPenalty

        distanceToCenterOfTriggers = math.sqrt((sharedData.CenterOfTriggersPoint[0]-lastLocation[0])**2 + 
                                        (sharedData.CenterOfTriggersPoint[1]-lastLocation[1])**2)

        # If the ball is not visible and an action is performed it should be penalized unless it is the start action.

        if self.goodTrigger:
            if self.lastGoodYBallLoc > lastLocation[0]:
                r += 200
            self.goodTrigger = False

        ballVisible = not (math.isclose(lastLocation[0],-1) and math.isclose(lastLocation[1],-1))

        # If it noop's and the ball is farther away from the flippers than some threshold we give a bonus
        # If it noop's and the ball is in trigger range we give a penalty 
        if(self.a == 0):
            # If the ball is far from the triggers and noop is selected
            if(ballVisible and distanceToCenterOfTriggers > sharedData.ballToCenterDistanceThreshold):
                r += 2
            elif(ballVisible): 
                r -= 1
        # If a trigger is activated and the ball is close to the centre of the flippers we give a bonus.
        # Otherwise we give a penalty 
        elif(self.a == 1 or self.a == 2 or self.a == 3):
            # If the ball is not detected and the flipper action is used
            if(not ballVisible):
                r -= 5
            # If the known location of the ball is far from the triggers
            elif(ballVisible and distanceToCenterOfTriggers > sharedData.ballToCenterDistanceThreshold):
                r -= 1
            # If the ball is close to the triggers 
            elif(ballVisible and distanceToCenterOfTriggers < sharedData.ballToCenterDistanceThreshold):
                r += 1
                if lastLocation[0] > sharedData.goodTriggerThreshold:
                    self.goodTrigger = True
                    self.lastGoodYBallLoc = lastLocation[0]
        elif(self.a == 4):
            if(ballVisible):
                r -= 10
            else:
                r += 1

        # If the state is terminal the reward is zero as there are no possible future rewards
        if(terminal):
            r = 0

        return r

    # NOTE Can call the episode start and end functions from this class if we pass episode start and end flags into the queue
    #episodeState: 0 = new episode, 1 = same episode, 2 = end episode
    def step_environment(self, ballLocationHistory, terminal, episodeState, episode):
        with tf.variable_scope('RLModel'):
            if episodeState == 0: #New episode started
                print("Starting new episode")
                self.init_training()
                self.s = ballLocationHistory
            elif episodeState == 2: #Terminal state
                sharedData.RLTraining = True
                print("Updating PPO")
                self.pinballController.clearActions()
                self.update_ppo()
                print("Resetting the pinball machine")
                self.pinballController.resetMachine()
                sharedData.RLTraining = False
            
            #Update the state
            self.s = ballLocationHistory
            #Evaluate the state
            self.a, self.v, self.lstm_state = self.ppo.evaluate_state(self.s, self.lstm_state)

            self.buffer_s.append(self.s)
            self.buffer_a.append(self.a)
            self.buffer_v.append(self.v)
            self.buffer_terminal.append(terminal)
            self.ep_a.append(self.a)

            # Control pinball machine here
            if sharedData.gameOver:
                self.pinballController.clearActions()
                controllerPenalty = 0
            else:
                #Acion: 0 = noop, 1 = left trigger, 2 = right trigger, 3 = both triggers, 4 = start button
                controllerPenalty = self.pinballController.triggerAction(self.a)

            prevLocation = (ballLocationHistory[ballLocationHistory.size-4],ballLocationHistory[ballLocationHistory.size-3])
            currLocation = (ballLocationHistory[ballLocationHistory.size-2],ballLocationHistory[ballLocationHistory.size-1])

            r = self.calculate_reward(currLocation, prevLocation, controllerPenalty, terminal)

            self.buffer_r.append(r)

            sharedData.currentAction = self.a 
            sharedData.currentReward = r 

            self.ep_r += r
            self.ep_t += 1
            sharedData.episodeStep = self.ep_t
            self.t += 1

            if episodeState == 2 and not self.graph_summary==None:
                self.write_end_of_episode_summary(episode)
                # Save the model
                if episode % 50 == 0 and episode > 0:
                    self.save_models(episode)

            return self.ep_r, self.a

    def write_end_of_episode_summary(self, episode):
        # End of episode summary
        print('Episode: %i' % episode, "| Reward: %.2f" % self.ep_r, '| Steps: %i' % self.ep_t)

        worker_summary = tf.Summary()
        worker_summary.value.add(tag="Reward", simple_value=self.ep_r)
        worker_summary.value.add(tag="BallVisibleRatio", simple_value=self.ballVisibleSteps/self.ep_t)

        # Create Action histograms for each dimension
        actions = np.array(self.ep_a)
        if self.ppo.discrete:
            add_histogram(self.ppo.writer, "Action", actions, episode, bins=self.ppo.a_dim)
        else:
            for a in range(self.ppo.a_dim):
                add_histogram(self.ppo.writer, "Action/Dim" + str(a), actions[:, a], episode)

        try:
            self.ppo.writer.add_summary(self.graph_summary, episode)
        except NameError:
            pass
        self.ppo.writer.add_summary(worker_summary, episode)
        self.ppo.writer.flush()

    def save_models(self, episode):
        path = self.ppo.save_model(self.SUMMARY_DIR, episode)
        print('Saved model at episode', episode, 'in', path)

    def stop_training(self):
        self.training = False 

    def __del__(self):
        self.pinballController.cleanup() 
        #Cleanup here
        del self.ppo 
