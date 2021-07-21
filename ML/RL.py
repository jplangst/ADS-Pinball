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
from PinballUtils import *

class RL_Controller(object):
    def __init__(self, OUTPUT_RESULTS_DIR='.', MODEL_RESTORE_PATH=None, CHECKPOINT_DIR=None, GAMMA=0.99, LAMBDA=0.95, BATCH=8192):
        ENVIRONMENT = 'PinballMachine'

        TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

        if CHECKPOINT_DIR is not None:
            self.SUMMARY_DIR = CHECKPOINT_DIR
        else:
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
        self.lastGoodXBallLoc = 0
        self.ballVisibleSteps = 0

        #Used in the good ball trigger check
        self.prevGoodBallXloc = 1

        self.goodTriggerCount = 0
        self.goodTriggerRewardTimes = 0

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

    def resetEpisodeVars(self):
        self.goodTrigger = False
        self.lastGoodYBallLoc = 0
        self.lastGoodXBallLoc = 0
        self.goodTriggerCount = 0
        self.goodTriggerRewardTimes = 0
        self.prevGoodBallXloc = 1

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
                r += 0.006
            return r
        
        if self.a == 4:
            r -= 0.03

        self.ballVisibleSteps += 1

        # If the good trigger flag has been set check how the ball moved since last frame
        if self.goodTrigger:
            if(currLocation[0] < self.prevGoodBallXloc):
                distance = math.sqrt(math.pow(currLocation[0]-self.lastGoodYBallLoc, 2) + math.pow(currLocation[1]-self.lastGoodXBallLoc,2))
                if(distance > sharedData.goodTriggerBallTravelDistanceThreshold):
                    r += 3
                    self.goodTrigger = False 
                    self.goodTriggerRewardTimes += 1

                self.prevGoodBallXloc = currLocation[0]
            else: 
                self.goodTrigger = False
            
        #If we are not already evaluating a good trigger, check if the good trigger flag should be set
        elif currLocation[0] > sharedData.goodTriggerThreshold:
            #If a flipper was triggered the location of the ball at the time of trigger is recorded and the good trigger flag is set.
            if(self.a == 1 or self.a == 2 or self.a == 3):
                self.lastGoodYBallLoc = currLocation[0]
                self.lastGoodXBallLoc = currLocation[1]
                self.prevGoodBallXloc = 1
                self.goodTrigger = True
                self.goodTriggerCount += 1
        #print("GT: ", self.goodTrigger, "Loc: ", self.lastGoodYBallLoc)
        #print("GT: ", self.goodTriggerCount, " GRC: ", self.goodTriggerRewardTimes)
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

                # reset episode variables
                self.resetEpisodeVars()

                # Save the model
                if episode % 50 == 0 and episode > 0:
                    self.save_models(episode)

            return self.ep_r, self.a

    def write_end_of_episode_summary(self, episode):
        # End of episode summary
        print('Episode: %i' % episode, "| Reward: %.2f" % self.ep_r, '| Steps: %i' % self.ep_t)

        worker_summary = tf.Summary()
        worker_summary.value.add(tag="Reward", simple_value=self.ep_r)
        worker_summary.value.add(tag="GoodTriggerCount", simple_value=self.goodTriggerRewardTimes)
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
        saveTrainingStateFile(episode, path, self.SUMMARY_DIR)
        print('Saved models and the training state. Episode:', episode, 'in', path)

    def stop_training(self):
        self.training = False 

    def __del__(self):
        self.pinballController.cleanup() 
        #Cleanup here
        del self.ppo 
