import Jetson.GPIO as GPIO
import numpy as np
import time
import math

from ML.TriggerLimiter import TriggerLimiter

class PinballController():
    def __init__(self):
        self.left_flipper_elems = TriggerLimiter(minOffTime=0.2, minOnTime=0.1, maxOnTime=2, maxOnTimeOverTime=2,overTime=4)
        self.right_flipper_elems = TriggerLimiter(minOffTime=0.2, minOnTime=0.1, maxOnTime=2, maxOnTimeOverTime=2,overTime=4)
        self.start_button_elems = TriggerLimiter(minOffTime=0.5, minOnTime=0.2, maxOnTime=0.4, maxOnTimeOverTime=0.5,overTime=4)

        ### Pins for controlling the pinball machine buttons
        self.left_flipper_out = 32
        self.start_button_out = 33
        self.right_flipper_out = 37

        ### Pins for recording the pinball machine buttons when they are pressed
        self.left_flipper_in = 16 
        self.start_button_in = 22
        self.right_flipper_in = 18        

        # Set so that we use the pin numbers inside the circles as reference
        # To use the gpio422 for instance change to mode GPIO.BCM 
        GPIO.setmode(GPIO.BOARD)

        # Initialize the pin states to the default off values
        GPIO.setup(self.start_button_out, GPIO.OUT, initial=0)
        GPIO.setup(self.left_flipper_out, GPIO.OUT, initial=0)
        GPIO.setup(self.right_flipper_out, GPIO.OUT, initial=0)

        GPIO.setup(self.left_flipper_in, GPIO.IN)
        GPIO.setup(self.start_button_in, GPIO.IN)
        GPIO.setup(self.right_flipper_in, GPIO.IN)
        return

    def readInputPins(self):
        leftPressed = GPIO.input(self.left_flipper_in)       
        rightPressed = GPIO.input(self.right_flipper_in)
        startPressed = GPIO.input(self.start_button_in)

        buttonPressedStates = [leftPressed, rightPressed, startPressed]

        buttonPressState = 0 #No buttons pressed

        # Actions: (0) Noop, (1) Left, (2) Right, (3) Both, (4) Start
        if buttonPressedStates[0] == 1 and buttonPressedStates[1] == 1: #Left and Right buttons pressed
            buttonPressState = 3
        elif buttonPressedStates[0] == 1: #Left button pressed
            buttonPressState = 1
        elif buttonPressedStates[1] == 1: #Right button pressed
            buttonPressState = 2
        elif buttonPressedStates[2] == 1: #Right button pressed
            buttonPressState = 4

        #return [leftPressed, startPressed, rightPressed]
        return buttonPressState

    def resetMachine(self):
        GPIO.output(self.left_flipper_out, 1)
        GPIO.output(self.right_flipper_out, 1)
        time.sleep(4)
        GPIO.output(self.start_button_out, 1)
        time.sleep(0.2)

    def clearActions(self):
        GPIO.output(self.left_flipper_out, 0)
        GPIO.output(self.right_flipper_out, 0)
        GPIO.output(self.start_button_out, 0)

    def checkRetState(self, retState):
        penalty = 0
        if retState == 1:
            penalty = 0.01
        elif retState == 2:
            penalty = 1
        return penalty


    # The action input an integer with one of the following values.
    # Action: 0 = Noop, 1 = Left Flipper, 2 = Right Flipper, 3 = Both Flippers, 4 = Start Button
    def triggerAction(self, action):
        retPenalty = 0

        actions = np.zeros(3)#[0,0,0]
        if action == 1:
            actions[0] = 1
            retPenalty += 0.005
        elif action == 2:
            actions[1] = 1
            retPenalty += 0.005
        elif action == 3:
            actions[0] = 1
            actions[1] = 1
            retPenalty += 0.005
        elif action == 4:
            actions[2] = 1
            retPenalty += 0.005

        gpioCommands = np.zeros(3)#[0,0,0]
        #Safety check to ensure we do not fry the pinball connectors
        gpioCommands[0],_,ret_state0 = self.left_flipper_elems.limitOutput(actions[0])
        gpioCommands[1],_,ret_state1 = self.right_flipper_elems.limitOutput(actions[1])
        gpioCommands[2],_,ret_state2 = self.start_button_elems.limitOutput(actions[2])

        retPenalty += self.checkRetState(ret_state0)
        retPenalty += self.checkRetState(ret_state1)
        retPenalty += self.checkRetState(ret_state2)
        retPenalty = min(retPenalty,1)

        #To further decentivise such behavior a penalty is added any time the script has to override the desired policy action
        if(not (gpioCommands==actions).all()):
            GPIO.output(self.left_flipper_out, 0)
            GPIO.output(self.right_flipper_out, 0)
            GPIO.output(self.start_button_out, 0)
        else:
            GPIO.output(self.left_flipper_out, gpioCommands[0])
            GPIO.output(self.right_flipper_out, gpioCommands[1])
            GPIO.output(self.start_button_out, gpioCommands[2])

        return retPenalty

    def cleanup(self):
        #Reset all the pins to their default modes and values when we are done
        GPIO.cleanup()
        return
