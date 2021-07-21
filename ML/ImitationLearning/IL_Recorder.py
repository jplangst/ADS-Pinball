### Recorder for Imitation Learning 
### Currently not used. Will move to this when we actually start recording IL data properly.

from ML.PinballController import PinballController

class IL_Recorder(object):

    def checkInputs(self):
        buttonPressedStates = PinballController.readInputPins()
