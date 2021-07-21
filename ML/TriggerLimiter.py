# Code to make sure the flippers on the pinball machine is not pressed for too long

import time
from collections import deque

class TriggerLimiter():
    #minOffTime - seconds minimum not being pressed.
    #minOnTime - seconds minimum being pressed.
    #maxOnTime - seconds continuosly on. Must be less than maxOnTimeOverTime.
    #maxOnTimeOverTime - seconds on over overTime. Must be less than overTime.
    #overTime - min time with maxOnTimeOver.
    def __init__(self, minOffTime=0.2, minOnTime=0.1, maxOnTime=2, maxOnTimeOverTime=2, 
        overTime=4):

        self.minOffTime = minOffTime 
        self.minOnTime = minOnTime 
        self.maxOnTime = maxOnTime 
        self.maxOnTimeOverTime = maxOnTimeOverTime 
        self.overTime = overTime 
        self.overTimeHysterese = self.maxOnTimeOverTime / 2  

        self.elems = deque()
        self.currTimeOnSum = 0
        self.lastChangeTime = 0

    def limitOutput(self, newVal):
        retFlag = 0

        t = time.perf_counter()
        tOld = t - self.overTime
        while self.elems:
            elem = self.elems[0]
            tt = elem['time']
            if (tt >= tOld):
                break
            if elem['onDt']:
                dt = elem['onDt']
                prevOnTimeSum = self.currTimeOnSum
                self.currTimeOnSum -= dt
                if (self.currTimeOnSum < self.maxOnTimeOverTime) and (prevOnTimeSum >= self.maxOnTimeOverTime):
                    self.currTimeOnSum -= self.overTimeHysterese
            self.elems.popleft()

        if self.elems:
            elem = self.elems[-1]
            lastV = elem['val']
            lastT = elem['time']
        else:
            lastV = 0
            lastT = t

        if (lastV != 0):
            if (newVal != 0):
                if ((t - self.lastChangeTime) > self.maxOnTime):
                    newVal = 0  # Overriding output value
                    retFlag = 2
            else:
                if ((t - self.lastChangeTime) < self.minOnTime):
                    newVal = 1  # Overriding output value
                    retFlag = 1
            dt = t - lastT
            prevOnTimeSum = self.currTimeOnSum
            self.currTimeOnSum += dt
            if (newVal != 0) and (self.currTimeOnSum > self.maxOnTimeOverTime):
                newVal = 0
                retFlag = 2
                # print("Total on more than", maxOnTimeOverTime, "seconds", "sumOnT", self.currTimeOnSum)
                if (prevOnTimeSum <= self.maxOnTimeOverTime):
                    self.currTimeOnSum += self.overTimeHysterese
        else:
            if (newVal != 0):
                if ((t - self.lastChangeTime) < self.minOffTime):
                    newVal = 0  # Overriding output value
                    retFlag = 1
                    # print("Off less than", minOffTime, "seconds", "sumOnT", self.currTimeOnSum)
                elif (self.currTimeOnSum + self.minOnTime) > self.maxOnTimeOverTime:
                    newVal = 0  # Overriding output value
                    retFlag = 2
                    # print("Total would be on more than", maxOnTimeOverTime, "seconds", "sumOnT", self.currTimeOnSum)
                    if (self.currTimeOnSum <= self.maxOnTimeOverTime):
                        self.currTimeOnSum += self.overTimeHysterese
            dt = 0
        
        if (lastV == 0):
            if (newVal != 0):
                self.lastChangeTime = t
        else:
            if (newVal == 0):
                self.lastChangeTime = t

        self.elems.append({ 'time': t, 'val': newVal, 'onDt': dt })
        if (self.lastChangeTime == 0):
            lt = 0
        else:
            lt = t - self.lastChangeTime
        return newVal, lt, retFlag
