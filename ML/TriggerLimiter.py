# Code to make sure the flippers on the pinball machine is not pressed for too long

import time
from collections import deque

minOffTime = 0.2 # seconds minimum not being pressed.
minOnTime = 0.1 # seconds minimum being pressed.
maxOnTime = 2 # seconds continuosly on. Must be less than maxOnTimeOverTime.
maxOnTimeOverTime = 2 # seconds on over 1 min. Must be less than overTime.
overTime = 4 # 1 min time with maxOnTimeOver.
overTimeHysterese = maxOnTimeOverTime / 2  # Must be less than maxOnTimeOverTime

def initOutputLimit():
    return { 'elems': deque(), 'currTimeOnSum': 0, 'lastChangeTime': 0 }


def limitOutput( elemList, newVal ):
    retFlag = 0
    onTimeSum = elemList['currTimeOnSum']
    lastChTime = elemList['lastChangeTime']
    t = time.perf_counter()
    tOld = t - overTime
    while elemList['elems']:
        elem = elemList['elems'][0]
        tt = elem['time']
        if (tt >= tOld):
            # print("End del elems, tt:", tt, "tOld:", tOld)
            break
        if elem['onDt']:
            dt = elem['onDt']
            # print("Rem seconds:", dt)
            prevOnTimeSum = onTimeSum
            onTimeSum -= dt
            if (onTimeSum < maxOnTimeOverTime) and (prevOnTimeSum >= maxOnTimeOverTime):
                onTimeSum -= overTimeHysterese
        elemList['elems'].popleft()
        # print("Removed elem")

    if elemList['elems']:
        elem = elemList['elems'][-1]
        lastV = elem['val']
        lastT = elem['time']
    else:
        lastV = 0
        lastT = t

    if (lastV != 0):
        if (newVal != 0):
            if ((t - lastChTime) > maxOnTime):
                newVal = 0  # Overriding output value
                retFlag = 2
                # print("On more than", maxOnTime, "seconds", "sumOnT", onTimeSum)
        else:
            if ((t - lastChTime) < minOnTime):
                newVal = 1  # Overriding output value
                retFlag = 1
                # print("On less than", minOnTime, "seconds", "sumOnT", onTimeSum)
        dt = t - lastT
        prevOnTimeSum = onTimeSum
        onTimeSum += dt
        if (newVal != 0) and (onTimeSum > maxOnTimeOverTime):
            newVal = 0
            retFlag = 2
            # print("Total on more than", maxOnTimeOverTime, "seconds", "sumOnT", onTimeSum)
            if (prevOnTimeSum <= maxOnTimeOverTime):
                onTimeSum += overTimeHysterese
    else:
        if (newVal != 0):
            if ((t - lastChTime) < minOffTime):
                newVal = 0  # Overriding output value
                retFlag = 1
                # print("Off less than", minOffTime, "seconds", "sumOnT", onTimeSum)
            elif (onTimeSum + minOnTime) > maxOnTimeOverTime:
                newVal = 0  # Overriding output value
                retFlag = 2
                # print("Total would be on more than", maxOnTimeOverTime, "seconds", "sumOnT", onTimeSum)
                if (onTimeSum <= maxOnTimeOverTime):
                    onTimeSum += overTimeHysterese
        dt = 0

    elemList['currTimeOnSum'] = onTimeSum
    
    if (lastV == 0):
        if (newVal != 0):
            elemList['lastChangeTime'] = t
    else:
        if (newVal == 0):
            elemList['lastChangeTime'] = t

    elemList['elems'].append({ 'time': t, 'val': newVal, 'onDt': dt })
    if (lastChTime == 0):
        lt = 0
    else:
        lt = t - lastChTime
    return newVal, lt, retFlag
