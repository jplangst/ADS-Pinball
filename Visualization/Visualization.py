import threading
import time

import numpy as np
import cv2 as cv

import Shared.sharedData as sharedData 

#Visualisation thread 
    #Uses the selected action and the current video frame to visualise the agents thinking

# This file is responsible for visualizing the current state of the playfield, including the location of the ball as discovered by the neural network.
# The agent the action chose is displayed along with the probabilities it calculate dof taking each action. 
# The current observation frame, the coordinates of the located ball and the selected action are given as input to update the visualisation. 

def drawAction(im, pos, selectedAction):
    if (selectedAction):
        col = (0,255,0)
    else:       
        col = (0,0,255)
    r = int(selectedAction * 10 + 3)
    cv.circle(im, pos, r, col, -1)

def pinballVis(im, x, y, score, selectedAction, episode):
    sc = 1.5
    # im scaled to 768 x 384
    w,h,_ = im.shape
    h = int(h * sc)
    w = int(w * sc)
    bX = int(x * sc)
    bY = int(y * sc)
    bR = 13

    im1 = np.zeros((900,580,3), dtype=np.uint8)
    im2 = np.rot90(im, k=-1)
    im3 = cv.resize(im2,(w,h))
    imX, imY = 100, 100

    im1[imY:h+imY,imX:w+imX] = im3
    if (x>=0) and (y>=0):
        cv.circle(im1, (bX+imX,bY+imY), bR, (0,255,255), 2)
    font = cv.FONT_HERSHEY_SIMPLEX



    cv.putText(im1, "E: "+ str(episode)+ " ER: "+("%.3f"%score), (50,40), font, 0.8, (255,255,255), 2, cv.LINE_AA)

    if not sharedData.RLTraining:
        cv.putText(im1, "A: "+ str(sharedData.currentAction) + " Step: " +str(sharedData.episodeStep) + 
            " R: "+("%.3f"%sharedData.currentReward), (50,80), font, 0.8, (255,255,255), 2, cv.LINE_AA)

    # Actions: (0) Noop, (1) Left, (2) Right, (3) Both, (4) Start
    # If noop draw all as red
    # If left or right draw left or right as green respectively
    # If Both then draw both left and right green
    # If start then draw start as green

    drawAsGreen = [False, False, False]
    if selectedAction == 1:
        drawAsGreen[0] = True
    elif selectedAction == 2:
        drawAsGreen[1] = True
    elif selectedAction == 3:
        drawAsGreen[0] = True
        drawAsGreen[1] = True
    elif selectedAction == 4:
        drawAsGreen[2] = True

    drawAction(im1, (imX - 10, int(imY + h*0.9)), drawAsGreen[0]) # Left
    drawAction(im1, (imX + w + 10, int(imY + h*0.9)), drawAsGreen[1]) # Right
    drawAction(im1, (int(imX + w / 2), int(imY + h + 10)), drawAsGreen[2]) #Start

    # NOTE - Visualise the centre of the flippers. Can remove when verified the location is correct
    cv.circle(im1, (int(imX + w*sharedData.CenterOfTriggersPoint[1]), 
        int(imY + h*sharedData.CenterOfTriggersPoint[0])), 3, (255,0,0), -1)

    #Draw the distance radius
    #cv.ellipse(im1, (int(imX+sharedData.CenterOfTriggersPoint[1]*w),int(imY+sharedData.CenterOfTriggersPoint[0]*h)), 
    #    (int(w*sharedData.ballToCenterDistanceThreshold),int(h*sharedData.ballToCenterDistanceThreshold)),0,0,360,(255,0,0),1)

    #Draw the good trigger threshold
    cv.line(im1, (imX,int(imY+h*sharedData.goodTriggerThreshold)),
        (imX+w,int(imY+h*sharedData.goodTriggerThreshold)), (255,0,0), 1)

    #Draw the game over threshold
    #cv.line(im1, (imX,int(imY+h*sharedData.gameOverBallLocationThreshold)),
    #    (imX+w,int(imY+h*sharedData.gameOverBallLocationThreshold)), (125,125,0), 1)

    return(im1)
  
class VisualizationThread(threading.Thread):
    def __init__(self, target=None, name=None, imageWidth=512, imageHeight=256):
        super(VisualizationThread,self).__init__()
        self.target = target
        self.name = name
        self.processing = True
        self.imgW = imageWidth
        self.imgH = imageHeight

        self.newEpisodeRecording = False 

    def recordEpisode(self, episode, frame):
        if episode % sharedData.episodeRecordingInterval == 0:
            if sharedData.gameOver:
                if self.newEpisodeRecording == True:
                    print("Stopping recording")
                    self.newEpisodeRecording = False
                    sharedData.recordEpisodeFramesQueue.put([None, 2, episode])
            else:      
                if not self.newEpisodeRecording:
                    episodeState = 0
                    self.newEpisodeRecording = True
                else:
                    episodeState = 1
                sharedData.recordEpisodeFramesQueue.put([frame, episodeState, episode])
 

    def run(self):
        while self.processing: 
            while(not sharedData.pinballVisQueue.empty()):
                visData = sharedData.pinballVisQueue.get_nowait()

                frame, ballLocation, currentTotalReward, action, episodeString, episode = visData

                playAreaImg = cv.resize(frame,(self.imgW,self.imgH))

                #Map the normalized ball location coordinates to corresponding image pixels
                bX, bY = ballLocation
                bX = bX * self.imgW
                bY = bY * self.imgH

                visFrame = pinballVis( playAreaImg, self.imgH-1-int(bY), int(bX), currentTotalReward, action, episodeString)
                
                cv.imshow("Pinball AI Vis", visFrame)

                #Episode recording
                self.recordEpisode(episode, visFrame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    sharedData.readingVideoFrames = False
                    print("Q pressed, stopping application")

            time.sleep(0.000001) 
        cv.destroyAllWindows()
        print("Visualization thread stopped")
        return
    
    def stopProcessing(self):
        print("Stopping visualization thread")
        self.processing = False