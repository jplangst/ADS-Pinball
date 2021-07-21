import numpy as np
import json
import cv2
import subprocess
import Shared.sharedData as sharedData

########### CAMERA RELATED #################
def loadCameraCalibration(filepath):
    try:
        with open(filepath) as json_file:
            calData = json.load(json_file)
            cameraMatrix = np.array(calData['cameraMatrix'])
            newCameraMatrix = np.array(calData['newCameraMatrix'])
            roi = calData['roi']
            dist = np.array(calData['dist'])
            print("Camera calibration data loaded")
            return (True, calData, cameraMatrix, newCameraMatrix, roi, dist)
    except Exception as error: 
        print('Unable to load a calibration file')
        return (False, None, None, None, None, None)

def perspectiveCorrect(frame, cameraMatrix, newCameraMatrix, dist, roi):
    x, y, w, h = roi

    # Undistort with Remapping (Faster)
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    frame = frame[y:y+h, x:x+w]

    # Perspective correct the frame using cv2 undistort (Slower)
    #frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    # crop the image
    #frame = frame[y:y+h, x:x+w]

    return frame

def rotateImage(image, degree):
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

###################### Jetson AGX Configuration ####################
def set_power_mode(power_mode):
    power_cmd0 = 'nvpmodel'
    power_cmd1 = str('-m'+str(power_mode))
    subprocess.call('sudo {} {}'.format(power_cmd0, power_cmd1), shell=True,
                    stdout=None)
    print('Setting Jetson in max performance mode')

def set_jetson_clocks():
    clocks_cmd = 'jetson_clocks'
    subprocess.call('sudo {}'.format(clocks_cmd), shell=True,
                    stdout=None)
    print("Jetson clocks are Set")

def set_jetson_fan(switch_opt):
    fan_cmd = "sh" + " " + "-c" + " " + "'echo" + " " + str(
        switch_opt) + " " + ">" + " " + "/sys/devices/pwm-fan/target_pwm'"
    subprocess.call('sudo {}'.format(fan_cmd), shell=True, stdout=None)

def restart_cam_daemon():
    subprocess.call('sudo {}'.format("sudo systemctl restart nvargus-daemon"), shell=True, stdout=None)

################# JSON state related #########################
def loadTrainingStateFile():
    try:
        with open(sharedData.stateFileFilepath) as json_file:
            stateData = json.load(json_file)
            episode = stateData['episode']
            modelCheckpointFilepath = stateData['modelCheckpointFilepath']
            checkpointsFolder = stateData['checkpointsFolder']
            print("Training state file loaded, continuing training from episode: "
                , episode, " and checkpoint file: ", modelCheckpointFilepath, " checkpoints folder: ", checkpointsFolder)
            return (True, episode, modelCheckpointFilepath, checkpointsFolder)
    except Exception as error: 
        print('Unable to locate or read the training state file')
        return (False, None, None, None)

def saveTrainingStateFile(episode, modelCheckpointFilepath, checkpointsFolder):
    jsonData = {"episode": episode,
                "modelCheckpointFilepath": modelCheckpointFilepath,
                "checkpointsFolder": checkpointsFolder
                }
    jsonObject = json.dumps(jsonData, indent=4)

    with open(sharedData.stateFileFilepath, "w") as stateFile:
        stateFile.write(jsonObject)