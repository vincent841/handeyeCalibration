
import indydcp_client as indycli
import pyrealsense2 as rs
import numpy as np 
import sys
from time import sleep
import cv2
import math
import UtilHM
import os
import CalibHandEye


def indyConnect(servIP, connName):
    # Connect
    obj = indycli.IndyDCPClient(servIP, connName)
    conResult = obj.connect()
    if conResult == False:
        print("Connection Failed")
        obj = None
    return obj

def initialzeRobot(indy):
    indy.reset_robot()
    status = indy.get_robot_status()
    print("Resetting robot")
    print("is in resetting? ", status['resetting'])
    print("is robot ready? ", status['ready'])

    if( status['emergency'] == True):
        indy.stop_emergency()
    sleep(5)
    status = indy.get_robot_status()
    print("Reset robot done")
    print("is in resetting? ", status['resetting'])
    print("is robot ready? ", status['ready'])

def indyPrintJointPosition():
    print('### Test: GetJointPos() ###')
    joint_pos = indy.get_joint_pos()
    print ("Joint Pos: ")
    print (joint_pos)    

def indyPrintTaskPosition():
    task_pos = indy.get_task_pos()
    task_pos_mm = [task_pos[0]*1000.0, task_pos[1]*1000.0, task_pos[2]*1000.0,task_pos[3], task_pos[4], task_pos[5]]
    print ("Task Pos: ")
    print (task_pos_mm) 
    hm = UtilHM.convertHMtoXYZABCDeg(task_pos)
    print("Homogeneous Matrix: ")
    print(hm)

def indyGetCurrentHMPose():
    task_pos = indy.get_task_pos()
    hm = UtilHM.convertHMtoXYZABCDeg(task_pos)
    return hm
    
# # Get Task Position
# print('### Test: GetTaskPos() ###')
# task_pos = indy.get_task_pos()
# print ("Task Pos: ")
# print (task_pos)

# # Get Joint Position
# print('### Test: GetJointPos() ###')
# joint_pos = indy.get_joint_pos()
# print ("Joint Pos: ")
# print (joint_pos)

# # Move to Task
# print('### Test: MoveToT() ###')
# indy.task_move_to(task_pos)

# # Move to Joint
# print('### Test: MoveToJ() ###')
# indy.joint_move_to(joint_pos)

# initialize a realsense camera
def initializeRealsense():
    #Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 760, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 760, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline

# create a directory to save captured images 
def makeFrameImageDirectory():
    now = datetime.datetime.now()
    dirString = now.strftime("%Y%m%d%H%M%S")
    try:
        if not(os.path.isdir(dirString)):
            os.makedirs(os.path.join(dirString))
    except OSError as e:
        print("Can't make the directory: %s" % dirFrameImage)
        raise
    return dirString

def drawAxis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def processCapture(color_image, dirFrameImage):
    # save the current frame to a jpg file
    cv2.imwrite(os.path.join(dirFrameImage, str(iteration) + '.jpg'), color_image)
    print('Image caputured - ' + os.path.join(dirFrameImage, str(iteration) + '.jpg'))

    # get the calibration pose
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (chessboardSize[0], chessboardSize[1]), None)
    if ret == True:
        # fix the coordinates of corenres
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvec, tvec, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        rotmat = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec, rotmat)
        hmCalToCam = makeHM(rotmat, tvec)
        HMCalibbaseToCam.append(hmCalToCam)

        # (Optional) project 3D points to image plane
        #imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #img = drawAxis(color_image, corners2, imgpts)
        #cv2.imshow('Images with Axis',img)

        # get the current robot pose
        hmpose = indyGetCurrentHMPose()
        HMRobotbaseToTCP.append(hmpose)
    else:
        print("Failed to capture an entire chessboard image. Please try to do it again..")

###############################################################################
# Hand-eye calibration process 
#   -                                                                
###############################################################################
if __name__ == '__main__':

    # set robot parameters
    _server_ip = "192.168.1.207"
    _name = "NRMK-Indy7"

    # set chessboard calibration parameters
    chessboardSize = [10, 7]
    objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    
    # set handeye calibration parameters
    HMRobotbaseToTCP = []
    HMCalibbaseToCam = []

    # connect to Indy
    indy = indyConnect(_server_ip, _name)
    if(indy == None):
        print("Can't connect the robot and exit this process..")
        sys.exit()

    # intialize the robot
    print("Intialize the robot...")
    initialzeRobot(indy)
    sleep(2)

    # set direct-teaching mode on
    print("Entering HandEye Calibartion Mode with direct teaching mode...")
    indy.direct_teaching(True)
    sleep(1)

    # ready to capture frames for realsense camera
    pipeline = initializeRealsense()

    # create a directory to image files
    dirFrameImage = makeFrameImageDirectory()

    # start to capture frames and process a key event
    try:
        while(True):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())        
            
            # display the captured image
            cv2.imshow('Capture Images',color_image)

            # handle key inputs
            pressedKey = (cv2.waitKey(1) & 0xFF)
            if pressedKey == ord('q'):
                break
            elif pressedKey == ord('p'):
                indyPrintTaskPosition()
            elif pressedKey == ord('c'):
                processCapture(color_image, dirFrameImage)
    finally:
        # Stop streaming
        pipeline.stop()

    indy.direct_teaching(False)

    CalibHandEye.calibrateHandEye(HMRobotbaseToTCP, HMCalibbaseToCam, False)
    print("Calibration Finished")
    
    # exit
    cv2.destroyAllWindows()
    indy.disconnect()
        