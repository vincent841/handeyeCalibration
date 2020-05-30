
import indydcp_client as indycli
import pyrealsense2 as rs
import numpy as np 
import sys
from time import sleep
import cv2
import math
import UtilHM
import os
import datetime
import CalibHandEye
import cv2.aruco as aruco

# calibration parameters
UseRealSenseInternalMatrix = True
VideoFrameWidth = 1280
VideoFrameHeight = 720
VideoFramePerSec = 30

# set robot parameters
_server_ip = "192.168.1.207"
_name = "NRMK-Indy7"

# set chessboard calibration parameters
chessboardSize = [10, 7]
objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# create handeye calibration parameters
HMRobotbaseToTCP = []
HMCalibbaseToCam = []
Cam3dCoord = []
Robot3dCoord = []

cam3DPoints = []
robot3DPoints = []

# opencv parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)          # termination criteria

'''
#####################################################################################
Robot Control/Status Functions
#####################################################################################
'''
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
    if( status['direct_teaching'] == True):
        indy.direct_teaching(False)
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
    task_pos_mm = [task_pos[0], task_pos[1], task_pos[2],task_pos[3], task_pos[4], task_pos[5]]
    print ("Task Pos: ")
    print (task_pos_mm) 
    # hm = UtilHM.convertXYZABCtoHMDeg(task_pos)
    # print("Homogeneous Matrix: ")
    # print(hm)

def indyGetCurrentHMPose():
    task_pos = indy.get_task_pos()
    hm = UtilHM.convertXYZABCtoHMDeg(task_pos)
    return hm

def indyGetTaskPose():
    task_pos = indy.get_task_pos()
    return task_pos    


'''
#####################################################################################
Realsense Camera Control/Status Functions
#####################################################################################
'''
# initialize a realsense camera
def initializeRealsense():
    #Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, VideoFrameWidth, VideoFrameHeight, rs.format.z16, VideoFramePerSec)
    config.enable_stream(rs.stream.color, VideoFrameWidth, VideoFrameHeight, rs.format.bgr8, VideoFramePerSec)

    # Start streaming
    pipeline.start(config)
    return pipeline

# covert realsense intrisic data to camera matrix
def convertIntrinsicsMat(intrinsics):
    mtx = np.array([[intrinsics.fx,             0, intrinsics.ppx],
                    [            0, intrinsics.fy, intrinsics.ppy],
                    [            0,             0,              1]])
    
    dist = np.array(intrinsics.coeffs[:4])
    return mtx, dist


'''
#####################################################################################
Utility Functions
#####################################################################################
'''
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

def drawText(img, text, imgpt):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, text, imgpt, font, 1, (0,255,0),1,cv2.LINE_AA)



'''
#####################################################################################
Transform Matrix Functions
#####################################################################################
'''
def findTransformMatrix(color_image, dirFrameImage, mtx, dist):
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

        # convert Euler angles to Homogeneous matrix
        rotmat = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec, rotmat)
        hmCalToCam = makeHM(rotmat, tvec)
        HMCalibbaseToCam.append(hmCalToCam)

        # (Optional) project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # img = drawAxis(color_image, corners2, imgpts)
        # cv2.imshow('Images with Axis',img)

        # get the current robot pose
        hmpose = indyGetCurrentHMPose()
        HMRobotbaseToTCP.append(hmpose)
    else:
        print("Failed to capture an entire chessboard image. Please try to do it again..")

# save a transform matrix to xml data as a file
def saveTransformMatrix(cam2calHM):
    calibFile = cv2.FileStorage("CalibResults.xml", cv2.FILE_STORAGE_WRITE)
    calibFile.write("cam2calHM", cam2calHM)
    calibFile.release()

'''
#####################################################################################
Mouse Event Handler
#####################################################################################
'''
def mouseEventCallback(event, x, y, flags, param):
    aligned_depth_frame = param
    if(event == cv2.EVENT_LBUTTONUP):
        print("-------------------------------------------------------")
        print("Event... " + str(x) + " , " + str(y))

        # get a camera-based coordinates of the current image pixel point
        depth = aligned_depth_frame.get_distance(int(x), int(y))
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(x), int(y)], depth)

        # get a transformation matrix which was created by calibration process
        calibFile = cv2.FileStorage("CalibResults.xml", cv2.FILE_STORAGE_READ)
        hmnode = calibFile.getNode("cam2calHM")
        hmmtx = hmnode.mat()
        # print("Transform Matrix: ")
        # print(hmmtx)
        
        # print a camera based coordi
        camCoord = np.array(((depth_point[0], depth_point[1], depth_point[2], 1)))
        text = "Camera Coord: %.5lf, %.5lf, %.5lf" % (depth_point[0], depth_point[1], depth_point[2])
        print(text)

        robotCoord = np.dot(camCoord, hmmtx)
        print("Transfomred Robot-based Coordinate: ")
        print(robotCoord)


###############################################################################
# Hand-eye calibration process 
#   -                                                                
###############################################################################

if __name__ == '__main__':

    # connect to Indy
    indy = indyConnect(_server_ip, _name)
    if(indy == None):
        print("Can't connect the robot and exit this process..")
        sys.exit()

    # intialize the robot
    print("Intialize the robot...")
    initialzeRobot(indy)
    sleep(1)

    # set direct-teaching mode on
    # print("Entering HandEye Calibartion Mode with direct teaching mode...")
    # indy.direct_teaching(True)
    sleep(1)

    # ready to capture frames for realsense camera
    pipeline = initializeRealsense()

    # create a directory to image files
    #dirFrameImage = makeFrameImageDirectory()

    # creates an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    cv2.namedWindow('Capture Images')

    # create a variable for frame indexing
    idxFrame = 0

    # use created camera matrix 
    if(UseRealSenseInternalMatrix == False):
        calibFile = cv2.FileStorage("calibData.xml", cv2.FILE_STORAGE_READ)
        cmnode = calibFile.getNode("cameraMatrix")
        mtx = cmnode.mat()
        dcnode = calibFile.getNode("distCoeff")
        dist = dcnode.mat()

    # start to capture frames and process a key event
    try:
        while(True):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            # get internal intrinsics & extrinsics in D435
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
            if(UseRealSenseInternalMatrix == True):    
                mtx, dist = convertIntrinsicsMat(color_intrin)
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # operations on the frame
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # set dictionary size depending on the aruco marker selected
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

            # detector parameters can be set here (List of detection parameters[3])
            parameters = aruco.DetectorParameters_create()
            parameters.adaptiveThreshConstant = 10

            aruco_list = {}

            # lists of ids and the corners belonging to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # check if the ids list is not empty
            # if no check is added the code will crash
            if np.all(ids != None):
                # get the center of an Aruco Makers
                # if len(corners):
                #     for k in range(len(corners)):
                #         temp_1 = corners[k]
                #         temp_1 = temp_1[0]
                #         temp_2 = ids[k]
                #         temp_2 = temp_2[0]
                #         aruco_list[temp_2] = temp_1
                # key_list = aruco_list.keys()
                # for key in key_list:
                #     dict_entry = aruco_list[key]    
                #     centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
                #     centre[:] = [int(x / 4) for x in centre]
                #     orient_centre = centre + [0.0,5.0]
                #     centre = tuple(centre)  
                #     orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
                #     #cv2.circle(color_image,centre,1,(0,0,255), -1)

                # estimate pose of each marker and return the values
                # rvet and tvec-different from camera coefficients
                rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

                # get the center of an Aruco Makers
                if((rvec[0].shape == (1,3)) or (rvec[0].shape == (3,1))):
                    inputObjPts = np.float32([[0.0,0.0,0.0]]).reshape(-1,3)
                    imgpts, jac = cv2.projectPoints(inputObjPts, rvec[0], tvec[0], mtx, dist)
                    centre = tuple(imgpts[0][0])
                    cv2.circle(color_image,centre,1,(0,0,255), -1)
                    
                #(rvec-tvec).any() # get rid of that nasty numpy value array error

                # for i in range(0, ids.size):
                    # draw axis for the aruco markers
                    # aruco.drawAxis(color_image, mtx, dist, rvec[i], tvec[i], 0.1p)

                # draw a square around the markers
                aruco.drawDetectedMarkers(color_image, corners)

                # code to show ids of the marker found
                strg = ''
                for i in range(0, ids.size):
                    strg += str(ids[i][0])+', '
            else:
                # code to show 'No Ids' when no markers are found
                #cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
                pass   

            # update text every 0.5 sec.
            if((idxFrame % (VideoFramePerSec/2)) == 0):
                robotStatus = indy.get_robot_status()
                statusDT = "On" if(robotStatus['direct_teaching'] == True) else "Off"
                text = "Direct Teaching Mode: " + statusDT
                drawText(color_image, text, (5, 20))

            
            # display the captured image
            cv2.imshow('Capture Images',color_image)
            cv2.setMouseCallback('Capture Images', mouseEventCallback, aligned_depth_frame)

            # handle key inputs
            pressedKey = (cv2.waitKey(1) & 0xFF)
            if pressedKey == ord('q'):
                break
            elif pressedKey == ord('p'):
                indyPrintTaskPosition()
            elif pressedKey == ord('z'):
                print('clear calibration data..')
                cam3DPoints.clear()
                robot3DPoints.clear()
            elif pressedKey == ord('c'):
                print("---------------------------------------------------------------")
                depth = aligned_depth_frame.get_distance(centre[0], centre[1])
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [centre[0], centre[1]], depth)
                text = "Camera Coord: %.5lf, %.5lf, %.5lf" % (depth_point[0], depth_point[1], depth_point[2])
                print(text)
                cam3DPoints.append(depth_point)
                currTaskPose = indyGetTaskPose()
                print("Robot Coord: %.5lf, %.5lf, %.5lf" % (currTaskPose[0], currTaskPose[1], currTaskPose[2]))
                robot3DPoints.append(([currTaskPose[0], currTaskPose[1], currTaskPose[2]]))
            
            elif pressedKey == ord('r'):
                # finally, we try to get HM here
                # camC = np.array( ((cam3DPoints[0]), (cam3DPoints[1]), (cam3DPoints[2])) )
                # print(camC.shape)
                # robotC = np.array( ((robot3DPoints[0]), (robot3DPoints[1]), (robot3DPoints[2])) )
                # result = CalibHandEye.calculateHM(camC, robotC)
                result = CalibHandEye.calculateTransformMatrix(cam3DPoints, robot3DPoints)
                print("Transform Matrix = ")
                print(result)   
                saveTransformMatrix(result[1])         

            elif pressedKey == ord('d'):
                # set direct-teaching mode on
                print("Entering HandEye Calibartion Mode with direct teaching mode...")
                indy.direct_teaching(True)
                sleep(1)

            elif pressedKey == ord('f'):
                # set direct-teaching mode on
                print("Entering HandEye Calibartion Mode with direct teaching mode...")
                indy.direct_teaching(False)
        idxFrame += 1
    finally:
        # Stop streaming
        pipeline.stop()


    robotStatus = indy.get_robot_status()
    if( robotStatus['direct_teaching'] == True):
        indy.direct_teaching(False)
    
    # exit
    cv2.destroyAllWindows()
    indy.disconnect()
        