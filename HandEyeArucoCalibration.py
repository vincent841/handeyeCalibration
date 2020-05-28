
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import UtilHM
import math

calibFile = cv2.FileStorage("calibData.xml", cv2.FILE_STORAGE_READ)
cmnode = calibFile.getNode("cameraMatrix")
mtx = cmnode.mat()
dcnode = calibFile.getNode("distCoeff")
dist = dcnode.mat()


HMTCPtoRobot = []
HMCalibbaseToCam = []


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(ids)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # # estimate pose of each marker and return the values
        # # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

    else:
        # code to show 'No Ids' when no markers are found
        #cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        pass

    # display the resulting frame
    cv2.imshow('frame',frame)
    pressedKey = (cv2.waitKey(1) & 0xFF)
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('p'):
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        hmCamToCal = UtilHM.inverseHM(hmCalToCam)
        print("-----------------------------------------")
        print(hmCalToCam)
        print()
        print(hmCamToCal)
        print()
        print(np.dot(hmCalToCam, hmCamToCal))

    elif pressedKey == ord('t'):
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        hmCamToCal = UtilHM.inverseHM(hmCalToCam)
        print("-----------------------------------------")
        print(hmCalToCam)
        print()
        print(hmCamToCal)
        print()
        print(np.dot(hmCalToCam, hmCamToCal))

        # HMTCP2CAL = UtilHM.convertXYZABCtoHMDeg([-0.058, 0.058, 0, 0, 0, 0])
        # HMCal2TCP = UtilHM.inverseHM(HMTCP2CAL)
        # print(np.dot(HMTCP2CAL, HMCal2TCP))
        # print(HMCal2TCP)
        # print()

        # HMR = UtilHM.convertXYZABCtoHMDeg([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # print(HMR)
        # print()
        

        # print("Result: ")
        # print(np.dot(np.dot(hmCamToCal, HMCal2TCP), HMR))
    elif pressedKey == ord('c'):
        print("rvec: ")
        print(rvec)
        print("tvec: ")
        print(tvec)
        #print(UtilHM.convertXYZABCtoHMRad([tvec[0][0,0], tvec[0][0,1], tvec[0][0,2], rvec[0][0,0], rvec[0][0,1], rvec[0][0,2]]))
        
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        print(rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        print(hmCalToCam)

        HMCalibbaseToCam.append(UtilHM.inverseHM(hmCalToCam))

        print()
        print(np.dot(hmCalToCam, UtilHM.inverseHM(hmCalToCam)))
        print()

        HMR = UtilHM.convertXYZABCtoHMDeg([0, 0, 0, 180, 0, 180])
        print(HMR)
        HMTCPtoRobot.append(UtilHM.inverseHM(HMR))


    

    elif pressedKey == ord('v'):
        print("rvec: ")
        print(rvec)
        print("tvec: ")
        print(tvec)
        #print(UtilHM.convertXYZABCtoHMRad([tvec[0][0,0], tvec[0][0,1], tvec[0][0,2], rvec[0][0,0], rvec[0][0,1], rvec[0][0,2]]))
        
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        print(rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        print(hmCalToCam)

        HMCalibbaseToCam.append(UtilHM.inverseHM(hmCalToCam))

        print()
        print(np.dot(hmCalToCam, UtilHM.inverseHM(hmCalToCam)))
        print()

        HMR = UtilHM.convertXYZABCtoHMDeg([-0.15, 0, 0, 180, 0, 180])
        print(HMR)
        HMTCPtoRobot.append(UtilHM.inverseHM(HMR))
    elif pressedKey == ord('b'):
        print("rvec: ")
        print(rvec)
        print("tvec: ")
        print(tvec)
        #print(UtilHM.convertXYZABCtoHMRad([tvec[0][0,0], tvec[0][0,1], tvec[0][0,2], rvec[0][0,0], rvec[0][0,1], rvec[0][0,2]]))
        
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        print(rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        print(hmCalToCam)

        HMCalibbaseToCam.append(UtilHM.inverseHM(hmCalToCam))

        print()
        print(np.dot(hmCalToCam, UtilHM.inverseHM(hmCalToCam)))
        print()

        HMR = UtilHM.convertXYZABCtoHMDeg([-0.15, -0.15, 0, 180, 0, 180])
        print(HMR)
        HMTCPtoRobot.append(UtilHM.inverseHM(HMR))
    elif pressedKey == ord('n'):
        print("rvec: ")
        print(rvec)
        print("tvec: ")
        print(tvec)
        #print(UtilHM.convertXYZABCtoHMRad([tvec[0][0,0], tvec[0][0,1], tvec[0][0,2], rvec[0][0,0], rvec[0][0,1], rvec[0][0,2]]))
        
        rotation_matrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(rvec[0], rotation_matrix)
        print(rotation_matrix)
        hmCalToCam = UtilHM.makeHM(rotation_matrix, tvec[0])
        print(hmCalToCam)

        HMCalibbaseToCam.append(UtilHM.inverseHM(hmCalToCam))

        print()
        print(np.dot(hmCalToCam, UtilHM.inverseHM(hmCalToCam)))
        print()

        HMR = UtilHM.convertXYZABCtoHMDeg([0, -0.15, 0, 180, 0, 180])
        print(HMR)
        HMTCPtoRobot.append(UtilHM.inverseHM(HMR))                        
       

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



