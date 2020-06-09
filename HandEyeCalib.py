
import numpy as np
import cv2
from UtilSet import *
import math

# opencv parameters
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)          # termination criteria

class HandEyeCalibration:

    def __init__(self):
        # input variables for handeye calibration
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        self.R_cam2gripper = []
        self.t_cam2gripper = []

    #deperecated
    # Ref) https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
    #      https://math.stackexchange.com/questions/222113/given-3-points-of-a-rigid-body-in-space-how-do-i-find-the-corresponding-orienta/222170#222170
    def calculateTransformMatrixUsing3Points(self, p, p_prime):
        # construct intermediate matrix
        Q       = p[1:]       - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
                np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        return np.column_stack((np.row_stack((R, t)),
                                (0, 0, 0, 1)))

    # deprecated...
    def calculateTransformMatrix(self, srcPoints, dstPoints):
        assert(len(srcPoints) == len(dstPoints))

        p = np.ones([len(srcPoints), 4])
        p_prime = np.ones([len(dstPoints), 4])
        for idx in range(len(srcPoints)):
            p[idx][0] = srcPoints[idx][0]
            p[idx][1] = srcPoints[idx][1]
            p[idx][2] = srcPoints[idx][2]

            p_prime[idx][0] = dstPoints[idx][0]
            p_prime[idx][1] = dstPoints[idx][1]
            p_prime[idx][2] = dstPoints[idx][2]

        trMatrix = cv2.solve(p, p_prime, flags=cv2.DECOMP_SVD)
        return trMatrix

    # handeye calibration test function
    def calibrateHandEyeTest(self, HMBase2TCPs, HMTarget2Cams):
        #assert (HMBase2TCPs.len() == HMTarget2Cams.len())
        for hmmat in HMBase2TCPs:
            rotataion = hmmat[0:3, 0:3]
            self.R_gripper2base.append(rotataion)
            translation = hmmat[0:3, 3]
            self.t_gripper2base.append(translation)

        for hmmat in HMTarget2Cams:
            rotataion = hmmat[0:3, 0:3]
            self.R_target2cam.append(rotataion)
            translation = hmmat[0:3, 3]
            self.t_target2cam.append(translation) 
        
        methodHE = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]

        for mth in methodHE:
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_target2cam, self.t_target2cam, None, None, mth)
            cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_target2cam, self.t_target2cam, None, None, mth)
            # output results
            print("--------------------------------------")
            print("Method %d" % mth)
            print(self.R_cam2gripper)
            print(self.t_cam2gripper)
            print("--------------------------------------")

    def captureHandEyeInputs(self, robotXYZABC, camRVec, camTVec):
        # prepare Gripper2Base inputs
        hmRobot = HMUtil.convertXYZABCtoHMDeg(robotXYZABC)
        self.R_gripper2base.append(hmRobot[0:3, 0:3])
        self.t_gripper2base.append(hmRobot[0:3, 3])

        # prepare Target2Cam inputs
        camRMatrix = np.zeros(shape=(3,3))
        cv2.Rodrigues(camRVec, camRMatrix)
        hmCam = HMUtil.makeHM(camRMatrix, camTVec)
        hmCam = HMUtil.inverseHM(hmCam)
        self.R_target2cam.append(hmCam[0:3, 0:3])
        self.t_target2cam.append(hmCam[0:3, 3])

    def getHandEyeResultMatrixUsingOpenCV(self):
        methodHE = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]
        for mth in methodHE:
            self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.R_target2cam, self.t_target2cam, None, None, mth)
            # output results
            print("--------------------------------------")
            print("Method %d" % mth)
            print(self.R_cam2gripper)
            print(self.t_cam2gripper)
            print("--------------------------------------")
            print("Distance: %f" % math.sqrt(math.pow(self.t_cam2gripper[0], 2.0)+math.pow(self.t_cam2gripper[1], 2.0)+math.pow(self.t_cam2gripper[2], 2.0)))
            print("--------------------------------------")

            if(mth == cv2.CALIB_HAND_EYE_HORAUD):
                for idx in range(len(self.R_gripper2base)):
                    print("######")
                    hmT2G = HMUtil.makeHM(self.R_cam2gripper, self.t_cam2gripper.T)
                    hmG2B = HMUtil.makeHM(self.R_gripper2base[idx], self.t_gripper2base[idx].reshape(1,3))
                    hmC2T = HMUtil.makeHM(self.R_target2cam[idx], self.t_target2cam[idx].reshape(1,3))
                    #hmC2T = HMUtil.inverseHM(hmC2T) # test
                    hmTransform = hmT2G
                    #hmTransform = np.dot(hmG2B, hmT2G)
                    #hmTransform = np.dot(hmTransform, hmC2T)
                    print(hmTransform)
                    hmTransform2 = np.dot(hmG2B, hmT2G)
                    hmTransform2 = np.dot(hmTransform2, hmC2T)            
                    print("Checkpoint #1: ")
                    print(hmTransform2)
                    print("Checkpoint #2: ")
                    print(HMUtil.inverseHM(hmTransform2))

        return hmTransform2


###############################################################################
# Test Codes using sample data in yml format
###############################################################################
if __name__ == '__main__':

    YMLHMBase2TCPs = []
    YMLHMTarget2Cams = []

    cam3DTestPoints = []
    robot3DTestPoints = []

    handEyeInput = cv2.FileStorage("./handEyeSample.yml", cv2.FILE_STORAGE_READ)
    fileNode = handEyeInput.root()  

    handeye = HandEyeCalibration()

    for key in fileNode.keys():
        ymlnode = handEyeInput.getNode(key)
        ymlmtx = ymlnode.mat()

        if key.find("target2cam") >= 0:
            YMLHMTarget2Cams.append(ymlmtx)

        if key.find("gripper2base") >= 0:
            ymlmtx = HMUtil.inverseHM(ymlmtx)
            YMLHMBase2TCPs.append(ymlmtx)

    handeye.calibrateHandEyeTest(YMLHMBase2TCPs, YMLHMTarget2Cams)

    # # calculateHM Test
    cam3DTestPoints.append([-0.10259, 0.07283, 0.40900])
    cam3DTestPoints.append([0.14604, 0.00431, 0.42700])
    cam3DTestPoints.append([-0.00145, 0.10705, 0.31100])
    cam3DTestPoints.append([-0.10259, 0.07283, 0.40900])
    cam3DTestPoints.append([0.14604, 0.00431, 0.42700])
    cam3DTestPoints.append([-0.00145, 0.10705, 0.31100])

    robot3DTestPoints.append([-0.18101, -0.52507, 0.01393])
    robot3DTestPoints.append([0.06137, -0.68306, 0.01546])
    robot3DTestPoints.append([-0.18807, -0.66342, 0.01510])
    robot3DTestPoints.append([-0.18101, -0.52507, 0.01393])
    robot3DTestPoints.append([0.06137, -0.68306, 0.01546])
    robot3DTestPoints.append([-0.18807, -0.66342, 0.01510])

    result = handeye.calculateTransformMatrixUsing3Points(np.array(((-0.10259, 0.07283, 0.40900),(0.14604, 0.00431, 0.42700), (-0.00145, 0.10705, 0.31100))), 
    np.array(((-0.18101, -0.52507, 0.01393),(0.06137, -0.68306, 0.01546), (-0.18807, -0.66342, 0.01510)))
    )
    print(result)
    print(result.shape)

    camC = np.array( ((cam3DTestPoints[0]), (cam3DTestPoints[1]), (cam3DTestPoints[2])) )
    print(camC.shape)
    robotC = np.array( ((robot3DTestPoints[0]), (robot3DTestPoints[1]), (robot3DTestPoints[2])) )
    result = handeye.calculateTransformMatrixUsing3Points(camC, robotC)
    print(result)

    result = handeye.calculateTransformMatrix(cam3DTestPoints, robot3DTestPoints)
    print(result)

    print(np.dot(np.array([-0.10259, 0.07283, 0.40900, 1]).reshape(1,4), result[1]))


