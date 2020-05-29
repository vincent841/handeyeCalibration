
import numpy as np
import cv2
import UtilHM

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# opencv parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)          # termination criteria

# Ref) https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
#      https://math.stackexchange.com/questions/222113/given-3-points-of-a-rigid-body-in-space-how-do-i-find-the-corresponding-orienta/222170#222170
def calculateHM(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q       = p[1:]       - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    print("R: ")
    print(R)

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)
    print("t: ")
    print(t)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))

# def calculateHM(p, p_prime):
#     cv2.solve()

def calibrateHandEye(HMBase2TCPs, HMTarget2Cams, HandInEye=True):
    #assert (HMBase2TCPs.len() == HMTarget2Cams.len())

    for hmmat in HMBase2TCPs:
        if(HandInEye == True):
            hmmat = UtilHM.inverseHM(hmmat)
        rotataion = hmmat[0:3, 0:3]
        R_gripper2base.append(rotataion)
        translation = hmmat[0:3, 3]
        t_gripper2base.append(translation)

    for hmmat in HMTarget2Cams:
        rotataion = hmmat[0:3, 0:3]
        R_target2cam.append(rotataion)
        translation = hmmat[0:3, 3]
        t_target2cam.append(translation) 
    
    methodHE = [cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, cv2.CALIB_HAND_EYE_DANIILIDIS]

    for mth in methodHE:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, None, None, mth)
        cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, None, None, mth)
        # output results
        print("--------------------------------------")
        print("Method %d" % mth)
        print(R_cam2gripper)
        print(t_cam2gripper)
        print("--------------------------------------")


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

    for key in fileNode.keys():
        ymlnode = handEyeInput.getNode(key)
        ymlmtx = ymlnode.mat()

        if key.find("target2cam") >= 0:
            YMLHMTarget2Cams.append(ymlmtx)

        if key.find("gripper2base") >= 0:
            ymlmtx = UtilHM.inverseHM(ymlmtx)
            YMLHMBase2TCPs.append(ymlmtx)
    
    calibrateHandEye(YMLHMBase2TCPs, YMLHMTarget2Cams, False)


    # calculateHM Test
    cam3DTestPoints.append([-0.10259, 0.07283, 0.40900])
    cam3DTestPoints.append([0.14604, 0.00431, 0.42700])
    cam3DTestPoints.append([-0.00145, 0.10705, 0.31100])

    robot3DTestPoints.append([-0.18101, -0.52507, 0.01393])
    robot3DTestPoints.append([0.06137, -0.68306, 0.01546])
    robot3DTestPoints.append([-0.18807, -0.66342, 0.01510])

    result = calculateHM(np.array(((-0.10259, 0.07283, 0.40900),(0.14604, 0.00431, 0.42700), (-0.00145, 0.10705, 0.31100))), 
    np.array(((-0.18101, -0.52507, 0.01393),(0.06137, -0.68306, 0.01546), (-0.18807, -0.66342, 0.01510)))
    )
    print(result)
    print(result.shape)

    camC = np.array( ((cam3DTestPoints[0]), (cam3DTestPoints[1]), (cam3DTestPoints[2])) )
    print(camC.shape)
    robotC = np.array( ((robot3DTestPoints[0]), (robot3DTestPoints[1]), (robot3DTestPoints[2])) )
    result = calculateHM(camC, robotC)
    print(result)



