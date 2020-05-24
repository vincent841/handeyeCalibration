
import numpy as np
import cv2
import UtilHM

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

# Ref) https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
def recover_homogenous_affine_transformation(p, p_prime):
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

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))

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
            

