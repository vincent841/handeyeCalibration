
import cv2.aruco as aruco
import numpy as np
import cv2
import glob

def saveArucoCalibData(mtx, dist, rvecs, tvecs):
    calibFile = cv2.FileStorage("CameraCalibResultAruco.xml", cv2.FILE_STORAGE_WRITE)
    calibFile.write("cameraMatrix", mtx)
    calibFile.write("distCoeff", dist)
    
    #rotation_matrix = np.zeros(shape=(3,3))
    # iter = 0
    # for rvec in rvecs:
    #     #cv2.Rodrigues(rvec, rotation_matrix)
    #     #calibFile.write("rvec" + str(iter), rotation_matrix)
    #     calibFile.write("rvec" + str(iter), rvec)
    #     iter+=1
    # iter = 0
    # for tvec in tvecs:
    #     calibFile.write("tvec" + str(iter), tvec)
    #     iter+=1
    calibFile.release()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.75  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

arucoParams = aruco.DetectorParameters_create()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# input directory name 
dirInput = input('Directory Name: ')
#dirInput = 'calibimgs'

# get image file names
images = glob.glob('./' + dirInput + '/*.jpg')

counter, corners_list, id_list = [], [], []
first = True

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the aruco board corners
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)
    if first == True:
            corners_list = corners
            id_list = ids
            first = False
    else:
        corners_list = np.vstack((corners_list, corners))
        id_list = np.vstack((id_list,ids))
    
    counter.append(len(ids))

print('Found {} unique markers'.format(np.unique(ids)))

counter = np.array(counter)
print ("Calibrating camera .... Please wait...")
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, gray.shape, None, None )

# save calibration data to the specific xml file
saveArucoCalibData(mtx, dist, rvecs, tvecs)

cv2.destroyAllWindows()
