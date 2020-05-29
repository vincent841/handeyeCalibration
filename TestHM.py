import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def getIntrinsicsMat(intrinsics):
    mtx = np.array([[intrinsics.fx,             0, intrinsics.ppx],
                    [            0, intrinsics.fy, intrinsics.ppy],
                    [            0,             0,              1]])
    dist = np.array(intrinsics.coeffs[:4])
    return mtx, dist

def indyPrintTaskPosition():
    task_pos = indy.get_task_pos()
    task_pos_mm = [task_pos[0]*1000.0, task_pos[1]*1000.0, task_pos[2]*1000.0,task_pos[3], task_pos[4], task_pos[5]]
    print ("Task Pos: ")
    print (task_pos_mm) 


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

curr_frame = 0

# creates an align object
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        #     continue
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)
        mtx, dist = getIntrinsicsMat(color_intrin)
        #print(color_intrin)

        # print(depth_intrin.ppx, depth_intrin.ppy)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

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
            if len(corners):
                for k in range(len(corners)):
                    temp_1 = corners[k]
                    temp_1 = temp_1[0]
                    temp_2 = ids[k]
                    temp_2 = temp_2[0]
                    aruco_list[temp_2] = temp_1
            key_list = aruco_list.keys()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for key in key_list:
                dict_entry = aruco_list[key]    
                centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
                centre[:] = [int(x / 4) for x in centre]
                orient_centre = centre + [0.0,5.0]
                centre = tuple(centre)  
                orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
                cv2.circle(color_image,centre,1,(0,0,255), -1)

            # # estimate pose of each marker and return the values
            # # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            # #(rvec-tvec).any() # get rid of that nasty numpy value array error

            # for i in range(0, ids.size):
            #     # draw axis for the aruco markers
            #     aruco.drawAxis(color_image, mtx, dist, rvec[i], tvec[i], 0.1)

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

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        # handle key inputs
        pressedKey = (cv2.waitKey(1) & 0xFF)
        if pressedKey == ord('q'):
            break
        elif pressedKey == ord('p'):
            indyPrintTaskPosition()
        elif pressedKey == ord('c'):
            depth = aligned_depth_frame.get_distance(centre[0], centre[1])
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [centre[0], centre[1]], depth)
            text = "%.5lf, %.5lf, %.5lf\n" % (depth_point[0], depth_point[1], depth_point[2])
            print(text)

        curr_frame += 1
finally:
    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()    