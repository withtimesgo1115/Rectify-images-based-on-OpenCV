##Created by Caining Liu
##11:19PM March 3, 2020

import cv2
import glob
import os
import numpy as np
from matplotlib import pyplot as plt 

nb_vertical = 9
nb_horizontal = 6
#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
#prepare object points, like(0 0 0),(1 0 0),(2 0 0) ,etc
objp = np.zeros((nb_horizontal*nb_vertical,3),np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

#Arrays to store object points and image points from all the images.
objpoints = []
imgpoints = []

#prepare object points, like(0 0 0),(1 0 0),(2 0 0) ,etc
objp_r = np.zeros((nb_horizontal*nb_vertical,3),np.float32)
objp_r[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

#Arrays to store object points and image points from all the images.
objpoints_r = []
imgpoints_r = []

def calibrate_each_camera(path,objpoints,imgpoints):
    images = sorted(glob.glob(path))
    print(len(images))
    assert images
    gray_list = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_list.append(gray)
    
    #Implement findChessboardCorners
    ret, corners = cv2.findChessboardCorners(gray,(nb_vertical,nb_horizontal),None)
    #If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    return mtx,dist,rvecs,tvecs

def undistort(images,mtx,dist):
    raw_images = []
    undst_set = []
    for fname in images:
        img = cv2.imread(fname)
        h,w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        undst = cv2.undistort(img,mtx,dist,None,newcameramtx)
        x,y,w,h = roi
        undst = undst[y:y+h, x:x+w]
        undst_set.append(undst)
    return undst_set


if __name__ == "__main__":
    path1 = 'left/*.png'
    mtx_l,dist_l,rvecs_l,tvecs_l = calibrate_each_camera(path1,objpoints,imgpoints)
    path2 = 'right/*.png'
    mtx_r,dist_r,rvecs_r,tvecs_r = calibrate_each_camera(path2,objpoints_r,imgpoints_r)
    images_l = sorted(glob.glob('left/*.png'))
    print(len(images_l))
    assert images_l
    images_r = sorted(glob.glob('right/*.png'))
    print(len(images_r))
    assert images_r
    if not os.path.exists('undistort_left'):
        os.makedirs('undistort_left')
    if not os.path.exists('undistort_right'):
        os.makedirs('undistort_right')
    undistort_img_left = undistort(images_l,mtx_l,dist_l)
    undistort_img_right = undistort(images_r,mtx_r,dist_r)
    for i in range(len(undistort_img_left)):
        cv2.imwrite('undistort_left/'+str(i)+".png",undistort_img_left[i])
    for j in range(len(undistort_img_right)):
        cv2.imwrite('undistort_right/'+str(j)+".png",undistort_img_right[j])

    #Get the relative pose between the two cameras(R T)
    stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,100,1e-5)
    stereocalibration_retval,newcameramtx_left,dist_left,newcameramtx_right,dist_right,R,T,E,F = cv2.stereoCalibrate(objpoints,imgpoints,imgpoints_r,mtx_l,dist_l,mtx_r,dist_r,undistort_img_left[0].shape[0:2][::-1],criteria=stereocalibration_criteria,flags=cv2.CALIB_FIX_INTRINSIC)

    #calculate rotation matrix for each camera that make l/r camera image planes equal
    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=newcameramtx_left,cameraMatrix2=newcameramtx_right,distCoeffs1=dist_left,distCoeffs2=dist_right,imageSize=undistort_img_left[0].shape[0:2][::-1],R=R,T=T,Q=None,flags=cv2.CALIB_ZERO_DISPARITY,newImageSize=(0,0))[0:4]

    #compute stereo rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(newcameramtx_left,dist_left,R1,P1,undistort_img_left[0].shape[0:2][::-1],cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(newcameramtx_right,dist_right,R2,P2,undistort_img_left[0].shape[0:2][::-1],cv2.CV_32FC1)

    #output the rectified images and save them
    if not os.path.exists('rectified_left'):
        os.makedirs('rectified_left')
    if not os.path.exists('rectified_right'):
        os.makedirs('rectified_right')
    for i in range(len(undistort_img_left)):
        rec = cv2.remap(undistort_img_left[i],map1x,map1y,cv2.INTER_LINEAR)
        cv2.imwrite('rectified_left/'+str(i)+".png",rec)
    for j in range(len(undistort_img_right)):
        rec = cv2.remap(undistort_img_right[j],map2x,map2y,cv2.INTER_LINEAR)
        cv2.imwrite('rectified_right/'+str(j)+".png",rec)