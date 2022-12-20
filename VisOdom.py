import cv2
import numpy as np
import os
debug=False

loadPath="KITTI_sequence_1/"
calibFile="calib.txt"
poseFile="poses.txt"
imagePath="image_l/"
requiredSize=(618,185)


def InitFeatureExtractor():
    orb=cv2.ORB_create()
    if debug:
        print(type(orb))
    
    return orb
    

def loadCalibMatrix():
    # Loads K and P matrices
    # K = intrinsic params
    # P Calibration Matrix
    x=np.loadtxt(loadPath+calibFile)
    output=[]
    for matrix in x:
        matrix=np.reshape(matrix,(3,4))
        output.append(matrix)
        if debug:
            print(matrix)
    return output
        
def loadPoses():
    # Loads pose of camera
    # Need to stack [0 0 0 1] as told by KITTI since matrix is actually supposed to be 4x4 but represented as 3x4 in file

    y=np.loadtxt(loadPath+poseFile)
    output=[]
    for matrix in y:
        #print(matrix)
        matrix=np.reshape(matrix,(3,4))
        #matrix=np.append(matrix,[0,0,0,1])
        matrix=np.vstack ( (matrix,np.array([0,0,0,1]))  )
        if debug:
            print(matrix)
    return output
    
    
def loadImages():
    list1=os.listdir(loadPath+imagePath)
    list1.sort()
    if debug:
        print(list1)
    return list1

def readImage(path):
    # Read image as grayscale
    img=cv2.imread(path,0)
    img=cv2.resize(img, dsize=requiredSize )
    if debug:
        print(img.shape)
    return img


def showImage(img):
    cv2.imshow("Window",img)
    cv2.waitKey(1000)


def detectFeatures(featureDet:cv2.ORB,img):
    keyPoints,descriptors=featureDet.detectAndCompute(img,None)
    return (keyPoints,descriptors)
    pass



def compareImages(featureDet,prevImage,currentImage):
    # Compare images using feature matching
    # Keep best 20 matches only
    # Return the coords of matched features
    
    output=None
    prev_img_featurePoints=[]
    curr_img_featurePoints=[]
    
    
    prevKeyPoints,prevDescriptors=detectFeatures(featureDet,prevImage)
    currentKeyPoints,currentDescriptors=detectFeatures(featureDet,currentImage)
    matcher=cv2.BFMatcher()
    matches=matcher.match(prevDescriptors,currentDescriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    matches=matches[:20]
    output=cv2.drawMatches(prevImage,prevKeyPoints,currentImage,currentKeyPoints,matches,output)
    showImage(output)
    for match in matches:
        prev_img_idx = match.queryIdx
        curr_img_idx = match.trainIdx
        
        img1_Coords=prevKeyPoints[prev_img_idx].pt
        img2_Coords=currentKeyPoints[curr_img_idx].pt
        prev_img_featurePoints.append(img1_Coords)
        curr_img_featurePoints.append(img2_Coords)
        
    prev_img_featurePoints=np.asarray(prev_img_featurePoints,dtype=np.float32)
    curr_img_featurePoints=np.asarray(curr_img_featurePoints,dtype=np.float32)
    
    if debug:
        print(prev_img_featurePoints.shape)
    
    return prev_img_featurePoints,curr_img_featurePoints
    

def findCameraTransform(prevImg_FeatureCoords,currImg_FeatureCoords,K):
    cv2.findEssentialMat(prevImg_FeatureCoords,currImg_FeatureCoords,K)
        
    

def main():
    K,P=loadCalibMatrix()
    featureDet=InitFeatureExtractor()
    imageNames=loadImages()
    prevImage=readImage(loadPath+imagePath+imageNames[0])
    
    for i in range(1,len(imageNames)):
        img=readImage(loadPath+imagePath+imageNames[i])
        prevImg_FeatureCoords,currImg_FeatureCoords= compareImages(featureDet,prevImage,img)
        findCameraTransform(prevImg_FeatureCoords,currImg_FeatureCoords,K)
        prevImage=img.copy()
        
        
        
    
    

if __name__=="__main__":
    main()


