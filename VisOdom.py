import cv2
import numpy as np
import os
debug=False

loadPath="KITTI_sequence_1/"
calibFile="calib.txt"
poseFile="poses.txt"
imagePath="image_l/"


def InitFeatureExtractor():
    orb=cv2.ORB_create()
    return orb
    

def loadCalibMatrix():
    x=np.loadtxt(loadPath+calibFile)
    output=[]
    for matrix in x:
        matrix=np.reshape(matrix,(3,4))
        output.append(matrix)
        if debug:
            print(matrix)
    return output
        
def loadPoses():
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
    img=cv2.imread(path)
    cv2.imshow("Window",img)
        
    

def main():
    loadCalibMatrix()
    loadPoses()
    featureDet=InitFeatureExtractor()
    imageNames=loadImages()
    for imageName in imageNames:
        readImage(loadPath+imagePath+imageName)
        cv2.waitKey(100)
        
    
    

if __name__=="__main__":
    main()


