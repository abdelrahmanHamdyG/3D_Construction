#Zeyad 10/24/2024

import zhang_algorithm as za
import numpy as np
import cv2


def getRelRT(RT1,RT2):
    
    R1=RT1[0:3,0:3]
    R2=RT2[0:3,0:3]
    T1=RT1[0:3,3]
    T2=RT2[0:3,3]

    relR=np.dot(R2,R1.T)
    relT=T2-np.dot(relR,T1)
    
    return np.hstack((relR,relT.reshape(3,1)))

def makeSkewSymmetric(m):
    return np.array([[0,-m[2],m[1]],
                     [m[2],0,-m[0]],
                     [-m[1],m[0],0]])

def getEssentialMatrix(RT1,RT2):
    relRT=getRelRT(RT1,RT2)
    relR=relRT[0:3,0:3]
    relT=relRT[0:3,3]

    return np.dot(makeSkewSymmetric(relT),relR)

def getFundamentalMatrix(K,RT1,RT2):
    E=getEssentialMatrix(RT1,RT2)
    return np.dot(np.dot(np.linalg.inv(K.T),E),np.linalg.inv(K))

def getEpipolarLine(F,points):
    if points.shape[0]==2:
        points=np.vstack((points,np.ones(points.shape[1])))
    epLines=np.dot(F,points)
    epLines=epLines/np.sqrt(epLines[0]**2+epLines[1]**2)
    return epLines
    
img1=cv2.imread('calibration/8.jpg')
img2=cv2.imread('calibration/9.jpg')

H1=za.compute_H(img1,1)
H2=za.compute_H(img2,2)

K=za.K

RT1=za.compute_extrinsic_for_each_image(H1,K)
RT2=za.compute_extrinsic_for_each_image(H2,K)

F=getFundamentalMatrix(K,RT1,RT2)

x_points = np.random.randint(100, img1.shape[1] - 100, (1, 5))  
y_points = np.random.randint(100, img1.shape[0] - 100, (1, 5))  
points1 = np.vstack((x_points, y_points))
img2EpLines=getEpipolarLine(F,points1)

for i in range(points1.shape[1]):
    cv2.circle(img1, (points1[0,i], points1[1,i]), 5, (0, 255, 0), -1)
    cv2.line(img2, (0, int(-img2EpLines[2,i]/img2EpLines[1,i])),
            (img2.shape[1], int((-img2EpLines[2,i]-img2EpLines[0,i]*img2.shape[1])/img2EpLines[1,i])), (0, 0, 255), 2)

#show the two images beside each other
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the images
cv2.imwrite('img1.jpg',img1)
cv2.imwrite('img2.jpg',img2)



