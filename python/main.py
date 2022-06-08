import os 
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from cell import Cell 
from scipy.stats import wasserstein_distance
import scipy
from scipy import stats as sts
img1 = cv.imread('./images/000001.png', cv.IMREAD_COLOR)
img2 = cv.imread('./images/000002.png', cv.IMREAD_COLOR)

# Initiate ORB detector
orb1 = cv.ORB_create(800)
orb2 = cv.ORB_create(800)
# find the keypoints with ORB
kp1 = orb1.detect(img1,None)
kp2 = orb2.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb1.compute(img1, kp1)
kp2, des2 = orb2.compute(img2, kp2)
# draw only keypoints location,not size and orientation
img1 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2 = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
#plt.imshow(img1), plt.show()
#plt.imshow(img2), plt.show()
bf = cv.BFMatcher(cv.NORM_HAMMING)
matches = bf.match(des1, des2)
min_distance = matches[0].distance
max_distance = matches[0].distance
for x in matches:
    if x.distance < min_distance:
        min_distance = x.distance
    if x.distance > max_distance:
        max_distance = x.distance
good_match = []

for x in matches:
    if x.distance <= max(2 * min_distance, 30):
        good_match.append(x)
matches1 = good_match
# outimage = cv.drawMatches(img1, kp1, img2, kp2, matches, outImg=None)
# cv.imshow("Match Result", outimage)
# cv.waitKey(0)

##---------------------------------------------------------------------------------

rows,cols,dpt=img1.shape
print(cols)
print(rows)
#1241, 376
Map1=np.array([])
Map2=np.array([])
col_m = 60
row_n = 30
Map1.resize(row_n, col_m)
Map2.resize(row_n, col_m)
# for i in range(0, col_m):
#     for j in range(0, row_n):
#         cell = Cell()
#         Map1.append(cell)
# for i in range(0, col_m):
#     for j in range(0, row_n):
#         cell = Cell()
#         Map2.append(cell)
# print(Map1)
# print(Map2)
#-----------------------------------------------
#kp[0].pt
print(len(kp1))
for k1 in range(0, len(kp1)):
    x, y = kp1[k1].pt
    id_x = int((y*row_n)//rows)
    id_y = int((x*col_m)//cols)
    Map1[id_x][id_y]=Map1[id_x][id_y]+1
    #print(id_x)
for k2 in range(0, len(kp2)):
    x, y = kp2[k2].pt
    id_x = int((y*row_n)//rows)
    id_y = int((x*col_m)//cols)
    Map2[id_x][id_y]=Map2[id_x][id_y]+1
np.savetxt('a.txt',Map1)
print("Map1 data!!")
print(Map1)
print("Map2 data!!")
print(Map2)
print("finish init")
#----------------------------------------------processing

map1_distr=[]
map2_distr=[]
r=8
# i_x=0
# j_y=0

# for p in range(i_x, i_x+r):
#     for l in range(j_y, j_y+r):
#         map1_distr.append(Map1[p][l])

# i_x=1
# j_y=0
# for p in range(i_x, i_x+r):
#     for l in range(j_y, j_y+r):
#         map2_distr.append(Map2[p][l])
# print(wasserstein_distance(map1_distr, map2_distr))
def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * sts.entropy(p, M) + 0.5 * sts.entropy(q, M)
for m in range(0, len(matches1)):
    map1_distr=[]
    map2_distr=[]
    x1, y1 = kp1[matches1[m].queryIdx].pt
    x2, y2 = kp2[matches1[m].trainIdx].pt
    id1_x = int((y1*row_n)//rows)
    id1_y =int ((x1*col_m)//cols)
    id2_x = int((y2*row_n)//rows)
    id2_y =int ((x2*col_m)//cols)
    for xi in range(0, 2*r):
        for yi in range(0, 2*r):
            if (id1_x-r+xi)<0 or (id1_x-r+xi)>=row_n or (id1_y-r+yi)<0 or (id1_y-r+yi)>col_m:
                map1_distr.append(0)
            else:
                map1_distr.append(Map1[id1_x][id1_y])
            if (id2_x-r+xi)<0 or (id2_x-r+xi)>=row_n or (id2_y-r+yi)<0 or (id2_y-r+yi)>col_m:
                map2_distr.append(0)
            else:
                map2_distr.append(Map2[id2_x][id2_y])
    print(wasserstein_distance(map1_distr, map2_distr))
    #print(scipy.stats.entropy(map2_distr/np.sum(map2_distr),  map1_distr/np.sum(map1_distr)))
    # xt = np.asarray(map1_distr)
    # yt = np.asarray(map2_distr)
    # print(JS_divergence(xt, yt))
#print(wasserstein_distance(map1_distr, map2_distr))

