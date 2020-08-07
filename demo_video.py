import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch


from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')

print(f"Using GPU device: {torch.cuda.get_device_name()}")

cap = cv2.VideoCapture('out.mp4')
cap.set(3, 640)
cap.set(4, 480)
frame = 0

np.set_printoptions(precision=3,suppress=True)


def printOutput(candidate,subset):
    #print(subset.shape)
    #print(len(subset))
    
    curPerson = 1
    for i in subset:
        print("Person " + str(curPerson))
        print("Index in candidate")
        print(i[0:17])
        
        personData = []
        count=0
        for itemid in i[0:17]:
            
            num = int(itemid)
            tempcandidate = candidate[num]
            #tempcandidate[3] = count
            #print(tempcandidate)
            personData.append(tempcandidate)
            count+=1
            
        print("Total score : " + str(i[18]))
        print("Total parts : " + str(i[19]))
        print("Individual person Data: ")
        print(personData)
        #print("Candidate Values: ")
        #print(candidate)
        curPerson+=1
        print("\n\n")
        

while True:

    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    printOutput(candidate,subset)
#     print("Subset\n")
#     print(subset)
#     print("Candidate\n")
#     print(candidate)
#     for i in len(candidate):
#         print(candidate[i][0:2])


    #cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    cv2.imwrite('output/out-' + str(frame).zfill(4) + '.jpg', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame+=1
    
cap.release()
cv2.destroyAllWindows()

