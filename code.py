pipimport cv2
import mediapipe as md
import math as m
import numpy as n


def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
        (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    print(degree)
    return degree

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose=md.solutions.pose

count = 0
position = None

cap = cv2.VideoCapture(0)
print(calculate_angle(12,14,16))


with md_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence = 0.7) as pose:
    while cap.isOpened():
        success,image=cap.read()
        if not success:
            print("empty camera")
            break
        
        image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        
        imlist=[]
        
        
        if result.pose_landmarks:
            md_drawing.draw_landmarks(
                image,result.pose_landmarks,md_pose.POSE_CONNECTIONS)
            for id,im in enumerate(result.pose_landmarks.landmark):
                h,w,_= image.shape
                X,Y=int(im.x*w),int(im.y*h)
                imlist.append([id,X,Y])
        
        if len(imlist) !=0:
            if (imlist[12][2] and imlist[11][2] >= imlist[14][2] and imlist[13][2]):
                position='down'
            if (imlist[12][2] and imlist[11][2] <= imlist[14][2] and imlist[13][2] and position=="down"):
                position="up"
                count=count+1
                print(count)
            
        cv2.imshow("Push-up counter",cv2.flip(image,1))
        key=cv2.waitKey(1)
        if key==ord('q'):
            break


cap.release()
