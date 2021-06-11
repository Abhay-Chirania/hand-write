import cv2     
import numpy as np
import os
from datetime import datetime
from hand_detector_lib import hand_detector
sw,sh = 800,600

def main():
    header_images=[]
    for i in os.listdir('./assets'):
        _img = cv2.imread(os.path.join("assets",i))
        _img = cv2.resize(_img,(sw,int(sw*(_img.shape[0]/_img.shape[1]))))
        header_images.append( _img)
    hh,hw,_ = header_images[0].shape
    header = 0
    
    cam = cv2.VideoCapture(0)
    detector = hand_detector()
    
    drawing_canvas = np.zeros((sh,sw,3),np.uint8)
    color = (0,0,255)
    thickness = 5
    eraser_thickness = 80
    x_prev,y_prev=0,0
    record=False
    out=None
    while True:
        status,frame = cam.read()
        frame=cv2.resize(frame,(sw,sh))
        frame = cv2.flip(frame,1) 
        frame = detector.find_hands(frame,draw=False)
        lmList = detector.find_position(frame,draw=False)
        
        if lmList:
            x1,y1 = lmList[8][1:]
            x2,y2 = lmList[12][1:]
        
            fingers = detector.fingers_up()
            
            if fingers[1] and fingers[2]==False:
                cv2.circle(frame,(x1,y1),8,color,-1)
                if x_prev == 0 and y_prev==0:   
                    x_prev,y_prev = x1,y1
                    
                if color==(0,0,0):
                    et_h = eraser_thickness//2
                    cv2.rectangle(frame,(x1-et_h ,y1- et_h ),(x1+ et_h ,y1+ et_h ),(0,0,0),-1)
                    cv2.line(drawing_canvas,(x_prev,y_prev),(x1,y1),color,eraser_thickness)
                else:
                    cv2.line(drawing_canvas,(x_prev,y_prev),(x1,y1),color,thickness)

                x_prev,y_prev = x1,y1
                    
                
            elif fingers[1] and fingers[2]:
                x_prev,y_prev = 0,0
                #cv2.rectangle(frame,(x1,y1-25),(x2,y2+25),(0,0,0),-1)
                
                if y1<hh:
                    if 120<=x1<=240:
                        color = (0,0,255)
                        header = 0 
                    
                    elif 260<=x1<=380:
                        color = (255,0,0)
                        header = 1
                        
                    elif 410<=x1<=530:
                        color = (0,255,0)
                        header = 2
                        
                    elif 540<=x1<=660:
                        color = (156,0,210)
                        header = 3
                    elif x1>=670:
                        color = (0,0,0)
                        header = 4
                    
                    elif x1<=100:
                        record = True
                        
        
        if not record:
            out = None
        
        elif record and not out:
            type = cv2.VideoWriter_fourcc(*'XVID' )
            name = str(datetime.now()).split()
            name= f"{name[0]}_{name[1][:name[1].index('.')]}.mp4"
            name=name.replace("-","_")
            name=name.replace(":","_")
            out = cv2.VideoWriter(os.path.join("recordings",name),type , 30 , (sw,sh) )
        
                    
                    
        img_gray = cv2.cvtColor(drawing_canvas,cv2.COLOR_BGR2GRAY)
        _,img_inv = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY_INV)

        img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame,img_inv)
        frame = cv2.bitwise_or(frame,drawing_canvas)
        
        frame[:hh,:hw]=header_images[header]

        if record and out:
            cv2.rectangle(frame,(0,0),(100,hh),(41,156,0),3)
            out.write(frame)
        
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1)==113:
            cam.release()
            if out:out.release()
            break
    
    cv2.destroyAllWindows()
    
main()
