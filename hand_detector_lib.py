'''
Credits: Murtaza's workshop
'''

import mediapipe as mp
import cv2
import time

class hand_detector():
    def __init__(self,mode=False,maxHands=1,detectionConf=0.5,trackConf=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConf = detectionConf
        self.trackConf=trackConf
        
        self.mpHands = mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        
        
    def find_hands(self,img,draw=True):
        '''
        Finds the landmark of the hand(s)
        Parameters:
                img(numpy array): Image to detect hands on
                draw(bool-optional): Draw the hand landmarks

        Returns:
                Image with hand landmarks drawn
        '''
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    
    def find_position(self,img,handNo=0,draw=True):
        '''
        Finds the position of the special points in hand
        Parameters:
                img(numpy array): Image to detect hands on
                handNo(int-optional): Which hand?
                draw(bool-optional): Draw the special points on hand
                

        Returns:
                List with coordinates of special points of the hand
        '''
        self.lmList = []
        if self.results.multi_hand_landmarks :
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                
                if draw:cv2.circle(img,(cx,cy),10,(255,0,255),-1)
        
        return self.lmList

    def fingers_up(self):
        '''
        Finds which fingers are up and straight
    
        Returns:
                List of five element corresponding to five fingers of hand.
                1-Lifted up, 0-Not lifted up
        '''
        fingers=[]
        if self.lmList[self.tipIds[0]][1]< self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2]<self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
