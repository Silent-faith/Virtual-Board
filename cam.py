import cv2
import numpy as np
import os 
import TrackHands as TH

class VideoCamera():
    
    def __init__(self,BrushBoard=[],draw_color =(255,200,100)):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4,720)
        self.xp =0
        self.yp =0
        self.x1 = 0
        self.y1 = 0
        self.x2= 0
        self.y2 = 0
        self.brush_thickness = 10
        self.eraser_thickness = 50
        self.BrushBoard = BrushBoard  
        self.draw_color = draw_color
        self.detector = TH.HandDetector(min_detection_confidence=0.85)
        self.image_canvas = np.zeros((720,1280,3), np.uint8)
        self.default_overlay = BrushBoard[0]


    def get_frame(self, BrushBoard):
        _,frame = self.cap.read()
        frame = cv2.resize(frame, (1280, 720))
        frame[0:125,0:1280] = self.default_overlay
        landmark_list = self.detector.findPosition(frame)

        if(len(landmark_list)!=0):
            self.x1, self.y1 =(landmark_list[8][1:]) #index
            self.x2, self.y2 = landmark_list[12][1:] #middle    
        
            my_fingers = self.detector.fingerStatus()
            if (my_fingers[1]and my_fingers[2]):
                self.xp, self.yp = 0,0
                if (self.y1<125):
                    if(200<self.x1<340):
                        self.default_overlay = BrushBoard[0]
                        frame[0:125,0:1280] = self.default_overlay 
                        self.draw_color = (255,0,0)
                    elif (340<self.x1<500):
                        self.default_overlay = BrushBoard[1]
                        self.draw_color = (47,225,245)
                        frame[0:125,0:1280] = self.default_overlay 
                    elif (500<self.x1<640):
                        self.default_overlay = BrushBoard[2]
                        self.draw_color = (197,47,245)
                        frame[0:125,0:1280] = self.default_overlay 
                    elif (640<self.x1<780):
                        self.default_overlay = BrushBoard[3]
                        self.draw_color = (53,245,47)
                        frame[0:125,0:1280] = self.default_overlay 
                    elif (1100<self.x1<1280):
                        self.default_overlay = BrushBoard[4]
                        self.draw_color = (0,0,0)
                        frame[0:125,0:1280] = self.default_overlay 

                cv2.putText(frame, 'Color Selector Mode', (900,680), fontFace=cv2.FONT_HERSHEY_COMPLEX, color= (0,255,255), thickness=2, fontScale=1)
                cv2.line(frame, (self.x1,self.y1), (self.x2,self.y2), color=self.draw_color, thickness=3)

            if (my_fingers[1] and not my_fingers[2]):
                        
                cv2.putText(frame, "Writing Mode", (900,680), fontFace= cv2.FONT_HERSHEY_COMPLEX, color= (255,255,0), thickness=2, fontScale=1)
                cv2.circle(frame, (self.x1,self.y1),15, self.draw_color, thickness=-1)

                if self.xp ==0 and self.yp ==0:
                    self.xp =self.x1 
                    self.yp =self.y1
                
                if self.draw_color == (0,0,0):
                    cv2.line(frame, (self.xp,self.yp),(self.x1,self.y1),color= self.draw_color, thickness=self.eraser_thickness)
                    cv2.line(self.image_canvas, (self.xp,self.yp),(self.x1,self.y1),color= self.draw_color, thickness=self.eraser_thickness)

                else:
                    cv2.line(frame, (self.xp,self.yp),(self.x1,self.y1),color= self.draw_color, thickness=self.brush_thickness)
                    cv2.line(self.image_canvas, (self.xp,self.yp),(self.x1,self.y1),color= self.draw_color, thickness=self.brush_thickness)
                
                self.xp , self.yp = self.x1, self.y1
          
        frame[0:125,0:1280] = self.default_overlay 
        img_gray = cv2.cvtColor(self.image_canvas, cv2.COLOR_BGR2GRAY)
        _, imginv= cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
        #print(frame.shape, imginv.shape)
        frame = cv2.bitwise_and(frame, imginv)
        frame =cv2.bitwise_or(frame, self.image_canvas)
        _,jpeg =cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def main():

    BrushBoard=[]
    header_img = "Images"
    header_img_list = os.listdir(header_img)
    for i in header_img_list:
        image = cv2.imread(f'{header_img}/{i}')
        BrushBoard.append(image)

    cam1 = VideoCamera(BrushBoard)

    while True:
        my_frame = cam1.get_frame(BrushBoard)   
        
        
    
if __name__ =="__main__" :
    main()
    