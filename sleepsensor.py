import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
alarm = mixer.Sound('Buzzer.wav')

face = cv2.CascadeClassifier('OpenCV\haarcascade_frontalface_alt.xml')
righteye = cv2.CascadeClassifier('OpenCV\haarcascade_righteye_2splits.xml')
lefteye = cv2.CascadeClassifier('OpenCV\haarcascade_lefteye_2splits.xml')




label=['Close','Open']

model = load_model('CNN/classifier1.h5')
#model = load_model('models/yawn_detection1.h5')
#path = os.getcwd()
camera = cv2.VideoCapture(0) 
style = cv2.FONT_HERSHEY_COMPLEX_SMALL


red=2
pr=[99]
pl=[99]
point=0
count=0


while(True):
    ret, frame = camera.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    f = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) # giving box for face
    le = lefteye.detectMultiScale(gray)
    re =  righteye.detectMultiScale(gray) #giving box for right eye

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED ) #set height and width of frame

    for (x,y,w,h) in f:
        #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 ) #iterating for face in frame using box
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 2 )
        
    for (x,y,w,h) in re:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        re1=frame[y:y+h,x:x+w] #iterating/checking/scanning for right eye in frame(from above  loop) using right eye box 
        
        count=count+1
        re1 = cv2.cvtColor(re1,cv2.COLOR_BGR2GRAY) # after finding right eye convert right eye image to gray image
        re1 = cv2.resize(re1,(24,24))              #applying filter to recognise right eye using 24*24 filter
        #re1 = cv2.resize(re1,(256,256)) 
        re1= re1/255
        re1=  re1.reshape(24,24,-1)
        #re1=  re1.reshape(256,256,-1)
        re1 = np.expand_dims(re1,axis=0)
        pr = np.argmax(model.predict(re1), axis=-1) #classifying if close or open
        if(pr[0]==1):
            label='Open' 
        if(pr[0]==0):
            label='Closed'
        #else:
             #='Face not detected'     
        break

    for (x,y,w,h) in le:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        le1=frame[y:y+h,x:x+w]
        
        count=count+1
        le1 = cv2.cvtColor(le1,cv2.COLOR_BGR2GRAY)  
        le1 = cv2.resize(le1,(24,24))
        #le1 = cv2.resize(le1,(256,256))
        le1= le1/255
        le1=le1.reshape(24,24,-1)
        #le1=le1.reshape(256,256,-1)
        le1 = np.expand_dims(le1,axis=0)
        pl = np.argmax(model.predict(le1), axis=-1)
        
        if(pl[0]==1):
            label='Open'   
        if(pl[0]==0):
            label='Closed'
        #else:
             #='Face not detected'  
        break
        

    if(pr[0]==0 and pl[0]==0):
        point=point+1
        cv2.putText(frame,"Closed",(10,height-20), style, 1,(255,255,255),1,cv2.LINE_AA)
        
    #if(pr[0]==1 or pl[0]==1):
    else:
        point=point-1
        cv2.putText(frame,"Open",(10,height-20), style, 1,(255,255,255),1,cv2.LINE_AA)
    #else:
        #cv2.putText(frame,"Face not detected",(10,height-20), style, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(point<0):
        point=0   
    cv2.putText(frame,'Score:'+str(point),(100,height-20), style, 1,(255,255,255),1,cv2.LINE_AA)
    if(point>15):
        #person is feeling sleepy so we beep the alarm
        #cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            alarm.play()
            
        except:  # isplaying = False
            pass
        if(red<16):
            red= red+2
        else:   
            red=red-2
            if(red<2):
                red=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),red) 
    cv2.imshow('DriveCam',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
