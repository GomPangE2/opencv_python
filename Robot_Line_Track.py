import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

def line_track():
       
    while(True):
        ret,img_ori = cap.read()
        
        img_color = cv2.resize(img_ori, dsize=(800, 600), interpolation=cv2.INTER_AREA)
        
        x1 = 0; x2 = 800; y1 = 0; y2 = 200; ya = 200; yb = 400; yA = 400; yB = 600;
        
        roi1 = img_color[y1:y2, x1:x2]
        roi2 = img_color[ya:yb, x1:x2]
        roi3 = img_color[yA:yB, x1:x2]
        
        img_hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        img_hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        img_hsv3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([19,30,30])          # 파랑색 범위
        upper_blue = np.array([29,255,255])

        lower_green = np.array([9, 30, 30])        # 초록색 범위
        upper_green = np.array([19, 255, 255])

        lower_red = np.array([9, 30, 30])        # 빨강색 범위
        upper_red = np.array([19, 255, 255])

    # Threshold the HSV image to get only blue colors
        img_mask11 = cv2.inRange(img_hsv1, lower_blue, upper_blue)     
        img_mask21 = cv2.inRange(img_hsv1, lower_green, upper_green) 
        img_mask31 = cv2.inRange(img_hsv1, lower_red, upper_red)
        img_mask12 = cv2.inRange(img_hsv2, lower_blue, upper_blue)     
        img_mask22 = cv2.inRange(img_hsv2, lower_green, upper_green) 
        img_mask32 = cv2.inRange(img_hsv2, lower_red, upper_red)
        img_mask13 = cv2.inRange(img_hsv3, lower_blue, upper_blue)     
        img_mask23 = cv2.inRange(img_hsv3, lower_green, upper_green) 
        img_mask33 = cv2.inRange(img_hsv3, lower_red, upper_red)
        
        img_mask1 = img_mask11 | img_mask21 | img_mask31
        img_mask2 = img_mask12 | img_mask22 | img_mask32
        img_mask3 = img_mask13 | img_mask23 | img_mask33
        
        
        img_result1 = cv2.bitwise_and(roi1, roi1, mask=img_mask1)
        img_result2 = cv2.bitwise_and(roi2, roi2, mask=img_mask2)
        img_result3 = cv2.bitwise_and(roi3, roi3, mask=img_mask3) 
        
        cdst1 = cv2.cvtColor(img_result1, cv2.COLOR_BGR2GRAY)
        cdst2 = cv2.cvtColor(img_result2, cv2.COLOR_BGR2GRAY)
        cdst3 = cv2.cvtColor(img_result3, cv2.COLOR_BGR2GRAY)
        
        _, binary1 = cv2.threshold(cdst1, 0, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(cdst2, 0, 255, cv2.THRESH_BINARY)
        _, binary3 = cv2.threshold(cdst3, 0, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        opening1 = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, k)
        opening2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, k)
        opening3 = cv2.morphologyEx(binary3, cv2.MORPH_OPEN, k)
        

        contour1, _ = cv2.findContours(opening1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour2, _ = cv2.findContours(opening2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour3, _ = cv2.findContours(opening3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mmx1 = 0; mmy1 = 0; mmx2 = 0; mmy2 = 0; mmx3 = 0; mmy3 = 0;
        
        for i in contour1:
            M = cv2.moments(i)
            cx1= int(M['m10']/M['m00'])
            cy1= int(M['m01']/M['m00']) 
            mmx1 = cx1
            mmy1 = cy1    
            cv2.drawContours(roi1, contour1, -1, (0,255,0), 4)
            cv2.circle(roi1, (cx1, cy1), 3, (255, 0, 0), 4)
                
        for i in contour2:
            M = cv2.moments(i)
            cx2= int(M['m10']/M['m00'])
            cy2= int(M['m01']/M['m00'])  
            mmx2 = cx2
            mmy2 = cy2     
            cv2.drawContours(roi2, contour2, -1, (0,255,0), 4)
            cv2.circle(roi2, (cx2, cy2), 3, (255, 0, 0), 4)
            
        for i in contour3:
            M = cv2.moments(i)
            cx3 = int(M['m10']/M['m00'])
            cy3 = int(M['m01']/M['m00']) 
            mmx3 = cx3
            mmy3 = cy3      
            cv2.drawContours(roi3, contour3, -1, (0,255,0), 4)
            cv2.circle(roi3, (cx3, cy3), 3, (255, 0, 0), 4)
            
        cv2.line(img_color, (600, 600),(600, 500),(125,0,0),2)
        tan = (float(mmx1-mmx2)/200)
        distance = mmx2 - 400
        cv2.line(img_color, (600, 600),(600 + (mmx1-mmx2), 500),(125,0,0),2)
        
        cv2.putText(img_color, "Tangent : ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img_color, repr(tan), (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img_color, "Distance : ", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img_color, repr(distance), (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        
        print(tan)
        cv2.imshow('img_color', img_color)
        #cv2.imshow('img_result', opening)
        
        
        if (cv2.waitKey(1)) & 0xFF == 27:
            break   
        
line_track()

cap.release()
cv2.destroyAllWindows()
        
        
        
         