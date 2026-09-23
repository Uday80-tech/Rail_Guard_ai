

# function to check side
def side_of_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    return 0 if ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 0 else 1


def count_peoples(side):
    if side == 0:
        global safe_count
        safe_count += 1

    elif side == 1:
        global danger_count
        danger_count += 1

def train(distance):
    if distance > 5:
        global train_status
        train_status = "Running"

    else:
        train_status = "Stopped"
        

# ==============================================================================================================================

from ultralytics import YOLO
import cv2
import math

model = YOLO("models/yolo26x.pt")  #yolo object detection model

cap = cv2.VideoCapture("videos/train run.mp4") #video source

safe_count = 0
danger_count = 0   
train_status = "Stopped"
prev_x = None
prev_y = None


while True:
    ret,frame = cap.read()
    if not ret:
        break

    # yellow line
    # cv2.line(img= frame , pt1= (215,800) , pt2= (275,171) ,color= (0,255,255),thickness=5)

    
    

    result = model.track(frame , persist=True , tracker="bytetrack.yaml")



    
    for box in result[0].boxes:
        if int(box.cls[0]) > -1:

            # identifying entities
            class_id = int(box.cls[0])
            x1,y1,x2,y2 = map(int , box.xyxy[0])


            #detecting train movement

            if class_id == 6:

                train_id = int(box.id[0]) if box.id is not None else -1

                cv2.rectangle(frame , pt1=(x1,y1),pt2=(x2,y2) , color =(255,0,0) , thickness = 1 )
                cv2.putText(frame , text = f"Train {train_id}"  , org=(x1,y1-10) , fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.35 , color=(255,0,0) , thickness=1)

                current_x = (x1 + x2) /2
                current_y = (y1 + y2)/2

                if prev_x is not None:
                    distance = math.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
                    train(distance)

                prev_x = current_x
                prev_y = current_y


 
            #if train moves then classify zones 
            if class_id == 0 :

                #asigning id to persons
                person_id = int(box.id[0]) if box.id is not None else -1

                # indicating feet
                feet = (int((x1 + x2)/2) , y2)
                cv2.circle(frame ,center = feet , radius= 1 , thickness=-1 , color = (255,255,255))
                    
                # checking and displaying person with zones
                side = side_of_line(feet ,(215,800), (275,171))
                cv2.rectangle(frame , pt1=(x1,y1),pt2=(x2,y2) , color =(0,255,0) if side == 0 else (0,0,255) , thickness = 1 )
                cv2.putText(frame , text = "In Safezone" if side == 0 else "In Danzerzone" , org=(x1,y1-10) , fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.35 , color=(255,0,0) , thickness=1)



                # Person ID text
                cv2.putText(frame, text = f"ID: {person_id}" ,org = (x1, y1 - 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 0), thickness=2)
                
                #counting person in both zones 
                          
                count_peoples(side)

            


    cv2.putText(frame , f"SAFE COUNT :- {safe_count}" , org=(30,40) ,fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6 , color=(0,255,0) , thickness=2 )
    cv2.putText(frame , f"DANGER COUNT :- {danger_count}" , org=(30,60) ,fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6 , color=(0,0,255) , thickness=2 )
    cv2.putText(frame, f"Train Status:- {train_status}" , org = (250, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6 , color=(255,0,0) , thickness=2)
                



    cv2.imshow("Video Frame", frame  )


    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
