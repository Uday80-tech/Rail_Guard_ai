from ultralytics import YOLO
import cv2

model = YOLO("models/yolo26m.pt")  #yolo object detection model

cap = cv2.VideoCapture("videos/platform.mp4") #video source

# function to check side
def side_of_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    return 0 if ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 0 else 1

previous_train_x = None
train_moving = False
while True:
    ret,frame = cap.read()
    if not ret:
        break

    # yellow line
    # cv2.line(img= frame , pt1= (215,800) , pt2= (275,171) ,color= (0,255,255),thickness=5)
    
    result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )
    for box in result[0].boxes:
        if int(box.cls[0]) > -1:

            # identifying entities
            class_id = int(box.cls[0])
            x1,y1,x2,y2 = map(int , box.xyxy[0])


            #detecting train movement
 
            #if train moves then classify zones 
            if class_id == 0 :

                    # indicating feet
                feet = (int((x1 + x2)/2) , y2)
                cv2.circle(frame ,center = feet , radius= 1 , thickness=-1 , color = (255,255,255))
                    
                    # checking and displaying side
                side = side_of_line(feet ,(215,800), (275,171))
                cv2.rectangle(frame , pt1=(x1,y1),pt2=(x2,y2) , color =(0,255,0) if side == 0 else (0,0,255) , thickness = 1 )
                cv2.putText(frame , text = "In Safezone" if side == 0 else "In Danzerzone" , org=(x1,y1-10) , fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.35 , color=(255,0,0) , thickness=1)
                person_id = int(box.id[0]) if box.id is not None else -1
    
                # Person ID text
                cv2.putText(
                    frame,
                    f"ID: {person_id}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                







    
    cv2.imshow("Video Frame", frame  )


    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
