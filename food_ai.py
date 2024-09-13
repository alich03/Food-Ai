from ultralytics import YOLO
import cv2
import numpy as np
import os

mymodel=YOLO("best.pt")
my_classes=['Bread','Katchup','Cabbage','Tomato','Fried Egg','Mayonnaise']

mymodel=YOLO("RecipeAi.pt")
my_classes=['bread','cheese','cucumber','tomato']

input_path=0
output_path=""  
writer = None

cap = cv2.VideoCapture(input_path)  

b_done_color=(111,0,117)
k_done_color=(111,0,117)
c_done_color=(111,0,117)
t_done_color=(111,0,117)
f_done_color=(111,0,117)
m_done_color=(111,0,117)

progress=420
while True:
    
    ret, frame = cap.read()

    if not ret:
        break

    result=mymodel.predict(frame,conf=0.9)
    cc_data=np.array(result[0].boxes.data)


    cv2.rectangle(frame,(400,0),(600,600),(111,0,117),-1)
    cv2.putText(frame, "FOOD AI", (450,30), cv2.FONT_HERSHEY_SIMPLEX,0.85,(255, 255, 255), 2)

    cv2.circle(frame, (430, 60), 6, b_done_color, -1)
    cv2.putText(frame, "Bread", (450,65), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)

    cv2.circle(frame, (430, 90), 6, k_done_color, -1)
    cv2.putText(frame, "Katchup", (450,95), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)

    # cv2.circle(frame, (430, 120), 6, c_done_color, -1)
    # cv2.putText(frame, "Cabbage", (450,125), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)

    cv2.circle(frame, (430, 120), 6, t_done_color, -1)
    cv2.putText(frame, "Tomato", (450,125), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)

    cv2.circle(frame, (430, 150), 6, f_done_color, -1)
    cv2.putText(frame, "Fried Egg", (450,155), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)

    cv2.circle(frame, (430, 180), 6, m_done_color, -1)
    cv2.putText(frame, "Mayonnaise", (450,185), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 123, 233), 1)


    cv2.putText(frame, "Completeness", (435,255), cv2.FONT_HERSHEY_SIMPLEX,0.59,(0, 213, 0), 1)
    cv2.rectangle(frame,(420,275),(580,295),(255,0,0),1)

    cv2.putText(frame, "Author : Ali Hasnain", (415,320), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 0), 1)
    cv2.rectangle(frame,(420,275),(progress,295),(0,255,0),-1)
    if len(cc_data) != 0:
                xywh=np.array(result[0].boxes.xywh).astype("int32")
                xyxy=np.array(result[0].boxes.xyxy).astype("int32")
                
                for (x1, y1, _, _), (_, _, w, h), (_,_,_,_,conf,clas) in zip(xyxy, xywh,cc_data):
                           
                                    cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,255),2)
                                    text = f"{my_classes[int(clas)]} {np.round(conf*100,1)}%"
                                    cv2.putText(frame, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0, 0, 255), 2)

                                    if int(clas) == 0:
                                           b_done_color=(0,255,0)
                                           progress=455
                                           
                                     
                                    if int(clas) == 1:
                                           k_done_color=(0,255,0)
                                           progress=484
                                     
                                    # if int(clas) == 2:
                                    #        g_done_color=(0,255,0)
                                    #        progress=516
                                     
                                    if int(clas) == 3:
                                           t_done_color=(0,255,0)
                                           progress=516
                                     
                                    if int(clas) == 4:
                                           f_done_color=(0,255,0)
                                           progress=548
                                     
                                    if int(clas) == 5:
                                           m_done_color=(0,255,0)
                                           progress=580
                                           

    cv2.imshow("Model Testing", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if output_path != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    
    if writer is not None:
        print("[INFO] writing stream to output")
        writer.write(frame)

cap.release()
cv2.destroyAllWindows()


mymodel=YOLO("yolov8n-pose.pt")


mymodel.predict(source=0,show=True,save=True)

import cv2
import numpy as np

video_reader = cv2.VideoCapture("food_ai.avi")

video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))


fps = video_reader.get(cv2.CAP_PROP_FPS)

IMAGE_HIGHT , IMAGE_WIDTH = 40, 40

SEQUENCE_LENGHT = 10

DATASET_DIR = " "

CLASSES_LIST = ['Normal','Shoplifting']

def frame_extraction(video_path):
  
  frame_list= []

  video_reader = cv2.VideoCapture(video_path)

  fps = video_reader.get(cv2.CAP_PROP_FPS)

  print(f" FPS  {fps}")

  video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

  skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGHT),1)

  for frame_counter in range(SEQUENCE_LENGHT):

    video_reader.set(cv2.CAP_PROP_POS_FRAMES,frame_counter * skip_frames_window)

    success, frame = video_reader.read()

    if not success:
      break

    resized_frame=cv2.resize(frame,(IMAGE_HIGHT, IMAGE_WIDTH))

    normalized_frame=resized_frame/255

    frame_list.append(normalized_frame)

  video_reader.release()

  return frame_list

fm=frame_extraction("food_ai.avi")





