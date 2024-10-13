import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from random import randint
import smtplib
import datetime

#PART OF THIS CODE WAS FOUND ON THE WEB

password = ""
from_email = ""  # must match the email used to generate the password


to_email = ""  # receiver email

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)



def send_email(to_email, from_email, object_detected, time_detected,frame):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "SMARTCAM Security Alert"
    # Add in the message body
    if object_detected ==1:
        message_body = f'ALERT - 1 human has been detected!! Detection time: {time_detected}'
    else:
        message_body = f'ALERT - {object_detected} humans have been detected!! Detection time: {time_detected}'

    message.attach(MIMEText(message_body, 'plain'))
    _, frame_jpeg = cv2.imencode('.jpg', frame)
    image_attachment = MIMEImage(frame_jpeg.tobytes())
    image_attachment.add_header('Content-Disposition', 'attachment', filename='frame.jpg')
    message.attach(image_attachment)
    server.sendmail(from_email, to_email, message.as_string())


class ObjectDetection:
    def __init__(self, capture_index):
        # default parameters
        self.capture_index = capture_index
        self.first_email_time= None
        self.first_email_number_people = None
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, conf, cls in zip(boxes, confs, clss):
            if conf >= 0.9:  
                label = f'{names[int(cls)]} {conf:.2f}'
                class_ids.append(int(cls)) 
                self.annotator.box_label(box, label=label, color=colors(int(cls), True))
        return im0, class_ids , confs

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids, confs = self.plot_bboxes(results, im0)
            zeroo= 0
            if (len(class_ids) > 0) and (zeroo in class_ids) and  (any(conf >= 0.6 for conf in confs)):  # Modify condition for email
                time_now= datetime.datetime.now()
                if self.first_email_time != None:
                    time_difference = time_now - self.first_email_time

                if self.first_email_time == None:
                    self.first_email_time = time_now 
                    self.first_email_number_people = class_ids.count(0)
                    ret, frame = cap.read()
                    send_email(to_email, from_email, class_ids.count(0), time_now, frame.copy())

                elif ( class_ids.count(0) > self.first_email_number_people):
                    self.first_email_time = time_now 
                    self.first_email_number_people = class_ids.count(0)
                    ret, frame = cap.read()
                    send_email(to_email, from_email, class_ids.count(0), time_now, frame.copy())
                    



                elif (time_difference.days * 24 * 3600 + time_difference.seconds >= randint( 45, 75) ) :
                    self.first_email_time = time_now 
                    self.first_email_number_people = class_ids.count(0)
                    ret, frame = cap.read()
                    send_email(to_email, from_email, class_ids.count(0), time_now, frame.copy())


            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()


detector = ObjectDetection(capture_index=0)
detector()

