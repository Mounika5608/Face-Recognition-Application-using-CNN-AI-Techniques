## python -m PyInstaller --name Time_Management_Application --icon ..\mmi.ico ..\test01.py

from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivy.lang.builder import Builder
from kivy.factory import Factory as Factory
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatIconButton
from kivymd.icon_definitions import md_icons
import sys
import sklearn
from kivy.uix.screenmanager import ScreenManager, Screen
import math
from kivy.clock import Clock
from kivy.graphics import Color, Line
import os
import time
import cv2
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
import numpy as np
import face_recognition
from datetime import datetime
import datetime
import pandas as pd
from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
import pickle
import imutils
from kivy.core.window import Window
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
import threading
#import Service as IPL_Servcie
import xml.etree.ElementTree as gfg 

prototxt = './deploy.prototxt'
caffemodel = './res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
model = load_model('liveness.model')
le = pickle.loads(open('le.pickle', "rb").read())

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

class SM(ScreenManager):
    pass
class HomeScreen(MDScreen):
    pass
class RegistrationScreen(MDScreen):
    def on_enter(self, *args):
        self.ids.empname.text = ""
        self.ids.empid.text = ""

class FaceEncodingInputScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.image = Image()
        self.success_dialog = None

    def on_enter(self, *args):
        self.ids.FIS_but.bind(on_press=self.callback)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video_FIS.texture = texture

    def callback(self, instance):
        global EN
        global EID
        select_screen = self.manager.get_screen("reg")
        EID = select_screen.ids.empid.text
        EN = select_screen.ids.empname.text        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("Face_Directory/{}.png".format(EN), frame)
            self.cap.release()
            Clock.unschedule(self.update)
            self.success_dialog = MDDialog(title="Registration Successful",
                                           text="Employee has been successfully registered.",
                                           size_hint=(.8, None), height=dp(200),
                                           buttons=[MDFlatButton(text="OK", on_release=self.change_screen)])
            self.success_dialog.open()
            sm = MDApp.get_running_app().root
            FR_Ref = sm.get_screen("FR")
            FR_Ref.Encoding_Reload_Prompt = True
            my_list = [EID, EN, " "]
            my_string = ','.join(my_list)
            #IPL_Servcie.service.User_Master('SP_Insert_User_Master',my_string)
        else:
            print("Failed to capture image")

    def change_screen(self, instance):
        self.manager.current = "home"
        self.success_dialog.dismiss()
        Clock.unschedule(self.update)
        self.capture.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def on_stop(self):
        self.capture.release()
        Clock.unschedule(self.update)
        cv2.destroyAllWindows()

    def on_leave(self):
        cv2.destroyAllWindows()


class LoginScreen(MDScreen):
    def Go_Home(self, *args):
        self.manager.current = "home"
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)
        self.on_stop()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classNames = []
        self.encodeListKnown = []
        self.load_images()
        self.capture = None
        self.Encoding_Reload_Prompt = False
        self.spoof_dialog = None
        self.Login_Success_dialog = None
        self.image = Image()

    def load_images(self):
        path = './Face_Directory'
        self.classNames = []
        self.encodeListKnown = []
        images = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = findEncodings(images)

    def on_enter(self, *args):
        global Encoding_Reload_Prompt
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        Clock.schedule_interval(self.Go_Home,20)
        if self.Encoding_Reload_Prompt == True:
            self.load_images()
            self.Encoding_Reload_Prompt = False
        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video.texture = texture
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]
                    if label == "real":
                        facesCurFrame = face_recognition.face_locations(frame)
                        encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)
                        rects = net.forward()
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, tolerance=0.4)
                            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)

                            if matches[matchIndex]:
                                name = self.classNames[matchIndex].upper()
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                            1, cv2.LINE_AA)
                                cv2.putText(frame, "[INFORMATION]: PLEASE CLICK ON PROCEED BUTTON TO CONTINUE:",
                                            (1, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                label = "{}: {:.4f}".format(label, preds[j])
                                cv2.putText(frame, name, (startX, startY - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                              (0, 255, 0), 2)
                                self.Login_Success_dialog = MDDialog(title="Welcome " + name,
                                                             text=" Please Press the Proceed Button to Continue to Time Entries ",
                                                             size_hint=(.8, None), height=dp(300),
                                                             buttons=[MDFlatButton(text=" Proceed ",
                                                                                   on_release=self.change_screen_Login_Success_dialog)])
                                self.Login_Success_dialog.open()
                                self.manager.current = "OP"
                                self.on_stop()
                                global name_db
                                name_db = name
                    else:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        file_name = "Spoof_Attacks/" + f"{current_time}.png"
                        if not os.path.exists("./Spoof_Attacks"):
                            os.makedirs("./Spoof_Attacks")
                        cv2.imwrite(file_name, frame)
                        self.spoof_dialog = MDDialog(title="SPOOF ATTACK ALERT!!!",
                                                       text="Please Do not try to spoof the system. This instance has been recorded",
                                                       size_hint=(.8, None), height=dp(300),
                                                       buttons=[MDFlatButton(text="Close", on_release=self.change_screen_Spoof)])
                        self.spoof_dialog.open()
                        self.manager.current = "home"
                        Clock.unschedule(self.update)
                        Clock.unschedule(self.Go_Home)
                        self.capture.release()
                        cv2.destroyAllWindows()

    def change_screen_Spoof(self, instance):
        self.spoof_dialog.dismiss()

    def change_screen_Login_Success_dialog(self, instance):
        self.Login_Success_dialog.dismiss()
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)

    def on_stop(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)

class OperationsScreen(Screen):
    dialog_label_clock_in = None
    dialog_label_clock_out = None
    dialog = None

    def show_dialog_box_clock_in(self):
        Clock.schedule_once(self.create_dialog_box_clock_in, 0)

    def create_dialog_box_clock_in(self, dt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.dialog = MDDialog(  # Store dialog instance as attribute
            title="Clock In",
            type="custom",
            size_hint=(.8, None), height=dp(300),
            content_cls=MDLabel(
                text=f"Are you sure you want to clock in at {current_time}?",
                theme_text_color="Custom",
                text_color=(0, 0, 0, 1),
                font_style='H6'
            ),
            buttons=[
                MDFlatButton(
                    text="Cancel", font_style='H6', on_release=lambda x: self.dialog.dismiss(),
                ),
                MDFlatButton(            
                    self.on_leave_CI(),text="Clock In", font_style='H6', on_release=lambda x: self.clock_in()
                    
                 
                ),
            ],
        )
        self.dialog.open()

    def show_dialog_box_clock_out(self):
        Clock.schedule_once(self.create_dialog_box_clock_out, 0)

    def create_dialog_box_clock_out(self, dt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.dialog = MDDialog(
            title="Clock Out",
            type="custom",
            size_hint=(.8, None), height=dp(300),
            content_cls=MDLabel(
                text=f"Are you sure you want to clock out at {current_time}?",
                theme_text_color="Custom",
                text_color=(0, 0, 0, 1),
                font_style='H6'
            ),
            buttons=[
                MDFlatButton(
                    text="Cancel",font_style='H6', on_release=lambda x: self.dialog.dismiss()
                ),
                MDFlatButton(
                    self.on_leave_CO(),text="Clock Out", font_style='H6', on_release=lambda x: self.clock_out()
                ),
            ],
        )
        self.dialog.open()

    def clock_in(self):
        self.dialog.dismiss()
        self.manager.current = "last"

    def clock_out(self):
        self.dialog.dismiss()
        self.manager.current = "last"

    def on_leave_CI(self, *args):
       
        Var=name_db
        my_list = [Var,'1']
        my_string = ','.join(my_list)
        #IPL_Servcie.service.Clock_IN_Out('SP_Insert_CLock_IN_OUT',my_string)

    def on_leave_CO(self, *args):
        Var=name_db
        my_list = [Var,'0']
        my_string = ','.join(my_list)
        #IPL_Servcie.service.Clock_IN_Out('SP_Insert_CLock_IN_OUT',my_string)


  
  

def GenerateXML(self,fileName) :

    global root 
    root = gfg.Element("NewDataSet")
      
    # m1 = gfg.Element("Table")
    # root.append(m1)

    for i in range(11-1): 

        job='job_num'+str(i+1)
        Operation_No='spinner_id'+str(i+1)
        Time_Work='time_work'+str(i+1)

        # global root 
        # root = gfg.Element("NewDataSet")
      
        m1 = gfg.Element("Table")
        root.append(m1)
        
        b1 = gfg.SubElement(m1, "Job_Num")
        b1.text = self.ids[str(job)].text

        b2 = gfg.SubElement(m1, "Operation_No")
        b2.text = self.ids[str(Operation_No)].text
                    
        c1 = gfg.SubElement(m1, "Time_Work")
        c1.text = self.ids[str(Time_Work)].text

        tree = gfg.ElementTree(root)
        with open (fileName, "wb") as files :
            tree.write(files)
            
    global XML
    XML=gfg.tostring(root).decode()     

    return XML  
        
        
    
        
          

class EnterTimeScreen(MDScreen):

    def ET(self):
        Job_No=self.ids.job_num.text
        OP_No=self.ids.spinner_id.text
        Time_Work=self.ids.result_label.text
        Var=name_db
        print(Job_No,OP_No,Time_Work)
        Para = Job_No,OP_No,Time_Work
        print(Para)
        my_list = [Job_No, OP_No, Time_Work,Var]
        my_string = ','.join(my_list)
        #IPL_Servcie.service.Insert('SP_Insert_CLock_Master',my_string)

    def calculate_sum(self, *args):
        num_list = []
        for i in range(1, 11):
            if f"time_work{i}" in self.ids:
                num_text = self.ids[f"time_work{i}"].text
                num = float(num_text) if num_text else 0.0
                num_list.append(num)
        result = sum(num_list)
        rounded_result = round(result, 2)
        self.ids.result_label.text = str(rounded_result)

    def Enter_Time(self):
        self.dialog = MDDialog(text="Confirm to submit yor Time Entries",
                                         size_hint=(.8, None), height=dp(300),
                                         buttons=[MDFlatButton(text="Cancel",font_style='H6', on_release=lambda x: self.dialog.dismiss()),
                                                  MDFlatButton(text=" Proceed ",font_style='H6',
                                                               on_release=lambda x: self.change_last())
                                          ])
       
       
        
        XML_NEW=GenerateXML(self,"Catalog.xml")
        Var=name_db
        my_list = [Var,XML_NEW]
        my_string = ','.join(my_list)
        #IPL_Servcie.service.Insert('SP_Insert_Master_DATA',my_string)
        # jobnumber=self.ids[str(jon)].text
        
       
        self.dialog.open()

    def change_last(self):
        self.dialog.dismiss()
        self.manager.current = "last"

    def submit_add_job_entries(self):
        self.ids.job_num1.text = ""
        self.ids.spinner_id1.text = ""
        self.ids.time_work1.text = ""

        self.ids.job_num2.text = ""
        self.ids.spinner_id2.text = ""
        self.ids.time_work2.text = ""

        self.ids.job_num3.text = ""
        self.ids.spinner_id3.text = ""
        self.ids.time_work3.text = ""

        self.ids.job_num4.text = ""
        self.ids.spinner_id4.text = ""
        self.ids.time_work4.text = ""

        self.ids.job_num5.text = ""
        self.ids.spinner_id5.text = ""
        self.ids.time_work5.text = ""

        self.ids.job_num6.text = ""
        self.ids.spinner_id6.text = ""
        self.ids.time_work6.text = ""

        self.ids.job_num7.text = ""
        self.ids.spinner_id7.text = ""
        self.ids.time_work7.text = ""

        self.ids.job_num8.text = ""
        self.ids.spinner_id8.text = ""
        self.ids.time_work8.text = ""

        self.ids.job_num9.text = ""
        self.ids.spinner_id9.text = ""
        self.ids.time_work9.text = ""

        self.ids.job_num10.text = ""
        self.ids.spinner_id10.text = ""
        self.ids.time_work10.text = ""

        self.ids.job_num11.text = ""
        self.ids.spinner_id11.text = ""
        self.ids.time_work11.text = ""

    def on_enter(self):
        self.ids.job_num1.text = ""
        self.ids.spinner_id1.text = ""
        self.ids.time_work1.text = ""

        self.ids.job_num2.text = ""
        self.ids.spinner_id2.text = ""
        self.ids.time_work2.text = ""

        self.ids.job_num3.text = ""
        self.ids.spinner_id3.text = ""
        self.ids.time_work3.text = ""

        self.ids.job_num4.text = ""
        self.ids.spinner_id4.text = ""
        self.ids.time_work4.text = ""

        self.ids.job_num5.text = ""
        self.ids.spinner_id5.text = ""
        self.ids.time_work5.text = ""

        self.ids.job_num6.text = ""
        self.ids.spinner_id6.text = ""
        self.ids.time_work6.text = ""

        self.ids.job_num7.text = ""
        self.ids.spinner_id7.text = ""
        self.ids.time_work7.text = ""

        self.ids.job_num8.text = ""
        self.ids.spinner_id8.text = ""
        self.ids.time_work8.text = ""

        self.ids.job_num9.text = ""
        self.ids.spinner_id9.text = ""
        self.ids.time_work9.text = ""

        self.ids.job_num10.text = ""
        self.ids.spinner_id10.text = ""
        self.ids.time_work10.text = ""

        self.ids.job_num11.text = ""
        self.ids.spinner_id11.text = ""
        self.ids.time_work11.text = ""


class LastScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.timeout, 3)

    def timeout(self, dt):
        self.manager.current = 'home'

    def on_stop(self):
        pass

class ContentNavigationDrawer(BoxLayout):
    pass
class IncrediblyCrudeClock(MDLabel):
    def __init__(self, **kwargs):
        super(IncrediblyCrudeClock, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1)

    def update(self, *args):
        now = datetime.datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        self.text = f"{date_str}\n{time_str}"


class MMI(MDApp):
    def build(self):
        Window.set_title("XYZ Company Application")
        Window.unbind(on_request_close=self.stop)
        # Window.borderless = True
        # Window.fullscreen = True
        # return Builder.load_string(kv)
        return Builder.load_file("MMI.kv")

if __name__ == "__main__":
    MMI().run()
