## python -m PyInstaller --name MegaMold_Time_Management_Application --icon ..\mmi.ico ..\test01.py

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
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import dlib
from kivy.core.window import Window
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
import threading


kv = """

SM:
    HomeScreen:
    RegistrationScreen:
    FaceEncodingInputScreen:
    LoginScreen:
    OperationsScreen:
    EnterTimeScreen:
    ETConfirmationScreen:
    LastScreen:

<HomeScreen>:
    canvas.before:
        Color: 
            rgba: (1,1,1,1)
        Rectangle:
            pos: self.pos
            size: self.size
    name: "home"
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        MDLabel:
            text: "Disclaimer : This data will be used for MegaMold International Employee time Management"
            pos_hint: 0.9, 0.1
            size_hint: 0.1, 0.1
            halign: 'center'
            font_size: 10
            bold: True
            italic: True
        IncrediblyCrudeClock:
            font_size: 25
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
        Image:
            source: 'Picture.png'
            size_hint_x: 0.5
            allow_stretch: True    
        MDRectangleFlatIconButton:
            icon: "login"
            text: " Login "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "FR"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
        MDRectangleFlatIconButton:
            icon: "account"
            text: " Register "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "reg"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"

<RegistrationScreen>:
    canvas.before:
        Color: 
            rgba: (1,1,1,1)
        Rectangle:
            pos: self.pos
            size: self.size
    name: "reg"
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        MDLabel:
            text: "Disclaimer : This data will be used for MegaMold International Employee time Management"
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 12
            bold: True
            italic: True
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.8)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
        Image:
            source: 'Picture.png'
            size_hint_x: 0.4
            allow_stretch: True 
        MDLabel:
            id: label_name
            text: "Employee Name:"
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 30
            bold: True
            italic: True
        MDTextField:
            id: empname
            hint_text_color: 'black'
            hint_text: "Please enter Full Name"
            mode: "fill"
            fill_color: 0, 0, 0, .4
            icon_right: "counter"
            text_color: 'black'
            bold: True
            required: True
            size_hint: 0.4, 0.4
            multiline: False
            width: 200
            font_size: 18
            pos_hint: {"center_x": 0.5}
            pos_hint: {"center_y": 0.5}  
        MDLabel:
            text: "Employee ID:"
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 30
            bold: True
            italic: True
        MDTextField:
            id: empid
            hint_text_color: 'black'
            hint_text: "Please enter Employee ID"
            mode: "fill"
            fill_color: 0, 0, 0, .4
            icon_right: "counter"
            text_color: 'black'
            bold: True
            required: True
            size_hint: 0.4, 0.4
            multiline: False
            input_filter: 'int'
            width: 200
            font_size: 18
            pos_hint: {"center_x": 0.5}
            pos_hint: {"center_y": 0.5} 
        MDRectangleFlatIconButton:
            id: reg_but
            icon: "account-plus"
            text: " Continue to capture Face encodings "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "FIS"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"    
        MDRectangleFlatIconButton:
            icon: "exit-run"
            text: " Go Back "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            on_release:
                app.root.current = "last"
                root.manager.transition.direction = "right"

<FaceEncodingInputScreen>:
    canvas.before:
        Color: 
            rgba: (1,1,1,1)
        Rectangle:
            pos: self.pos
            size: self.size
    name: "FIS"
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        MDLabel:
            text: "Disclaimer : This data will be used for MegaMold International Employee time Management"
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 12
            bold: True
            italic: True
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.8)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2   
        MDLabel:
            text: "Instructions: Please Capture an Still Image facing towards the camera"
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 12
            bold: True
            italic: True
        Image:
            id: video_FIS
            size_hint: 0.9, 0.9
            halign: 'center'
            allow_stretch: True  
        MDRectangleFlatIconButton:
            id: FIS_but
            icon: "camera-gopro"
            text: " Capture Face Encodings "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "home"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"    
        MDRectangleFlatIconButton:
            icon: "exit-run"
            text: " Go Back "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            on_press: root.on_stop()
            on_release:
                app.root.current = "last"
                root.manager.transition.direction = "right" 

<LoginScreen>:
    name: "FR"
    canvas.before:
        Color: 
            rgba: (1,1,1,1)
        Rectangle:
            pos: self.pos
            size: self.size
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.1, 0.1    
        MDLabel:
            text: "Instructions: Kindly Be Stationary and face towards the Camera"
            pos_hint: 0.9, 0.1
            size_hint: 0.1, 0.1
            halign: 'center'
            font_size: 12
            bold: True
            italic: True
        Image:
            id: video
            size_hint: 0.9, 0.9
            halign: 'center'
            allow_stretch: True

<OperationsScreen>:
    name: "OP"
    canvas.before:
        Color: 
            rgba: (1,1,1,1)
        Rectangle:
            pos: self.pos
            size: self.size
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.1, 0.1
        Image:
            source: 'Picture.png'
            size_hint_x: 0.5
            allow_stretch: True    
        MDRectangleFlatIconButton:
            icon: "clock-check-outline"
            text: " Clock IN "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            # on_press: root.on_leave_CI()
            on_release:
                app.root.current = "last"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"   
        MDRectangleFlatIconButton:
            icon: "clock-fast"
            text: " Clock Out "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            # on_press: root.on_leave_CI()
            on_release:
                app.root.current = "last"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"   
        MDRectangleFlatIconButton:
            icon: "timetable"
            text: " Enter Time "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "ET"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"    
        MDRectangleFlatIconButton:
            icon: "exit-run"
            text: " Go Back "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            on_release:
                app.root.current = "last"
                root.manager.transition.direction = "right"

#:import Factory kivy.factory.Factory

<MySpinnerOption@SpinnerOption>:
    background_color: '#4169E1'
    background_down: ''       

<EnterTimeScreen>:
    name: "ET"
    canvas.before:
        Color:
            rgba: (191,205,214,1)
        Rectangle:
            pos: self.pos
            size: self.size
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.2, 0.2
        Image:
            source: 'Picture.png'
            size_hint_x: 0.4
            allow_stretch: True   
        MDLabel:
            id: jobnum_label
            text: "Job Number:"
            size_hint: 0.2, 0.2
            halign: 'center'
            font_size: 30
            bold: True
            italic: True
        MDTextField:
            id: job_num
            hint_text_color: 'black'
            hint_text: "Please enter valid Job Number"
            mode: "fill"
            fill_color: 0, 0, 0, .4
            icon_right: "counter"
            text_color: 'black'
            bold: True
            required: True
            size_hint: 0.4, 0.4
            multiline: False
            input_filter: 'float'
            width: 200
            font_size: 18
            pos_hint: {"center_x": 0.5}
            pos_hint: {"center_y": 0.5}
        MDLabel:
            id: op_num
            text: "Please Select Operation Name:"
            size_hint: 0.4, 0.4
            halign: 'center'
            font_size: 30
            bold: True
            italic: True  
        Spinner:
            background_color: '#4169E1'
            id: spinner_id
            text: "Operation Selected : None"
            color: '#4169E1'
            size_hint: 0.3, 0.3
            option_cls: Factory.get("MySpinnerOption")
            values: [" 3 - Design [Engineering Department] ", " 5 - 3D Machining [ CNC Department ] ", " 6 - 2D Machining [ CNC Department ] ", " 7 - Radial Drill/Boring Mill [ CNC Department ] ", " 60 - GunDrill [ CNC Department ] ", " 10 - Quality Control [ Inspection ] ", " 11 - Trucking [ Trucking Department ] ", " 59 - Freight [ Trucking Department ] ", " 20 - Mold Making [ Mold Making ] ", " 24 - Spotting [ Mold Making ] ", " 23 - EDM ", " 40 - Shop Maintenance [ Indirect Labour ] "," 41 - Computer Maintenance [ Indirect Labour ] "," 42 - Training [ Indirect Labour ] "," 43 - Machine Repair [ Indirect Labour ] "," 44 - Indirect [ Indirect Labour ] "," 45 - Janitorial [ Indirect Labour ] "]
            on_text: root.spinner_clicked(spinner_id.text)
        MDTextField:
            id: time_work
            hint_text: "Please enter the time for the operation in Hours. Eg: 1.5 means 1 hour and 30 minutes"
            mode: "fill"
            fill_color: 0, 0, 0, .4
            icon_right: "timer-plus-outline"
            bold: True
            required: True
            size_hint: 0.4, 0.4
            multiline: False
            input_filter: 'float'
            width: 200
            font_size: 18
            pos_hint: {"center_x": 0.5}
            pos_hint: {"center_y": 0.5}
        MDRectangleFlatIconButton:
            id: et_but
            icon: "tag-arrow-up-outline"
            text: " Add Work Entry "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            # on_press:root.ET()
            on_release:
                app.root.current = "ETC"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
        MDRectangleFlatIconButton:
            icon: "exit-run"
            text: " Go Back "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            on_release:
                app.root.current = "last"
                root.manager.transition.direction = "right"

<ETConfirmationScreen>:
    name: "ETC"
    canvas.before:
        Color:
            rgba: (191,205,214,1)
        Rectangle:
            pos: self.pos
            size: self.size
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.5, 0.5
        Image:
            source: 'Picture.png'
            size_hint_x: 0.5
            allow_stretch: True    
        MDRectangleFlatIconButton:
            # id: et_but
            icon: "pen-plus"
            text: " Thank you, Make Another Work Entry "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.6)
            bold: True
            background_color: '#4169E1'
            on_release:
                app.root.current = "ET"
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"   
        MDRectangleFlatIconButton:
            icon: "exit-run"
            text: " Go Back "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            # on_press: root.stop()
            on_release:
                app.root.current = "last"
                root.manager.transition.direction = "right"
<LastScreen>:
    name: "last"
    canvas.before:
        Color:
            rgba: (191,205,214,1)
        Rectangle:
            pos: self.pos
            size: self.size
    GridLayout:
        cols:1
        pos_hint: {"center_x": 0.5, "center_y": 0.5}
        IncrediblyCrudeClock:
            font_size: 30
            bold: True
            background_color: (1,1,1,.9)
            halign: 'center'
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            pos_hint: 0.9, 0.1
            size_hint: 0.5, 0.5
        Image:
            source: 'Picture.png'
            size_hint_x: 0.5
            allow_stretch: True    
        MDRectangleFlatIconButton:
            icon: "archive-check-outline"
            text: " Thank you "
            font_size: "40sp"
            md_bg_color: '#4169E1'
            text_size: 20
            size_hint: (1,0.3)
            bold: True
            background_color: '#4169E1'
            theme_text_color: "Custom"
            text_color: "white"
            line_color: "green"
            icon_color: "white"
            on_release:
                app.root.current = "home"
                root.manager.transition.direction = "right"
"""

detector1 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
prototxt = 'face_detector/deploy.prototxt'
caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
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
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
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
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("ImagesAttendance/{}.png".format(EN), frame)
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
        self.cap.release()
        Clock.unschedule(self.update)
        cv2.destroyAllWindows()

    def on_leave(self):
        cv2.destroyAllWindows()


class LoginScreen(MDScreen):
    # COUNTER = False
    def Go_Home(self, *args):
        self.manager.current = "home"
        self.on_stop()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.COUNTER = True
        self.classNames = []
        self.encodeListKnown = []
        self.load_images()
        self.capture = None
        self.Encoding_Reload_Prompt = False
        self.spoof_dialog = None
        # self.Spoof_Counter = 0
        # self.Unidentified_Counter = 0
        # self.capture = cv2.VideoCapture(0)
        self.image = Image()

    def load_images(self):
        path = '.\ImagesAttendance'
        images = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = findEncodings(images)

    def on_enter(self, *args):
        global Encoding_Reload_Prompt
        # self.ids.pro_but.disabled = True
        # self.COUNTER = True
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        Clock.schedule_interval(self.Go_Home,20)
        # self.Spoof_Counter = 0
        # self.Unidentified_Counter = 0
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
            self.ids.video.texture = texture  # id video is defined in kv
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
                            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, tolerance=0.45)
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
                                # self.ids.pro_but.disabled = False
                                self.manager.current = "OP"
                                self.on_stop()
                    else:
                        # current_time = time.asctime()
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        file_name = "C:/Users/Research/PycharmProjects/MMI_Test06/Spoof_Attacks/" + f"{current_time}.png"
                        if not os.path.exists("C:/Users/Research/PycharmProjects/MMI_Test07/Spoof_Attacks/"):
                            os.makedirs("C:/Users/Research/PycharmProjects/MMI_Test07/Spoof_Attacks/")
                        print(file_name)
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

    def on_stop(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)


class OperationsScreen(Screen):
    pass


class EnterTimeScreen(MDScreen):
    def spinner_clicked(self, value):
        self.ids.op_num.text = f' Operation Selected : {value}'

    def on_leave(self, *args):
        self.ids.job_num.text = ""
        self.ids.op_num.text = 'No Operation Selected'
        self.ids.time_work.text = ""


class ETConfirmationScreen(Screen):
    pass
class LastScreen(Screen):
    pass
class IncrediblyCrudeClock(MDLabel):
    def __init__(self, **kwargs):
        super(IncrediblyCrudeClock, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1)
    def update(self, *args):
        self.text = time.asctime()

class MMI(MDApp):
    def build(self):
        Window.set_title("COMPANY NAME")
        Window.icon = "mmi.ico"
        Window.unbind(on_request_close=self.stop)
        Window.borderless = True
        Window.fullscreen = True
        return Builder.load_string(kv)

if __name__ == "__main__":
    MMI().run()
