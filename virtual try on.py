import tkinter as tk
import cv2
import PIL.Image,PIL.ImageTk
import time
import numpy as np
import keyboard

frame_width=1300
frame_height=700
ID = 0
offset=0


class App:
    def __init__(self, window, window_title):
        self.window=window
        self.window.title(window_title)
        global offset

        self.cap=VideoCapture()

        self.back_label = tk.Label(window, bg='black')
        self.back_label.place(relwidth=1, relheight=1,relx=0,rely=0)

        self.frame = tk.Frame(window, bg='black', bd=5)
        self.frame.place(relx=0.44, rely=0.05, width=frame_width-11, height=frame_height-2, anchor='n')

        self.buttonframe = tk.Frame(window, bg='black', bd=5)
        self.buttonframe.place(relx=0.93, rely=0.5, width=200, height=195, anchor='n')

        self.logoframe = tk.Frame(window, bg='black', bd=5)
        self.logoframe.place(relx=0.90, rely=0.055, width=200, height=250, anchor='n')

        self.canvas = tk.Canvas(self.frame, width=self.cap.width, height=self.cap.height)
        self.canvas.pack()

        logo = tk.PhotoImage(file=r'images/logo.png')

        self.label = tk.Label(self.logoframe, image=logo)
        self.label.place(relwidth=1, relheight=1 ,relx=0, rely=0)

        self.btn_snapshot=tk.Button(self.buttonframe,text='Snapshot',width=20,command=self.snapshot,bg='#ffffcc',font=('Courier',16))
        #self.btn_snapshot.grid(row=0,column=0,sticky = 's', pady = 2)
        self.btn_snapshot.pack(anchor='n',expand=True)

        #self.btn_shirtincrease = tk.Button(self.buttonframe, text='+', width=20, command=self.increase(offset),bg='#ffffcc',font=('Courier',16))
        #self.btn_snapshot.grid(row=1, column=0,sticky = 's', pady = 2)
        #self.btn_shirtincrease.pack(anchor='n', expand=True)

        #self.btn_shirtdecrease = tk.Button(self.buttonframe, text='-', width=20, command=self.decrease(offset),bg='#ffffcc',font=('Courier',16))
        #self.btn_snapshot.grid(row=2, column=0,sticky = 's', pady = 2)
        #self.btn_shirtdecrease.pack(anchor='n', expand=True)

        self.delay = 5
        self.update()

        self.window.mainloop()

    def snapshot(self):
        _,frame=self.cap.get_frame()
        if _:
            cv2.imwrite("frame-"+ time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def update(self):
        _,frame=self.cap.get_frame()

        if _:
            self.photo=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0,image=self.photo,anchor=tk.NW)
        self.window.after(self.delay, self.update)



class VideoCapture:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Unable to open Camera as ",self.cap.isOpened())

        self.width=frame_width
            #self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height=frame_height
            #self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        #self.window.mainloop()

    def get_frame(self):
        if self.cap.isOpened():
            _,frame=self.cap.read()
            cv2.flip(frame, 1)
            if _:
                assert not isinstance(frame,type(None)),'frame not found'

                global ID,offset
                shirts_type = ['images/tshirt4.jpg', 'images/top4.jpg']
                threshold=[200,254]
                shirt_id = ID
                imgshirt = cv2.imread(shirts_type[shirt_id])
                musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # grayscale conversion
                ret, orig_mask = cv2.threshold(musgray, threshold[ID], 255, cv2.THRESH_BINARY)
                #cv2.imshow('frame',orig_mask)
                orig_mask_inv = cv2.bitwise_not(orig_mask)
                origshirtHeight, origshirtWidth = imgshirt.shape[:2]
                face_cascade = cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


                img_h, img_w = frame.shape[:2]
                self.cap.set(3, frame_width)
                self.cap.set(4, frame_height)
            # img=cv2.imread('tshirt.jpg')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)

                    face_w = w
                    face_h = h
                    face_x1 = x
                    face_x2 = face_x1 + face_w
                    face_y1 = y
                    face_y2 = face_y1 + face_h

                # set the shirt size in relation to tracked face
                    shirtWidth = int(2.9 * face_w+ offset)
                    shirtHeight = int((shirtWidth * origshirtHeight / origshirtWidth)+offset/3)
                    cv2.putText(frame,(str(shirtWidth)+" "+str(shirtHeight)),(x+w,y+h),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

                    shirt_x1 = face_x2 - int(face_w / 2) - int(shirtWidth / 2)                             # setting shirt centered wrt recognized face
                    shirt_x2 = shirt_x1 + shirtWidth
                    shirt_y1 = face_y2 + 5                                                       # some padding between face and upper shirt. Depends on the shirt img
                    shirt_y2 = shirt_y1 + shirtHeight

                # Check for clipping
                    if shirt_x1 < 0:
                        shirt_x1 = 0
                    if shirt_y1 < 0:
                        shirt_y1 = 0
                    if shirt_x2 > img_w:
                        shirt_x2 = img_w
                    if shirt_y2 > img_h:
                        shirt_y2 = img_h

                    shirtWidth = shirt_x2 - shirt_x1
                    shirtHeight = shirt_y2 - shirt_y1
                    if shirtWidth < 0 or shirtHeight < 0:
                        continue

                # Re-size the original image and the masks to the shirt sizes
                    shirt = cv2.resize(imgshirt, (shirtWidth, shirtHeight),interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(orig_mask, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
                    mask_inv = cv2.resize(orig_mask_inv, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)

                # take ROI for shirt from background equal to size of shirt image
                    roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]

                # roi_bg contains the original image only where the shirt is not
                # in the region that is the size of the shirt.
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
                    roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask_inv)
                    dst = cv2.add(roi_bg, roi_fg)


                    kernel = np.ones((5, 5), np.float32) / 25
                    imgshirt = cv2.filter2D(dst, -1, kernel)

                    if face_y1 + shirtHeight +face_h< frame_height:
                        #cv2.putText(frame, "press 'n' key for next item and 'p' for previous item", (x, y),cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255),1)
                        frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst

                    else:
                        text = 'Too close to Screen'
                        #cv2.putText(frame, "press 'n'  key for next item and 'p' for previous item", (x-200, y-200),cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 1)
                        cv2.putText(frame, text, (int(face_x1-face_w/4.3), int(face_y1)), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 250), 1)

                    if keyboard.is_pressed('m' or 'M'):
                        ID= 0

                    if keyboard.is_pressed('W' or 'w'):
                        ID= 1

                    if keyboard.is_pressed('i'):
                        if offset>100:
                            print("THIS IS THE MAX SIZE AVAILABLE")
                        else:
                            offset+=50
                            print('+ pressed')

                    if keyboard.is_pressed('d'):
                        if offset <0:
                            print("THIS IS THE MIN SIZE AVAILABLE")
                        else:
                            offset -= 50
                            print('- pressed')



                if _:
                    return (_,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                else:
                    return (_,None)
        else:
            return (None, None)

App(tk.Tk(),"Virtual Mirror")