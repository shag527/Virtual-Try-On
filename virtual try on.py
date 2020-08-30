import cv2

imgshirt = cv2.imread('top.jpg')
musgray = cv2.cvtColor(imgshirt,cv2.COLOR_BGR2GRAY) #grayscale conversion
ret, orig_mask = cv2.threshold(musgray,150 , 255, cv2.THRESH_BINARY)
orig_mask_inv = cv2.bitwise_not(orig_mask)
origshirtHeight, origshirtWidth = imgshirt.shape[:2]
face_cascade=cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#ret,img=cap.read()
#img=cv2.imread('person2.jpg')
#img_h, img_w = img.shape[:2]

frame_width=1200
frame_height=700


while True:
    _,img=cap.read()
    cv2.flip(img,1)
    img_h, img_w = img.shape[:2]
    cap.set(3,frame_width)
    cap.set(4,frame_height)
    #img=cv2.imread('tshirt.jpg')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),1)

        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_h
        face_y1 = y
        face_y2 = face_y1 + face_h

        # set the shirt size in relation to tracked face
        shirtWidth = 3 * face_w
        shirtHeight = int(shirtWidth * origshirtHeight / origshirtWidth)


        shirt_x1 = face_x2 - int(face_w/2) - int(shirtWidth/2) #setting shirt centered wrt recognized face
        shirt_x2 = shirt_x1 + shirtWidth
        shirt_y1 = face_y2 + 5 # some padding between face and upper shirt. Depends on the shirt img
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
        shirt = cv2.resize(imgshirt, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA) #resize all,the masks you made,the originla image,everything
        mask = cv2.resize(orig_mask, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (shirtWidth,shirtHeight), interpolation = cv2.INTER_AREA)

        # take ROI for shirt from background equal to size of shirt image
        roi = img[shirt_y1:shirt_y2, shirt_x1:shirt_x2]


        # roi_bg contains the original image only where the shirt is not
        # in the region that is the size of the shirt.
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(shirt,shirt,mask = mask_inv)
        dst = cv2.add(roi_bg,roi_fg)
        if face_y1+face_w+shirtHeight<frame_height:
            img[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst
        else:
            text='Too close to Screen'
            cv2.putText(img,text,(int(face_x1-face_w),int(face_y1)),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),1)
        break

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release() # Destroys the cap object
cv2.destroyAllWindows() # Destroys all the windows created by imshow