import time
import datetime
import cv2
import boto3

rekog = boto3.client('rekognition')
video = cv2.VideoCapture(0)
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        #self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        print('camera loaded')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #success, image = self.video.read()
        success, image = video.read()
        overlay = image.copy()
        overlayTxt = image.copy()

        h,w = image.shape[:2]
        regImg = cv2.resize(image, (int(0.2*w), int(0.2*h)))
        _,newjpeg = cv2.imencode('.jpg', regImg)
        imgbytes = newjpeg.tobytes()
        t0 = time.time()
        resp = rekog.detect_labels(Image={'Bytes':imgbytes})

        cv2.rectangle(overlay, (10,10),(300,50+50*len(resp['Labels'])), (0,0,0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cnt = 1
        now = str(datetime.datetime.now())
        cv2.putText(image, now , (20,40*cnt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cnt = cnt + 1
        for label in resp['Labels']:
            outTxt = label['Name'] + ' (' + str(int(label['Confidence'])) + '%)'
            cv2.putText(image, outTxt, (20,40*cnt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cnt=cnt+1
      
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        t0 = time.time()
        resp = rekog.detect_labels(Image={'Bytes':imgbytes})
        print("{} ---> {}".format((time.time()-t0),resp["Labels"]))
        
        return jpeg.tobytes()
