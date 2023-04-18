import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path1, config_path1,model_path2, config_path2, class_names1,class_names2, conf_threshold=0.5):
        self.model1 = cv2.dnn.readNetFromCaffe(config_path1, model_path1) # model 1 is mobilenetssd, mobilenetssd uses caffe
        self.model2 =  cv2.dnn.readNetFromDarknet(model_path2,config_path2) # model 2 is yolo, yolo uses darknet
        self.conf_threshold = conf_threshold
        # threshold, pretty much how accurate we want the algorithm to be, how strict
        # we want our predictions. We are going to compare our predictions score with this,
        # so if we want to be more loose, we make the threshold value lower ( now it is at 0.5)
        self.class_names1 = class_names1
        self.class_names2 = class_names2
        self.video_capture = None

    def start_video_capture(self, video_source):
        self.video_capture = cv2.VideoCapture(video_source)
        # create the two windows
        cv2.namedWindow('MobileNet SSD', cv2.WINDOW_NORMAL)
        cv2.namedWindow('YoloV3', cv2.WINDOW_NORMAL)

    def stop_video_capture(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

    def detect_objects(self):
        while True:
            ret, frame = self.video_capture.read()
            #  we read each frame
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            # we make a blob, and we resize and normalize it, so that way lighting is not going to affect our algorithm
            self.model1.setInput(blob)
            detections = self.model1.forward()

            # we go through each detection and if it satisfies our condition of accuracy we draw the border and type the name
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    class_name = self.class_names1[class_id]
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x, y, w, h = box.astype('int')
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('MobileNet SSD', frame)

            cv2.waitKey(1)

            blob2 = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.model2.setInput(blob2)
            detections2 = self.model2.forward()

            for i in range(detections2.shape[0]):
                confidence = detections2[i, 4]
                if confidence > self.conf_threshold:
                    class_id = int(detections2[i, 5])
                    class_name = class_names2[class_id]
                    x, y, w, h = detections2[i, :4] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('YoloV3', frame)

            cv2.waitKey(1)

            if cv2.waitKey(1) == ord('q'):
                break

        self.stop_video_capture()


model_path1 = 'MobileNetSSD_deploy.caffemodel'
config_path1 = 'MobileNetSSD_deploy.prototxt'
model_path2 = 'yolov3.cfg'
config_path2 = 'yolov3.weights'
# objects that can be detected by mobilenetssd, do NOT delete any because the program will crash
class_names1 = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# objects that can be detected by yolov3, do NOT delete any because the program will crash
class_names2 = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
               'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

detector = ObjectDetector(model_path1, config_path1,model_path2, config_path2, class_names1,class_names2)
# the public cam link, I got it from insecam.org
detector.start_video_capture('http://38.81.159.248/mjpg/video.mjpg')
detector.detect_objects()
