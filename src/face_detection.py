import cv2 as cv
# from ultralytics import YOLO

class FaceDetector:

    @staticmethod
    def YuNet(img, Foreground=False):
        # Load the model
        detector = cv.FaceDetectorYN.create(
            r"C:\Users\lathi\Documents\projects\background blur\face_detection_yunet_2023mar.onnx", "", (320, 320)
        )

        height, width, _ = img.shape
        detector.setInputSize((width, height))

        # results[1] contains [x, y, w, h, x_eye_l, y_eye_l, ...]
        _, faces = detector.detect(img)

        bounding_boxes = []

        threshold = 0

        if faces is not None:
            threshold = len(faces)
            for face in faces:
                box = list(map(int, face[:4])) # x, y, w, h
                bounding_boxes.append(box)
              
        if Foreground:
            return bounding_boxes, threshold
        return bounding_boxes


    
