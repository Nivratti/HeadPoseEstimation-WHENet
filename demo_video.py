import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis
import os
import argparse
from PIL import Image
from demo import process_detection

def main(args):
    from loguru import logger
    from face_detection import FaceDetection_RetinaFaceTF2

    whenet = WHENet(snapshot=args.snapshot)

    VIDEO_SRC = 0 if args.video == '' else args.video # if video clip is passed, use web cam
    cap = cv2.VideoCapture(VIDEO_SRC)
    print('cap info',VIDEO_SRC)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(args.output, fourcc, 30, (frame.shape[1], frame.shape[0]))  # write the result to a video

    face_detector = FaceDetection_RetinaFaceTF2()
    logger.info("Success! Initialized face detector..")
    
    while True:
        try:
            ret, frame = cap.read()
        except:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, result_img = face_detector.detect_highlight_face(image_rgb, max_side_len=640)

        logger.info(f"Total faces: {len(faces)}")
        # logger.info(f"faces: {faces}")

        bboxes = []
        for face in faces:
            bbox = [faces[0]["x1"], faces[0]["y1"], faces[0]["x2"], faces[0]["y2"]]
            bboxes.append(bbox)

        ## Detect
        for bbox in bboxes:
            out_img, yaw, pitch, roll = process_detection(whenet, frame, bbox, args)

        cv2.imshow('output', out_img)
        out.write(out_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='whenet demo with yolo')
    parser.add_argument('--video', type=str, default='IMG_0176.mp4',         help='path to video file. use camera if no file is given')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--display', type=str, default='simple', help='display all euler angle (simple, full)')
    parser.add_argument('--score', type=float, default=0.3, help='yolo confidence score threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='yolo iou threshold')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--output', type=str, default='test.avi', help='output video name')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)