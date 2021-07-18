import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis

def process_detection( model, img_bgr, bbox, is_draw_axis=True, display_all_euler_angle="full"):

    img = img_bgr
    # y_min, x_min, y_max, x_max = bbox # yolo v3 face detector
    x_min, y_min, x_max, y_max = bbox # retinaface tf2

    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img.shape[1])

    img_cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    # cv2.imwrite("img_cropped.jpg", img_cropped)
    
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = np.expand_dims(img_rgb, axis=0)

    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(img_rgb)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

    if is_draw_axis:
        draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min)//2 )

    if display_all_euler_angle == 'full':
        cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_min), int(y_min) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
        cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_min), int(y_min)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    return (img, yaw, pitch, roll)
    

def main(args):
    import os
    from loguru import logger
    from face_detection import FaceDetection_RetinaFaceTF2

    ## Set env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = WHENet('WHENet.h5')
    print(model.model.summary())

    image_bgr = cv2.imread("./Sample/maruti.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_detector = FaceDetection_RetinaFaceTF2()
    logger.info("Success! Initialized face detector..")
    faces, result_img = face_detector.detect_highlight_face(image_rgb, max_side_len=640)

    logger.info(f"Total faces: {len(faces)}")
    # logger.info(f"faces: {faces}")

    for idx, face in enumerate(faces):
        bbox = [faces[idx]["x1"], faces[idx]["y1"], faces[idx]["x2"], faces[idx]["y2"]]
        bboxes.append(bbox)

    ## Detect
    for bbox in bboxes:
        out_img, yaw, pitch, roll = process_detection(model, image_bgr, bbox, args)

    logger.info(f"Storing result in out.jpg ... \n")
    cv2.imwrite("out.jpg", out_img)
    cv2.imshow('output', out_img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='whenet single image demo')
    parser.add_argument('--snapshot', type=str, default='WHENet.h5', help='whenet snapshot path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    main(args)