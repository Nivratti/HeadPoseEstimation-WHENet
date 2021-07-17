#@title Define FaceDetection_RetinaFaceTF2
import cv2 
import imutils
from retinaface import RetinaFace

class FaceDetection_RetinaFaceTF2:
    def __init__(self, quality="normal"):
        self.face_detector = RetinaFace(quality=quality)

    def read(self, image_path):
        return self.face_detector.read(image_path)

    def detect_highlight_face(self, rgb_image, max_side_len=640, max_faces=1, return_boxformat="xywh"):
        ## Resize image
        scale_factor = None

        height, width, c = rgb_image.shape
        if width > max_side_len or height > max_side_len:
            if height > width:
                scale_factor = max_side_len / height

                resized = imutils.resize(rgb_image, height=max_side_len)
            else:
                scale_factor = max_side_len / width
                resized = imutils.resize(rgb_image, width=max_side_len)

            logger.info(f"scale_factor: {scale_factor}")
            # tn_image = resize((int(width * scale_factor), int(height * scale_factor)))

        if scale_factor:
            faces = self.face_detector.predict(resized)

            ## Rescaled face boxes -- restore
            if faces:
              for face in faces:
                  face["x1"] = int(face["x1"] / scale_factor)
                  face["y1"] = int(face["y1"] / scale_factor)
                  face["x2"] = int(face["x2"] / scale_factor)
                  face["y2"] = int(face["y2"] / scale_factor)
        else:
            faces = self.face_detector.predict(rgb_image)
    
        # print(faces)

        # faces is list of face dictionary
        # each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
        # faces=[{"x1":20,"y1":32, ... }, ...]

        if faces:
            result_img = self.face_detector.draw(rgb_image, faces)
        else:
            result_img = None
        # # save ([...,::-1] : rgb -> bgr )
        # cv2.imwrite("result_img.jpg",result_img[...,::-1])

        return (faces, result_img)