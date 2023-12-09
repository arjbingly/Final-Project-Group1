import cv2
import numpy as np
from PIL import Image

def enlarge_bounding_boxes(image, faces, padding=20):
    if isinstance(image, np.ndarray):
        height, width, _ = image.shape
    elif isinstance(image, Image.Image):
        width, height = image.size
    else:
        raise ValueError("Unsupported image format. Use either numpy array (OpenCV) or PIL Image.")

    enlarged_boxes = []
    for (x, y, w, h) in faces:
        x_new = max(0, x - padding)
        y_new = max(0, y - padding)
        w_new = min(width - x_new, w + 2 * padding)
        h_new = min(height - y_new, h + 2 * padding)
        enlarged_boxes.append((x_new, y_new, x_new + w_new, y_new + h_new))
    return enlarged_boxes
def bounding_box(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return np.nan, None
    enlarged_faces = enlarge_bounding_boxes(image, faces, padding=20)
    for (x1, y1, x2, y2) in enlarged_faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return enlarged_faces, image
def bounding_box_pil(image):
    if isinstance(image, Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        enlarged_boxes, image_with_boxes = bounding_box(image_cv)
        if enlarged_boxes is None:
            print("No faces found.")
        elif len(enlarged_boxes) == 0:
            print("No faces found.")
        else:
            for box in enlarged_boxes:
                print(f"Bounding Box: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}")
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_with_boxes)
    else:
        raise ValueError("Unsupported image format. Use PIL Image.")

# Usage with OpenCV image
image_path ='test_img_01.png'
image = cv2.imread(image_path)
enlarged_boxes, image_with_boxes = bounding_box(image)

if enlarged_boxes is None:
    print("No faces found.")
elif len(enlarged_boxes) == 0:
    print("No faces found.")
else:
    for box in enlarged_boxes:
        print(f"Bounding Box: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}")
pil_image = Image.open('test_img_01.png')
bounding_box_pil(pil_image)
