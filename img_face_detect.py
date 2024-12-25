import cv2
import os

face_dir = 'haar_faces'
detect_dir = 'haar_detected'
path = os.path.dirname(__file__)


def get_images(path):
    exts = ['.jpg', '.png']
    images = []
    for img in os.listdir(path):
        ext = os.path.splitext(img)[1]
        if ext.lower() not in exts:
            continue
        images.append(os.path.join(path, img))
    if images:
        create_dir()
    return images


def create_dir():
    dirs = [face_dir, detect_dir]
    for dir in dirs:
        dir = os.path.join(path, dir)
        if not os.path.exists(dir):
            os.makedirs(dir)


def haar_detection(image):
    img_path = os.path.join(image)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces!".format(len(faces)))

    if len(faces):
        i = 0
        copy_img = img.copy()

        # image name without extention
        base = os.path.split(image)[-1]
        name = os.path.splitext(base)[0]

        for (x, y, w, h) in faces:
            roi_color = img[y:y + h, x:x + w]
            resized_image = cv2.resize(roi_color, (64, 64))
            cv2.imwrite(os.path.join(path, face_dir, str(i) + '_' + name + '_detected.jpg'), resized_image)
            cv2.rectangle(copy_img, (x, y), (x + w, y + h), (255, 242, 0), 2)
            i += 1

        cv2.imwrite(os.path.join(path, detect_dir, name+'_detect_face.jpg'), copy_img)
        return copy_img


def show_img(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # path = 'D:\PythonSenior\images'
    images = get_images(path)
    print(images)

    for image in images:
        img = haar_detection(image)
        show_img(img)
