import cv2
import glob
import os
face_cascade = cv2.CascadeClassifier('C:/Users/eslamomar/Desktop/ML/haarcascade_frontalface_default.xml')# face detection algorithm


def pre(path, saved_path):
    num = 1
    for i in glob.glob(path):#iterating the file of the images
        img = cv2.imread(i, 1)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)# detect objects min size 1.3 and max of 5
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4) #function that draw rec on the face [image, top left corner, bottom right corner, color of the image]
            cropped_img = img[y: y + w, x: x + h]

        #cv2.imshow('original!', img)
        resised= cv2.resize(cropped_img, (50, 50))
        #cv2.imshow('Done!', cropped_img)
        cv2.imwrite(os.path.join(saved_path, ('img'+str(num)+'.jpg')), resised)
        num += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    path = 'F:/4th year/Pattern/task1,2_python/images/*'
    saved_path = 'F:/4th year/Pattern/task1,2_python/detected faces/'
    pre(path, saved_path)
