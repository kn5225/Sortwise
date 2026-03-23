import cv2 as cv

from cvzone.ClassificationModule import Classifier

cap = cv.VideoCapture(0)
maskClassifier = Classifier('keras_model.h5', 'labels.txt')
while True:
    _, img = cap.read()
    predection = maskClassifier.getPrediction(img)
    print(predection)
    cv.imshow("Image", img)
    key  = cv.waitKey(5)
    if key==27:
        break
cap.release()
cv.destroyAllWindows()
