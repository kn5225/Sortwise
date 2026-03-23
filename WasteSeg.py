import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2 as cv
import numpy as np
import tensorflow as tf

# Use legacy loader directly
import tf_keras
model = tf_keras.models.load_model('keras_model.h5')

labels = ["Paper", "Plastic", "Can", "Cardboard", "Cloth", "Empty", "Humans", "Glass"]

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    
    resized = cv.resize(img, (224, 224))
    normalized = (resized.astype("float32") / 127.5) - 1.0
    input_tensor = np.expand_dims(normalized, axis=0)
    
    preds = model.predict(input_tensor, verbose=0)[0]
    label_index = int(np.argmax(preds))
    confidence = float(preds[label_index])
    
    cv.putText(img, f"{labels[label_index]}: {confidence:.0%}",
               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv.imshow("Waste Segmentation", img)
    
    if cv.waitKey(5) == 27:
        break

cap.release()
cv.destroyAllWindows()
