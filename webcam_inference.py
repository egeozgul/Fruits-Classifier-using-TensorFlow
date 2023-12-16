import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained MobileNetV2 model
model = tf.keras.models.load_model('path_to_your_fruit_model.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

def segment_fruit_area(frame):
    """Segment potential fruit areas based on color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # This range should capture many fruits but might need adjustment for specific fruits or lighting conditions
    lower_range = np.array([20, 100, 100])
    upper_range = np.array([160, 255, 255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return the largest contour (if any)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

while True:
    ret, frame = cap.read()

    contour = segment_fruit_area(frame)
    
    if contour is not None:
        # Compute the bounding rectangle of the contour to crop
        x, y, w, h = cv2.boundingRect(contour)
        cropped = frame[y:y+h, x:x+w]
        cropped_resized = cv2.resize(cropped, (224, 224))
        
        image = tf.convert_to_tensor(cropped_resized, dtype=tf.float32) / 255.0
        image = tf.expand_dims(image, axis=0)
        
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = predictions[0][predicted_class[0]]

        # If the classifier confirms there's a fruit, draw the contour
        THRESHOLD = 0.9
        if confidence > THRESHOLD:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    cv2.imshow('Fruit Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
