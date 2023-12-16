import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import Callback

import os

# 1. Dataset Acquisition
train_dir = './fruits-360/Training'
test_dir = './fruits-360/Test'

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. Model Selection
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training started")


class myCallback(Callback):    
    def on_epoch_end(self, epoch,logs=None):
        print("Checking loss at end of epoch...")
        if logs['loss'] <= 0.25:
               self.model.stop_training = True
               
loss_callback_obj = myCallback()

# 3. Training
model.fit(train_generator, epochs=10, validation_data=test_generator,callbacks=[loss_callback_obj])

print("Training done")

# Save the trained model
model_path = "trainedModel.h5"
model.save(model_path)
print("MODEL GENERATED AND SAVED")

# Load the saved model for later use
#model = load_model(model_path)

# Inference function remains unchanged
def predict_fruit(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = tf.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    predicted_class = tf.argmax(prediction, axis=1).numpy()
    
    # Map class index to class label
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[predicted_class[0]]

# Example usage:
# image_path = "path_to_your_image.jpg"
# print(predict_fruit(image_path))
