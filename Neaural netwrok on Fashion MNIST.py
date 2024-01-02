# Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# Load and preprocess Fashion MNIST dataset (again)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Define ImageDataGenerator for data augmentation during training
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define ImageDataGenerator without augmentation for validation data
validation_datagen = ImageDataGenerator()

# Build a new CNN model with Batch Normalization and Dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Use an adaptive learning rate optimizer and a learning rate scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define a learning rate scheduler function
def scheduler(epoch, lr):
    if epoch % 5 == 0:
        return lr * 0.9
    else:
        return lr

# Create LearningRateScheduler object
lr_schedule = LearningRateScheduler(scheduler)

# Use ImageDataGenerator for data augmentation during training
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
validation_generator = validation_datagen.flow(test_images, test_labels, batch_size=32)

# Train the model using the generators
history = model.fit(train_generator,
                    steps_per_epoch=len(train_images) // 32,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(test_images) // 32,
                    callbacks=[lr_schedule])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Import necessary libraries for evaluation and visualization
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Plot Epoch vs Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Generate predictions for the test set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Create Confusion Matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Display Classification Report
class_names = [str(i) for i in range(10)]
report = classification_report(test_labels, predicted_labels, target_names=class_names)
print('Classification Report:\n', report)

# Show the plots
plt.show()

# Display the Confusion Matrix
print(cm)
