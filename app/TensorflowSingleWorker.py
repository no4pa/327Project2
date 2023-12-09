import tensorflow as tf
import tensorflow_datasets as tfds
import time

start_time = time.time()
dataset_name = 'cats_vs_dogs'
# download the dataset and split it into train and test
(train_dataset, test_dataset), info = tfds.load(
    name=dataset_name,
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

# Display dataset information
print(info)

IMAGE_SIZE = (128, 128)

# Prepare the images so they can be used as input for the model
def preprocess_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

train_dataset = train_dataset.map(preprocess_image)
test_dataset = test_dataset.map(preprocess_image)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs_TF_single', histogram_freq=1)

# Define batch size
batch_size = 32
# Set batch size for datasets
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# calculate setup time
setup_time = time.time() - start_time

# Train the model
start_time = time.time()
model.fit(train_dataset, epochs=3, callbacks=[tensorboard_callback])
training_time = time.time() - start_time
# Test the model
start_time = time.time()
test_loss, test_acc = model.evaluate(test_dataset)
testing_time = time.time() - start_time
# Print output
print(f"Setup time: {setup_time} seconds")
print(f"Training time: {training_time} seconds")
print(f"Testing time: {testing_time} seconds")
print(f"Total time: {training_time + setup_time + testing_time} seconds")
print(f"Test accuracy: {test_acc}")