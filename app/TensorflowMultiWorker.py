import json
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
# from tensorflow.keras.callbacks import TensorBoard


start_time = time.time()
# Get your own index from environment variable
worker_index = int(os.environ.get("WORKER_INDEX", 0))
print(f"Running code for Worker {worker_index}")

cluster_config = {
    "worker": ["worker-0:2222", "worker-1:2222", "worker-2:2222"]
}

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': cluster_config,
    'task': {'type': 'worker', 'index': worker_index}
})

# Define the distributed strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Prepare the images so they can be used as input for the model
def preprocess_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

# Use Tensorflow strategy to create multiple workers and combine their results
with strategy.scope():
    IMAGE_SIZE = (128, 128)
    dataset_name = 'cats_vs_dogs'

    # Load dataset and split the data
    (train_dataset, test_dataset), info = tfds.load(
        name=dataset_name,
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True
    )

    train_dataset = train_dataset.map(preprocess_image)
    test_dataset = test_dataset.map(preprocess_image)

    # Define batch size
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    # Set batch size for datasets
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA  # Or OFF

    train_dataset = train_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)

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

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Get the number from the environment variable in the docker compose
    num_epochs = int(os.environ.get("NUM_EPOCHS", 1))  # Default to 1 if not set

    # Define TensorBoard callback for graphics
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs_TF_multi', update_freq=1)

    # Calculate the setup time
    setup_time = time.time() - start_time

    # Training
    start_time = time.time()
    model.fit(train_dataset, epochs= num_epochs, callbacks=[tensorboard_callback])
    training_time = time.time() - start_time

    # Evaluation/ testing the model
    start_time = time.time()
    test_loss, test_acc = model.evaluate(test_dataset)
    testing_time = time.time() - start_time
    # Printing the output
    print(f"Setup time: {setup_time} seconds")
    print(f"Training time: {training_time} seconds")
    print(f"Testing time: {testing_time} seconds")
    print(f"Total time: {training_time + setup_time + testing_time} seconds")
    print(f"Test accuracy: {test_acc}")