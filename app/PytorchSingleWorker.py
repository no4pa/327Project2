import tensorflow as tf
import torch
import torch.utils.data
import time
import torch.nn as nn
import torch.optim as optim
import tensorflow_datasets as tfds

start_time = time.time()
# Load the dataset using TensorFlow an splitting data into train and test
dataset_name = 'cats_vs_dogs'
(train_dataset_tf, test_dataset_tf), info = tfds.load(
    name=dataset_name,
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

# Define a function to preprocess TensorFlow data and convert to PyTorch tensors
def preprocess_tf_to_torch(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    image = tf.transpose(image, [2, 0, 1])  # Change the format to (channels, height, width)
    return image, label


# Transform TensorFlow datasets to PyTorch tensors
train_dataset_torch = list(train_dataset_tf.map(preprocess_tf_to_torch))
test_dataset_torch = list(test_dataset_tf.map(preprocess_tf_to_torch))


# Create a custom collate function to handle conversion
def custom_collate(batch):
    images = torch.stack([torch.tensor(item[0].numpy()) for item in batch])
    labels = torch.tensor([item[1].numpy() for item in batch]).unsqueeze(1)
    return images, labels

# Define batch size
batch_size = 32

# Create data loaders with a batch size of 32 and custom collate function
train_loader = torch.utils.data.DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate)

test_loader = torch.utils.data.DataLoader(test_dataset_torch, batch_size=batch_size, collate_fn=custom_collate)

# Display dataset information
print(f"Train samples: {len(train_dataset_torch)}")
print(f"Test samples: {len(test_dataset_torch)}")


# Define the model architecture same to the original TensorFlow model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return torch.FloatTensor(x)

# Create instance of model
model = Net()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Setting the cpu as training device since tensorflow does the same
device = torch.device("cpu")
model.to(device)

# Calculate the setup time
setup_time = time.time() - start_time

start_time = time.time()
# We use the pre-defined number of epochs to determine how many iterations to train the network on
# Train the model
for epoch in range(3):
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.to(device).float()  # Cast labels to float

        # Forward pass
        outputs = model(images)
        # Cast outputs to float
        outputs = outputs.float()
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 3, loss.item()))
training_time = time.time() - start_time

# Test the model
start_time = time.time()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
        correct += (predicted == labels).sum().item()  # Accumulate correct predictions
        total += labels.size(0)  # Accumulate total samples

    accuracy = 100 * correct / total  # Calculate accuracy as percentage

test_accuracy = accuracy
testing_time = time.time() - start_time

# Print output
print(f"Setup time: {setup_time} seconds")
print(f"Training time: {training_time} seconds")
print(f"Testing time: {testing_time} seconds")
print(f"Total time: {training_time + setup_time + testing_time} seconds")

print('Accuracy of the network on the {} train images: {:.2f} %'.format(total, test_accuracy))

