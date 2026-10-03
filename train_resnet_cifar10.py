import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import weightwatcher as ww

# --- 1. Configuration ---
# Check for available devices in order: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DATASET = 'CIFAR10' # 'CIFAR10' or 'MNIST'

print(f"Using device: {DEVICE}")
print(f"Training on: {DATASET}")

# --- 2. Data Loading and Preprocessing ---
if DATASET == 'CIFAR10':
    # For CIFAR10, images are 3x32x32
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for 3 channels
    ])
    trainset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=False, download=True, transform=transform)
    num_classes = 10
    input_channels = 3
elif DATASET == 'MNIST':
    # For MNIST, images are 1x28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize for 1 channel
    ])
    trainset = torchvision.datasets.MNIST(root='/Users/tanmoy/research/data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='/Users/tanmoy/research/data', train=False, download=True, transform=transform)
    num_classes = 10
    input_channels = 1
else:
    raise ValueError("Dataset must be 'CIFAR10' or 'MNIST'")


trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. Model Definition (ResNet-18) ---
model = torchvision.models.resnet18(weights=None) # Not using pre-trained weights

# Modify ResNet for the chosen dataset
if DATASET == 'CIFAR10':
    # CIFAR-10 images are 32x32. The default ResNet conv1 and maxpool are for larger images.
    # We'll use a smaller kernel and stride for the first convolution.
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # We can remove the initial max pooling
    model.maxpool = nn.Identity()
elif DATASET == 'MNIST':
    # MNIST images are 28x28 and grayscale.
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()


# Adjust the final fully connected layer for the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model.to(DEVICE)

# --- 4. Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
def train(epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for i, data in enumerate(progress_bar):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{running_loss / (i + 1):.3f}'})

# --- 6. Evaluation Loop ---
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Evaluating")
        for data in progress_bar:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            progress_bar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
    print(f'Accuracy on the test set: {100 * correct / total:.2f} %')


# --- 7. Main Execution ---
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train(epoch)
        test()
    print('Finished Training')
