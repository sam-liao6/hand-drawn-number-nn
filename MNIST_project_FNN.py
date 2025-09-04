import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define the transformation to convert images to PyTorch tensors
transform = ToTensor()

# Load the MNIST dataset with the specified transformation
training_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set Batch Size
batch_size = 64

# Create a DataLoader to load the dataset in batches
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# plt.figure(figsize=(15, 3))

# for i, (image, label) in enumerate(train_dataloader):
#     if i < 5:
#         plt.subplot(1, 5, i + 1)
#         plt.imshow(image[0].squeeze(), cmap='gray') 
#         plt.title(f"Label: {label[0]}")
#         plt.axis('off')
#     else:
#         break

# plt.tight_layout()
# plt.show()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 

print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Flattens the image into 728 pixel values
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(28*28, 512, bias=True), # Linear transfomration
            nn.ReLU(), # Applies ReLU function, if input is negative: 0, positive: remains the same
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 10, bias=True)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc
    
def train(model: torch.nn.Module, 
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim.Optimizer,
          device: torch.device = device):
    
    size = len(dataloader.dataset)
    
    # Put model into training mode
    model.train()
    
    # Add a loop to loop throguh the training batches
    for batch, (X, y) in enumerate(dataloader):
        # Put data on target device
        X, y = X.to(device), y.to(device)
        
        # Computer prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        
        # Optimizxer step
        optimizer.step()
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(model: torch.nn.Module, 
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module, 
          device: torch.device = device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Put model in testing mode
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward Pass
            pred = model(X)
            
            # Calculate Loss
            test_loss += loss_fn(pred, y).item()
            
            # Calculate Accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
    test(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
print("Done!")