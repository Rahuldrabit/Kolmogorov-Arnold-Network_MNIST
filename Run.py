import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Create directories for saving results and checkpoints if they don't exist
os.makedirs('e:/MNIST_Dataset_Work/KAN/checkpoints', exist_ok=True)
os.makedirs('e:/MNIST_Dataset_Work/KAN/results', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 15
input_dim = 28 * 28  # Flattened MNIST images
hidden_dim = 64
num_classes = 10
num_layers = 2

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# B-Spline basis function
class BSpline(nn.Module):
    def __init__(self, num_basis=10, domain_range=(-1, 1), degree=3):
        super(BSpline, self).__init__()
        self.num_basis = num_basis
        self.domain_min, self.domain_max = domain_range
        self.degree = degree
        
        # Initialize knots
        # We add 2*degree knots at the boundaries to ensure proper behavior
        self.knots = torch.linspace(self.domain_min, self.domain_max, num_basis + degree + 1)
        
    def cox_de_boor(self, x, i, k):
        """
        Evaluate the Cox-de Boor recursion formula at x for the i-th basis function of degree k
        """
        if k == 0:
            return torch.where((self.knots[i] <= x) & (x < self.knots[i+1]), 
                               torch.ones_like(x), torch.zeros_like(x))
        
        # Handle division by zero
        denominator_1 = self.knots[i+k] - self.knots[i]
        denominator_2 = self.knots[i+k+1] - self.knots[i+1]
        
        term_1 = torch.zeros_like(x)
        term_2 = torch.zeros_like(x)
        
        if denominator_1 > 1e-10:
            term_1 = ((x - self.knots[i]) / denominator_1) * self.cox_de_boor(x, i, k-1)
            
        if denominator_2 > 1e-10:
            term_2 = ((self.knots[i+k+1] - x) / denominator_2) * self.cox_de_boor(x, i+1, k-1)
            
        return term_1 + term_2
    
    def forward(self, x):
        # Clamp x to be within domain range
        x = torch.clamp(x, self.domain_min, self.domain_max - 1e-7)
        
        # Evaluate all basis functions at x
        basis_values = torch.stack([self.cox_de_boor(x, i, self.degree) 
                                   for i in range(self.num_basis)])
        
        return basis_values.transpose(0, 1)

# More efficient implementation using PyTorch's functional API
class BSplineBasis(nn.Module):
    def __init__(self, num_basis=10, domain_range=(-1, 1)):
        super(BSplineBasis, self).__init__()
        self.num_basis = num_basis
        self.domain_min, self.domain_max = domain_range
        
        # Create centers and widths for the basis functions
        centers = torch.linspace(domain_range[0], domain_range[1], num_basis)
        widths = (domain_range[1] - domain_range[0]) / (num_basis - 1) if num_basis > 1 else 1.0
        
        self.centers = nn.Parameter(centers, requires_grad=False)
        self.widths = widths
        
    def forward(self, x):
        # Clamp inputs to domain range
        x = torch.clamp(x, self.domain_min, self.domain_max)
        
        # Compute squared distances
        x_expanded = x.unsqueeze(-1)  # Shape: [batch_size, 1]
        centers_expanded = self.centers.unsqueeze(0)  # Shape: [1, num_basis]
        
        # Compute B-spline basis
        z = (x_expanded - centers_expanded) / self.widths
        z = torch.clamp(1 - torch.abs(z), min=0)
        
        return z

# KAN Layer
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_basis=10):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        
        # Initialize the basis functions
        self.basis_functions = nn.ModuleList([
            BSplineBasis(num_basis=num_basis) for _ in range(in_dim)
        ])
        
        # Initialize weights (n_basis x n_inputs x n_outputs)
        self.weights = nn.Parameter(
            torch.Tensor(num_basis, in_dim, out_dim).uniform_(-0.1, 0.1)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Apply basis functions to each input dimension
        basis_outputs = []
        for i in range(self.in_dim):
            basis_output = self.basis_functions[i](x[:, i])
            basis_outputs.append(basis_output)
        
        # Combine basis outputs
        combined_output = torch.zeros(batch_size, self.out_dim, device=x.device)
        
        for i in range(self.in_dim):
            basis_output = basis_outputs[i]  # Shape: [batch_size, num_basis]
            weights_i = self.weights[:, i, :]  # Shape: [num_basis, out_dim]
            combined_output += torch.matmul(basis_output, weights_i)
        
        return combined_output + self.bias
    
# Kolmogorov-Arnold Network
class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_basis=10, layers=2):
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(KANLayer(input_dim, hidden_dim, num_basis))
        
        # Hidden layers
        for _ in range(layers - 2):
            self.layers.append(KANLayer(hidden_dim, hidden_dim, num_basis))
        
        # Output layer
        self.layers.append(KANLayer(hidden_dim, output_dim, num_basis))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        # Apply dimensionality reduction for high-dimensional inputs
        if self.input_dim > 100:
            # Simple PCA-like linear projection
            x = x[:, :100]  # Take first 100 dimensions
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)  # Apply ReLU activation
            
        x = self.layers[-1](x)  # Final layer without activation
        return x

# Modified KAN for MNIST
class MNISTKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_basis=10):
        super(MNISTKAN, self).__init__()
        self.input_dim = input_dim
        
        # Use a dimension reduction layer to handle the high-dimensional input
        self.dim_reduction = nn.Linear(input_dim, 32)
        
        # KAN layers
        self.kan_layer1 = KANLayer(32, hidden_dim, num_basis)
        self.kan_layer2 = KANLayer(hidden_dim, output_dim, num_basis)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        
        # Dimension reduction
        x = self.dim_reduction(x)
        x = torch.relu(x)
        
        # KAN layers
        x = self.kan_layer1(x)
        x = torch.relu(x)
        x = self.kan_layer2(x)
        
        return x

# Initialize model
model = MNISTKAN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    num_basis=16
).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(dataloader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# Training loop
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print("-" * 40)
    
    # Save model checkpoint for each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, f'e:/MNIST_Dataset_Work/KAN/checkpoints/kan_model_epoch_{epoch+1}.pt')

# Final evaluation
final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

# Save final model
torch.save(model.state_dict(), 'e:/MNIST_Dataset_Work/KAN/checkpoints/kan_model_final.pt')

# Save metrics to JSON file
metrics = {
    'train_losses': train_losses,
    'train_accuracies': train_accs,
    'test_losses': test_losses,
    'test_accuracies': test_accs,
    'final_test_accuracy': final_test_acc,
    'final_test_loss': final_test_loss,
    'hyperparameters': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'hidden_dim': hidden_dim,
        'num_basis': 16
    }
}

with open('e:/MNIST_Dataset_Work/KAN/results/kan_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(test_accs, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('e:/MNIST_Dataset_Work/KAN/results/kan_training_curves.png')
plt.show()

# Confusion matrix
def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.savefig('e:/MNIST_Dataset_Work/KAN/results/kan_confusion_matrix.png')
    plt.show()

plot_confusion_matrix(model, test_loader, device)

# Visualize B-spline basis functions
def visualize_basis_functions(num_basis=16):
    basis = BSplineBasis(num_basis=num_basis)
    x = torch.linspace(-1, 1, 1000)
    
    plt.figure(figsize=(10, 6))
    
    with torch.no_grad():
        y = basis(x)
        for i in range(num_basis):
            plt.plot(x.numpy(), y[:, i].numpy(), label=f'Basis {i+1}')
    
    plt.title('B-Spline Basis Functions')
    plt.xlabel('Input')
    plt.ylabel('Activation')
    plt.grid(True)
    plt.legend()
    plt.savefig('e:/MNIST_Dataset_Work/KAN/results/basis_functions.png')
    plt.show()

visualize_basis_functions(16)