import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassCalibrationError

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)


transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))

])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),   # 1 input channel, 6 output channels, filter size 5x5
            nn.ReLU(),
            nn.MaxPool2d(2),      # 2x2 max pooling
            nn.Conv2d(6, 16, 5),  # 16 output channels, kernel 5x5
            nn.ReLU(),
            nn.MaxPool2d(2)       # 2x2 max pooling
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


modelA = Net().to(device)
modelB = Net().to(device)

criterionA = nn.CrossEntropyLoss(label_smoothing=0)     
criterionB = nn.CrossEntropyLoss(label_smoothing=0.05)     

optA = optim.Adam(modelA.parameters(), lr=1e-3)
optB = optim.Adam(modelB.parameters(), lr=1e-3)


f1_metric = MulticlassF1Score(num_classes=10).to(device)
ece_metric = MulticlassCalibrationError(num_classes=10, n_bins=10).to(device)



def train(model, loader, opt, criterion):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * y.size(0) #average loss per batch size*batch size=total loss of batch
        _, preds = out.max(1)
        correct += preds.eq(y).sum().item()      
        total += y.size(0) 

    train_acc = correct / total
    train_err = 1 - train_acc
    avg_loss = running_loss / total

    return train_acc, train_err, avg_loss


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            # accuracy
            _, preds = out.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

            preds_all.append(preds)
            labels_all.append(y)
            probs_all.append(out.softmax(1))

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    probs_all = torch.cat(probs_all)

    accuracy = correct / total
    error = 1 - accuracy
    f1 = f1_metric(preds_all, labels_all).item()
    ece = ece_metric(probs_all, labels_all).item()

    return accuracy, error, f1, ece


EPOCHS = 10

for epoch in range(1, EPOCHS+1):
    print(f"\n===== Epoch {epoch} =====")

    # Train both models
    taccA, terrA, tlossA = train(modelA, trainloader, optA, criterionA)
    taccB, terrB, tlossB = train(modelB, trainloader, optB, criterionB)

    # Test both models
    accA, errA, f1A, eceA = test(modelA, testloader)
    accB, errB, f1B, eceB = test(modelB, testloader)

    print("\n--- Model A: Smooth=0 ---")
    print(f"Train Acc:{taccA:.4f}  Train Err:{terrA:.4f}  Loss:{tlossA:.4f}")
    print(f"Test  Acc:{accA:.4f}  Test  Err:{errA:.4f}")
    print(f"F1 Score:{f1A:.4f}   ECE:{eceA:.4f}")

    print("\n--- Model B: Smooth=0.05 ---")
    print(f"Train Acc:{taccB:.4f}  Train Err:{terrB:.4f}  Loss:{tlossB:.4f}")
    print(f"Test  Acc:{accB:.4f}  Test  Err:{errB:.4f}")
    print(f"F1 Score:{f1B:.4f}   ECE:{eceB:.4f}")
