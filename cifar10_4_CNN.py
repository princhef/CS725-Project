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
    transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))

])

#loading the dataset ; CIFAR has 32x32 color images with 3 channels
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2)



#4_layer_CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            #layer1
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),  #(depth of current layer,depth of next layer,size of filter)
            #layer2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),   
            nn.MaxPool2d(2),                             #reduce size of layer by 2

            #layer3
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),

            #layer4
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 512), nn.ReLU(),      #(input layer no of nodes,output layer no of nodes)
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)                #(batch size,infer remaining dimension automatically)
        return self.fc(x)

# Creating 2 models 
modelA = Net().to(device)
modelB = Net().to(device)

#setting the crossentropyloss
criterionA = nn.CrossEntropyLoss(label_smoothing=0)     
criterionB = nn.CrossEntropyLoss(label_smoothing=0.05)     

#setting the adam optimizer
optA = optim.Adam(modelA.parameters(), lr=1e-3)
optB = optim.Adam(modelB.parameters(), lr=1e-3)

#Setting the metrics for evaluation
f1_metric = MulticlassF1Score(num_classes=10).to(device)
ece_metric = MulticlassCalibrationError(num_classes=10, n_bins=10).to(device)



# Train function
def train(model, loader, opt, criterion):
    model.train()     #setting the model to train mode
    correct = 0
    total = 0
    running_loss = 0

    for x, y in loader:          #loading (input,output) these will be according to batch size
        x, y = x.to(device), y.to(device)   #sending the input & output to CPU or GPU

        opt.zero_grad()           #resetting the gradient calculated in previous step
        out = model(x)            #forward pass
        loss = criterion(out, y)   #calculating loss (actual output, expected output)
        loss.backward()            #backward pass
        opt.step()                  #updating the values of parameters

        running_loss += loss.item() * y.size(0)   #(max value,index of max value)
        _, preds = out.max(1)                      
        correct += preds.eq(y).sum().item()       #finds if equal to expected output then it will be 1 and add to sum
        total += y.size(0)                       #adds the total number of samples
 
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


EPOCHS = 50

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
