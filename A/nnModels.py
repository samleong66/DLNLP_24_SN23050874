import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from A.polClassification_ML import getFeaturesAndPolsFromFile

label_mapping = {
    'conflict': 0,
    'negative': 1,
    'neutral': 2,
    'positive': 3
}

def create_dataloader(X, y, model, batch_size=32):
    if type(X) != torch.tensor:
        if model == 'cnn':
            X = X.reshape(-1, 1, 20, 30)
            X = torch.tensor(X, dtype=torch.float32)
        elif model == 'lstm':
            X = X.reshape(-1, 1, 600)
            X = torch.tensor(X, dtype=torch.float32)
    try:
        y = torch.tensor([label_mapping[label] for label in y], dtype=torch.long)
    except:
        print("Fail to transform labels data")

    dataset = TensorDataset(X, y)

    return DataLoader(dataset, batch_size, shuffle=True)


class CNNforABSA(nn.Module):
    def __init__(self):
        super(CNNforABSA, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (16, 10, 15)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (32, 5, 7)
        
        # Fully connected layer
        self.fc1 = nn.Linear(32 * 5 * 7, 120)  # Input features are 32 channels, 5 height, 7 width
        self.fc2 = nn.Linear(120, 10)  # Example output for 10 classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 32 * 5 * 7)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class LSTMforABSA(nn.Module):
    def __init__(self):
        super(LSTMforABSA, self).__init__()
        self.hidden_size = 128
        self.num_layers = 4
        self.lstm = nn.LSTM(600, 128, 4, batch_first=True)
        self.fc = nn.Linear(128, 4)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out


def trainNNforABSA(model, dataloader, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, data in progress_bar:
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            losses.append(loss.item())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

    return model, loss

def evaluate_model(model, test_dataloader, device):
    # Switch model to evaluation mode
    model.eval()

    # Initialize metrics
    correct = 0
    total = 0

    # Disable gradient computation
    with torch.no_grad():
        for data in test_dataloader:
            inputs, label = data
            # Send data to the correct device
            inputs = inputs.to(device)
            label = label.to(device)

            # Forward pass to get outputs
            outputs = model(inputs)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += label.size(0)

            # Total correct predictions
            correct += (predicted == label).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy



def evaluate_model_per_label(model, test_dataloader, device):
    # Switch model to evaluation mode
    model.eval()

    # Initialize dictionaries to store correct predictions and total counts for each label
    correct_pred = {label: 0 for label in range(4)}
    total_pred = {label: 0 for label in range(4)}

    # Disable gradient computation
    with torch.no_grad():
        for data in test_dataloader:
            inputs, label = data
            # Send data to the correct device
            inputs = inputs.to(device)
            label = label.to(device)

            # Forward pass to get outputs
            outputs = model(inputs)

            # Get predictions from the maximum value
            _, predictions = torch.max(outputs, 1)

            # Collect the correct predictions for each class
            for label, prediction in zip(label, predictions):
                if label == prediction:
                    correct_pred[label.item()] += 1
                total_pred[label.item()] += 1

    # Calculate accuracy for each label
    label_accuracy = {label: 100 * correct_pred[label] / total_pred[label] if total_pred[label] > 0 else 0 for label in range(4)}
    return label_accuracy



def examByNN(d_type: str, model_name: str, per=0.8, epochs=100):
    model_path = 'B/model/%s_clf.pth'%model_name
    if d_type=='re':
        filepath='B/contextFiles/re_train.cox'
    else:
        filepath='B/contextFiles/lp_train.cox'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name=='cnn':
        model = CNNforABSA().to(device)
    elif model_name == 'lstm':
        model = LSTMforABSA().to(device)

    trainX,trainY,testX,testY=getFeaturesAndPolsFromFile(filepath,d_type,per)
    
    train_dataset = create_dataloader(trainX, trainY, model_name)
    test_dataset = create_dataloader(testX, testY, model_name)
    if not os.path.exists(model_path):
        model, loss = trainNNforABSA(model, train_dataset, device, epochs)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))

    gross_acc = evaluate_model(model, test_dataset, device)
    label_acc = evaluate_model_per_label(model, test_dataset, device)

    print("Classification Results:")
    print("Gross Accuracy: %.4f%%"%gross_acc)
    for label in label_acc:
        print(f'Accuracy for label {list(label_mapping.keys())[label]}: {label_acc[label]:.2f}%')





    

    