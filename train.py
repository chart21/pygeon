import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_and_evaluate(model, train_loader, test_loader, num_epochs=20, optimizer_class=optim.Adam, learning_rate=0.001, criterion_class=nn.CrossEntropyLoss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training the model on {device} ...")

    # Define the loss and the optimizer
    criterion = criterion_class()
    optimizer = optimizer_class(params=model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
    return accuracy


def evaluate(model, test_loader, criterion_class=nn.CrossEntropyLoss):
    model.eval()
    correct = 0
    total = 0
    criterion = criterion_class()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            print("label: ", labels, "predicted: ", predicted)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")

def train_test(model, train_loader, test_loader, num_epochs=20, optimizer_class=optim.Adam, learning_rate=0.001, criterion_class=nn.CrossEntropyLoss, model_name="model", dataset_name="dataset", transform="standard"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training {model_name} on {dataset_name} with {transform} transform ({device}) ..." )

    # Define the loss and the optimizer
    criterion = criterion_class()
    optimizer = optimizer_class(params=model.parameters(), lr=learning_rate)
    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            print(f"Accuracy improved, saving model")
            best_accuracy = accuracy
            torch.save(model.state_dict(), "./models/" + model_name + "_" + dataset_name + "_" + transform + "_best.pth")
            with open("./models/" + model_name + "_" + dataset_name + "_" + transform + "_best.txt", "w") as f:
                f.write(f"Epoch: {epoch + 1}/{num_epochs}\n")
                f.write(f"Best accuracy: {best_accuracy:.2f}%")

