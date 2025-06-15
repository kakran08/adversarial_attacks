import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary

class Model(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), cnn=True):
        super(Model, self).__init__()
        self.shape = input_shape
        self.cnn = cnn
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        self.out = nn.Linear(in_features= 100, out_features=10)

        if cnn:
            self.cnn_1_16 = nn.Conv2d(1, 16, 2)
            self.cnn_16_32 = nn.Conv2d(16, 32, 2)
            self.cnn_32_64 = nn.Conv2d(32, 64, 2)
            self.pool = nn.MaxPool2d(2)
            self.fcl_1000 = nn.Linear(in_features=2304, out_features=1000)
            self.fcl_100 = nn.Linear(in_features=1000, out_features=100)
        else:
            self.fcl_1000 = nn.Linear(in_features=784, out_features=1000)
            self.fcl_100 = nn.Linear(in_features=1000, out_features=100)

    def forward(self,x):
        if self.cnn:
            x = F.relu(self.cnn_1_16(x))
            x = self.pool(F.relu(self.cnn_16_32(x)))
            x = self.pool(F.relu(self.cnn_32_64(x)))
            x = F.relu(self.flat(x))
            x = F.relu(self.fcl_1000(x))
            x = F.relu(self.fcl_100(x))
        else:
            x = F.relu(self.flat(x))
            x = F.relu(self.fcl_1000(x))
            x = F.relu(self.fcl_100(x))
        x = self.out(x)
        return x
    
def model_summary(model, input_size=(1,28,28)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(summary(model, input_size))

def load_mnist():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=64)
    return train_loader, test_loader


def train(model,train_loader, loss_fn, optim, epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loss = 0
    train_correct = 0

    for i in range(epochs):   

        epoch_loss = 0
        epoch_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            batch_loss = loss_fn(y_pred, y)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            epoch_loss += batch_loss.item()

            y_pred = torch.argmax(y_pred, 1) 
            epoch_correct += (y_pred == y).sum().item()

        train_loss += epoch_loss/len(train_loader)
        train_correct += epoch_correct

        print(f"Epoch: {i}, Loss: {epoch_loss/len(train_loader)}, Accuracy: {epoch_correct/(len(train_loader)*train_loader.batch_size)}")

    print(f"Training complete. Training loss: {train_loss/epochs}, Training accuracy: {train_correct/(epochs*len(train_loader)*train_loader.batch_size)}")
    
    return model, {"loss":train_loss/epochs, "acc":train_correct/(epochs*len(train_loader)*train_loader.batch_size)}

def test(model, test_loader, loss_fn=nn.CrossEntropyLoss()):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            y_pred = torch.argmax(y_pred, 1) 
            correct += (y_pred == y).sum().item()

    print(f"Testing complete. Testing loss: {total_loss/len(test_loader)}, Testing accuracy: {correct/(len(test_loader)*test_loader.batch_size)}")

def save(model):
    torch.save(model.state_dict(),'model.pth')


if __name__ == "__main__":
    
    model = Model()
    model_summary(model)
    train_loader, test_loader = load_mnist()
    model, state = train(model, train_loader, loss_fn=nn.CrossEntropyLoss(), optim=torch.optim.Adam(model.parameters()), epochs=10)
    test(model, test_loader, loss_fn=nn.CrossEntropyLoss())
    save(model)