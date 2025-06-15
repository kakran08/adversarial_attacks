import torch
import torch.nn as nn
from model import Model, load_mnist, model_summary

def fgsm(model, loss_fn, images, labels, epsilon):

    images = images.clone().detach()
    images.requires_grad_(True)

    outputs = model(images)
    loss = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    image_grad = images.grad.data

    perturbed_images = images + epsilon*image_grad.sign()

    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images


def fgsm_attack(model, data_loader, loss_fn, epsilon = 0.1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    
    total_loss = 0
    correct = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        x_pert = fgsm(model,loss_fn, x, y, epsilon)
        y_pred = model(x_pert)

        loss = loss_fn(y_pred, y)
        total_loss += loss.item()

        y_pred = torch.argmax(y_pred, 1) 
        correct += (y_pred != y).sum().item()

    print(f"FGSM atack: Complete, Loss: {total_loss/len(data_loader)}, Accuracy: {correct/(len(data_loader)*data_loader.batch_size)}")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_mnist()

    model = Model().to(device)
    model.load_state_dict(torch.load('model.pth'))
    print(model_summary(model, input_size=(1,28,28)))

    fgsm_attack(model, test_loader, loss_fn = nn.CrossEntropyLoss(), epsilon=0.9)
