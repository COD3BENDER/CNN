import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(root=".", train=True,
                                              download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root=".", train=False,
                                             download=True, transform=transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
torch.manual_seed(0)
# If you are using CuDNN , otherwise you can just ignore
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# CNN

class LitcCNN(nn.Module):
    def __init__(self):
        super(LitcCNN, self).__init__()
        # Change parameter below to test different Activation Functions.
        activation_function = nn.ReLU()
        # 1 = use dropout, 0 = dont use dropout
        use_dropout = 0
        dropout = 0.5

        self.cnn_model = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5), activation_function, nn.MaxPool2d(2, stride=2), # Convolutional Layers 1 input 1 image out 32 max pooling 2X2,kernal size 5X5
                                       nn.Conv2d(32,64, kernel_size=5), activation_function, nn.MaxPool2d(2, stride=2))# Convolutional Layers 1 input 1 image out 32 max pooling 2X2,kernal size 5X5
        if use_dropout == 1:
            self.fc_model = nn.Sequential(nn.Linear(1024, 1024), activation_function, # FC Layer 1 1024 in 1024 out
                                          nn.Dropout(dropout),nn.Linear(1024, 256), activation_function, #FC Layer 2 1024 in 256 out, dropout applied to the input of second FC layer
                                          nn.Linear(256, 10)) # Output Layer 256 in 10 out,uses cross entropy loss function which has a softmax activation function
        else:
            self.fc_model = nn.Sequential(nn.Linear(1024, 1024), activation_function, # FC Layer 1 1024 in 1024 out
                                          nn.Linear(1024, 256),activation_function,#FC Layer 2 1024 in 256 out
                                          nn.Linear(256, 10))# Output Layer 256 in 10 out,uses cross entropy loss function which has a softmax activation function

        self.initialize_weights()

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight) # using Xavier normal to initialize weights




device = torch.device("cuda:0")

net = LitcCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.1 # modify learning rate
opt = torch.optim.SGD(list(net.parameters()), lr=learning_rate) # SGD optimizer used


def evaluation(dataloader):
    total, correct = 0, 0
    net.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


loss_epoch_array = []
max_epochs = 50
loss_epoch = 0
training_accuracy = []
testing_accuracy = []
for epoch in range(max_epochs):
    loss_epoch = 0
    for i, data in enumerate(training_loader, 0):
        net.train()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        loss_epoch += loss.item()
    loss_epoch_array.append(loss_epoch)
    training_accuracy.append(evaluation(training_loader))
    testing_accuracy.append(evaluation(test_loader))
    print("Epoch {}: Loss: {}, Train accuracy: {}, Test accuracy:{}".format(epoch + 1, loss_epoch_array[-1],
                                                                             training_accuracy[-1], testing_accuracy[-1]))
plt.plot(testing_accuracy, label="Test")
plt.plot(training_accuracy, label="Train")
plt.title('Test and Train Accuracy With Learning Rate={}'.format(learning_rate))
plt.legend()
plt.show()

plt.plot(loss_epoch_array, label="Loss")
plt.title('Loss per epoch With Learning Rate={}'.format(learning_rate))
plt.legend()
plt.show()
