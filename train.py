import numpy as np
import torch

from model import ResNet50, ResNet34
import data

net = ResNet34(num_classes=275)
net = net.cuda()
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.1, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()
train_loader, test_loader = data.getloader()


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


for epoch in range(0, 10):
    running_loss = 0.0
    for index, (images, labels) in enumerate(train_loader):
        inputs = images.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, running_loss / 300))
            running_loss = 0.0
            test()
