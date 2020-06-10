from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class SigmoidLoss(nn.Module):
    def __init__(self, K, alpha, device):
        super(SigmoidLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.K = K
        self.alpha = alpha
        self.M1 = self.K * (self.K - 1) / 2
        self.M2 = self.K - 1

    def forward(self, output, target):
        '''learning from complementary labels'''
        if torch.rand(1) > self.alpha:
            '''make complementary labels'''
            noise = torch.randint(1, self.K, target.shape).to(self.device)
            comp_target = (target + noise) % self.K

            '''caluculate loss'''
            comp_output = output.gather(1, comp_target.long().view(-1, 1)).repeat(1, self.K)
            comp_loss = self.sigmoid(-(output - comp_output))
            pc_loss = torch.sum(comp_loss) * (self.K - 1) / len(target) - self.M1 + self.M2
            return pc_loss
        else:
            loss = self.crossentropy(output, target)
            return loss

class RampLoss(nn.Module):
    def __init__(self, K, alpha, device):
        super(RampLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.device = device
        self.K = K
        self.alpha = alpha
        self.M1 = self.K * (self.K - 1) / 2
        self.M2 = self.K - 1

    def forward(self, output, target):
        '''learning from complementary labels'''
        if torch.rand(1) > self.alpha:
            '''make complementary labels'''
            noise = torch.randint(1, self.K, target.shape).to(self.device)
            comp_target = (target + noise) % self.K

            '''caluculate loss'''
            comp_output = output.gather(1, comp_target.long().view(-1, 1)).repeat(1, self.K)
            comp_loss = torch.max(torch.zeros_like(torch.empty(len(target), self.K)).to(self.device), torch.min(torch.ones_like(torch.empty(len(target), self.K)).to(self.device)*2, 1-(output - comp_output))) / 2
            pc_loss = torch.sum(comp_loss) * (self.K - 1) / len(target) - self.M1 + self.M2
            return pc_loss
        else:
            loss = self.crossentropy(output, target)
            return loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, K, alpha, loss_mode):
    model.train()

    if loss_mode == "sigmoid":
        criterion = SigmoidLoss(K, alpha, device)
    elif loss_mode == "ramp":
        criterion = RampLoss(K, alpha, device)
    else:
        raise ValueError("Invalid loss mode")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example from Complementary Labels')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--alpha', type=float, default=0, help='Rate of ordinary labels (default: 0)')
    parser.add_argument('--loss_mode', help='sigmoid: Sigmoid Loss. ramp: Ramp Loss', choices=['sigmoid', 'ramp'], type=str, default='sigmoid', required=True)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    K = 10 # 10クラス分類

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("loss mode : {}".format(args.loss_mode))
    print("rate of complementary labels : {}".format(1 - args.alpha))

    accuracy_list = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, K, args.alpha, args.loss_mode)
        accuracy = test(model, device, test_loader)
        accuracy_list.append(accuracy)
        scheduler.step()
    
    plt.figure(figsize=(10,6))
    plt.plot(range(args.epochs), accuracy_list)
    plt.xlim(0, args.epochs)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.title('accuracy')
    plt.savefig("accuracy.png")
    print("max accuracy : {}".format(max(accuracy_list)))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()