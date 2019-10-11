import argparse
import sys
import torch
from dataloader import Raise1K
from models import Net
import torch.nn

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', help='path to training dataset')
parser.add_argument('--test_path', help='path to testing dataset')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--num_train', type=int, help='number of training images', default=750)
parser.add_argument('--num_test', type=int, help='number of validation images', default=250)
# parser.add_argument('--test_indices_path', help='paths to val indices')
# parser.add_argument('--train_indices_path', help='paths to train indices')
parser.add_argument('--weights_file', default="network_weights")
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_epoch', default=80, type=int, help='number of traing epochs')
args = parser.parse_args()

# run model on multiple GPU's
net = torch.nn.DataParallel(Net().cuda())

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
best_loss = torch.tensor(100.0).cuda()


def train(dataloader, epoch):
    """
    :param dataloader:
    :param epoch:
    :return:
    """

    for idx, data in enumerate(dataloader):
        img = data
        img = img.cuda()

        optimizer.zero_grad()
        doc = net(img)
        loss = torch.mean(torch.abs(img - doc))
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(f"Epoch: {epoch} Training Loss: {loss.item()}")


def test(dataloader, epoch):

    """
    :param dataloader:
    :param epoch:
    :return:
    """

    net.eval()
    loss = 0

    for idx, data in enumerate(dataloader):

        img = data

        with torch.no_grad():
            img = img.cuda()
            doc = net(img, isTest=True)

            # compute validation L1 loss in 0, 1 space
            loss += torch.mean(torch.abs(doc - img))

    loss /= len(dataloader)
    print("Epoch: ", epoch, " Testing Loss: ", loss.item())

    global best_loss
    if loss < best_loss:
        best_loss = loss.item()

    net.train()


def main():

    data = Raise1K(args.train_path, args.test_path)
    train_loader, val_loader = data.data_loader(8, args.batch_size, args.num_train, args.num_test, args.train_path, args.test_path)

    # train model
    for i in range(args.num_epoch):
        train(train_loader, i)


    # save weights file
    torch.save(net.state_dict(), args.weight_file)


main()
