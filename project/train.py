import argparse
import torch
from dataloader import *
from models import Net
import torch.nn
from torch.utils.data import DataLoader

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', help='path to training dataset')
parser.add_argument('--test_path', help='path to testing dataset')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_train', type=int, help='number of training images', default=750)
parser.add_argument('--num_test', type=int, help='number of validation images', default=250)
parser.add_argument('--weights_file', default="network_weights")
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_epoch', default=80, type=int, help='number of training epochs')
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

        # encode image
        doc = net(img)

        # decode image
        doc = net(doc, encode=False)

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

        # limit to test set
        if idx > args.num_test:
            break

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

    print("===> Loading Data")
    train_data = get_training_set(args.train_path)
    print("===> Constructing DataLoader")

    dataloader = DataLoader(dataset=train_data, num_workers=4, batch_size=args.batch_size, shuffle=True)

    print("Training")
    # train model
    for i in range(args.num_epoch):
        train(dataloader, i)


    # save weights file
    torch.save(net.state_dict(), args.weight_file)


main()
