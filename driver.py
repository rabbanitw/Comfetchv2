import torch
from torchvision import transforms, datasets, models
from argparse import ArgumentParser
from mpi4py import MPI


def load_cifar(train_bs, test_bs):
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    cifar_data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
    cifar_data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar_test)
    num_classes = len(cifar_data_train.class_to_idx.values())

    # add in an iid split method for multiple workers

    trainloader = torch.utils.data.DataLoader(cifar_data_train, batch_size=train_bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(cifar_data_test, batch_size=test_bs, shuffle=False)

    return trainloader, testloader, num_classes


def train(model, optimizer, loss_fn, train_dl, device, epochs, freq):

    # train
    print('Beginning Training')
    for epoch in range(1, epochs+1):

        running_loss = 0.0
        batch_idx = 1
        for inputs, labels in train_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(dim=0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute accuracy
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            total_correct = 0
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    total_correct += 1
                    # correct_pred[classes[label]] += 1
                # total_pred[classes[label]] += 1
            accuracy = total_correct / batch_size

            # print statistics
            running_loss += loss.item()
            if batch_idx % freq == 0:
                running_loss = running_loss / freq
                print('batch %d: accuracy %f, average loss: %f' % (batch_idx, accuracy, running_loss))
                running_loss = 0.0

            batch_idx += 1

    print('Finished Training')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--norm', default="bn")
    parser.add_argument('--partition', default="noniid")
    parser.add_argument('--client_number', default=100)
    parser.add_argument('--alpha_partition', default=0.001)
    parser.add_argument('--commrounds', type=int, default=200)
    parser.add_argument('--clientfr', type=float, default=1.0)
    parser.add_argument('--clientepochs', type=int, default=1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=128)
    parser.add_argument('--clientlr', type=float, default=0.001)
    parser.add_argument('--sch_flag', default=False)
    parser.add_argument('--mixup_prop', type=float, default=0.0)
    parser.add_argument('--natural_img_prop', type=float, default=0.)
    parser.add_argument('--real_prop', type=float, default=0.)
    parser.add_argument('--mix_num', type=int, default=3)
    parser.add_argument('--laplace_scale', type=float, default=50.)
    parser.add_argument('--supplement', type=bool, default=True)
    parser.add_argument('--comp_rate', type=int, default=1)

    args = parser.parse_args()

    # initialize MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # determine torch device available (default to GPU if available)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        dev = "cuda:" + str(gpu_id)
    else:
        dev = "cpu"
    device = torch.device(dev)

    # Hyperparameters_List
    comm_rounds = args.commrounds
    train_bs = args.train_bs
    test_bs = args.test_bs
    learning_rate = args.clientlr
    epochs = args.clientepochs
    batch_freq = 5

    # load data
    train_dl, test_dl, num_classes = load_cifar(train_bs, test_bs)

    # initialize model
    model = models.resnet18()

    # initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss_fn, train_dl, device, epochs, batch_freq)

