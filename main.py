import torch
from torchvision import transforms, datasets, models
from argparse import ArgumentParser
from mpi4py import MPI
from resnet import ResNet
from communicator_sd import Communicator
import time
import numpy as np
from util import Recorder


def load_cifar(rank, size, train_bs, test_bs):
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

    # split data evently amongst devices (first shuffle to ensure iid)
    num_data = cifar_data_train.data.shape[0]
    num_test_data = cifar_data_test.data.shape[0]
    if rank == 0:
        shuffle_idx = np.arange(num_data, dtype=np.int32)
        np.random.shuffle(shuffle_idx)
    else:
        shuffle_idx = np.zeros(num_data, dtype=np.int32)

    MPI.COMM_WORLD.Bcast(shuffle_idx, root=0)
    shuffle_idx = np.array_split(shuffle_idx, size)[rank]

    cifar_data_train.data = cifar_data_train.data[shuffle_idx, :, :, :]
    cifar_data_train.targets = np.array(cifar_data_train.targets)[shuffle_idx]

    # load data into dataloader
    trainloader = torch.utils.data.DataLoader(cifar_data_train, batch_size=train_bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(cifar_data_test, batch_size=test_bs, shuffle=False)

    return trainloader, testloader, num_classes, num_test_data


def train(rank, model, Comm, optimizer, loss_fn, train_dl, test_dl, recorder, device, epochs, freq, num_test_data):

    # train
    model.train()
    if rank == 0:
        print('Beginning Training')

    for epoch in range(1, epochs+1):

        if rank == 0:
            print('Starting Epoch %d' % epoch)
        running_loss = 0.0
        batch_idx = 1
        epoch_time = 0.0
        for inputs, labels in train_dl:

            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(dim=0)

            t = time.time()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_time = time.time()-t

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
            loss_val = loss.item()
            recorder.add_batch_stats(batch_time, accuracy, loss.detach().cpu().numpy())

            # print statistics
            running_loss += loss_val
            if batch_idx % freq == 0:
                running_loss = running_loss / freq
                print('rank %d, batch %d: accuracy %f, average loss: %f, time: %f' % (rank, batch_idx, accuracy,
                                                                                      running_loss, batch_time))
                running_loss = 0.0
                recorder.save_to_file()

            batch_idx += 1
            epoch_time += batch_time

        recorder.save_to_file()

        # perform federated averaging after every epoch
        comm_time = Comm.communicate(model)

        # compute test accuracy
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                labels = labels.to(device)
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)

                # compute accuracy
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        total_correct += 1
        test_accuracy = total_correct / num_test_data
        print('     rank %d, epoch %d: test accuracy %f' % (rank, epoch, test_accuracy))
        recorder.add_epoch_stats(test_accuracy, epoch_time, comm_time)
        recorder.save_epoch_stats()

        MPI.COMM_WORLD.Barrier()

    if rank == 0:
        print('Finished Training')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--norm', default="bn")
    parser.add_argument('--partition', default="noniid")
    parser.add_argument('--alpha_partition', default=0.001)
    parser.add_argument('--commrounds', type=int, default=200)
    parser.add_argument('--clientfr', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=1024)
    parser.add_argument('--clientlr', type=float, default=0.001)
    parser.add_argument('--sketch', type=int, default=1)
    parser.add_argument('--same_client_sketch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--cr', type=float, default=0.5)
    parser.add_argument('--name', type=str, default='test')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    epochs = args.epochs
    cr = args.cr
    batch_freq = 20
    resnet_size = 18
    sketch = bool(args.sketch)
    same_client_sketch = bool(args.same_client_sketch)
    if not sketch:
        cr = 1

    # load data
    train_dl, test_dl, num_classes, num_test_data = load_cifar(rank, size, train_bs, test_bs)

    # initialize communicator
    Comm = Communicator(rank, size, comm, device)

    # initialize model
    # model = models.resnet18()
    model = ResNet(rank, resnet_size, num_classes, cr=cr, sketch=sketch, device=device, same_sketch=same_client_sketch)
    model.to(device)

    # synchronize model amongst all devices
    Comm.sync_models(model)

    # initialize recorder
    recorder = Recorder('output', size, rank, args, cr)

    # initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    MPI.COMM_WORLD.Barrier()

    train(rank, model, Comm, optimizer, loss_fn, train_dl, test_dl, recorder, device, epochs, batch_freq, num_test_data)

