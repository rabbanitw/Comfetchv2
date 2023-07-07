import torch
from torchvision import transforms, datasets, models
from argparse import ArgumentParser
from mpi4py import MPI
from resnet import ResNet
from communicator_sd import Communicator
import time
import numpy as np
from util import Recorder
from torch.nn.utils import prune


def load_cifar(rank, size, train_bs, test_bs, cifar10=True):
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    if cifar10:
        cifar_data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
        cifar_data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar_test)
    else:
        cifar_data_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms_cifar_train)
        cifar_data_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms_cifar_test)
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


def load_cifar_noniid(rank, size, train_bs, test_bs, alpha=0.1, cifar10=True):
    # create transforms
    # We will just convert to tensor and normalize since no special transforms are mentioned in the paper
    stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    transforms_cifar_train = transforms.Compose([transforms.ToTensor(),
                                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.Normalize(*stats)])
    transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    if cifar10:
        cifar_data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
        cifar_data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar_test)
    else:
        cifar_data_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms_cifar_train)
        cifar_data_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms_cifar_test)
    num_classes = len(cifar_data_train.class_to_idx.values())

    # split data evently amongst devices (first shuffle to ensure iid)
    num_data = cifar_data_train.data.shape[0]
    num_test_data = cifar_data_test.data.shape[0]

    # dirichlet split
    if rank == 0:
        min_size = 0
        labels = np.array(cifar_data_train.targets)
        dataidx_map = {}
        while min_size < 10:
            idx_batch = [[] for _ in range(size)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, size))
                # Balance
                proportions = np.array([p * (len(idx_j) < num_data / size) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(size):
            dataidx_map[j] = idx_batch[j]
    else:
        dataidx_map = None

    dataidx_map = MPI.COMM_WORLD.bcast(dataidx_map, root=0)
    cifar_data_train.data = cifar_data_train.data[dataidx_map[rank], :, :, :]
    cifar_data_train.targets = np.array(cifar_data_train.targets)[dataidx_map[rank]]

    # load data into dataloader
    trainloader = torch.utils.data.DataLoader(cifar_data_train, batch_size=train_bs, shuffle=True, drop_last=True)
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
        model.train()
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


        # print(model.conv1.weight)
        
        # compute test accuracy
        model.eval()
        total_correct = 0
        model.eval()
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
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--alpha_partition', default=1.0)
    parser.add_argument('--clientfr', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=1024)
    parser.add_argument('--clientlr', type=float, default=0.001)
    parser.add_argument('--sketch', type=int, default=0)
    parser.add_argument('--iid', type=int, default=1)
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
    train_bs = args.train_bs
    test_bs = args.test_bs
    learning_rate = args.clientlr
    epochs = args.epochs
    cr = args.cr
    batch_freq = 20
    resnet_size = 18
    alpha = args.alpha_partition
    sketch = bool(args.sketch)
    iid = bool(args.iid)
    same_client_sketch = bool(args.same_client_sketch)
    if args.dataset == 'cifar10':
        cifar10 = True
    else:
        cifar10 = False

    # load data (iid or non-iid)
    if iid:
        train_dl, test_dl, num_classes, num_test_data = load_cifar(rank, size, train_bs, test_bs, cifar10=cifar10)
    else:
        train_dl, test_dl, num_classes, num_test_data = load_cifar_noniid(rank, size, train_bs, test_bs, alpha=alpha,
                                                                          cifar10=cifar10)

    # initialize communicator
    Comm = Communicator(rank, size, comm, device)

    # initialize model
    model = ResNet(rank, resnet_size, num_classes, cr=cr, sketch=False, device=device)

    '''
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.layer1[0].conv1, 'weight'),
        (model.layer1[0].conv2, 'weight'),
        (model.layer1[1].conv1, 'weight'),
        (model.layer1[1].conv2, 'weight'),
        (model.layer2[0].conv1, 'weight'),
        (model.layer2[0].conv2, 'weight'),
        (model.layer2[0].shortcut[0], 'weight'),
        (model.layer2[1].conv1, 'weight'),
        (model.layer2[1].conv2, 'weight'),
        (model.layer3[0].conv1, 'weight'),
        (model.layer3[0].conv2, 'weight'),
        (model.layer3[0].shortcut[0], 'weight'),
        (model.layer3[1].conv1, 'weight'),
        (model.layer3[1].conv2, 'weight'),
        (model.layer4[0].conv1, 'weight'),
        (model.layer4[0].conv2, 'weight'),
        (model.layer4[0].shortcut[0], 'weight'),
        (model.layer4[1].conv1, 'weight'),
        (model.layer4[1].conv2, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=cr,
    )
    '''

    #'''
    comp = 1-cr
    prune.random_unstructured(model.conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer1[0].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer1[0].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer1[1].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer1[1].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer2[0].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer2[0].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer2[1].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer2[1].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer3[0].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer3[0].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer3[1].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer3[1].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer4[0].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer4[0].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer4[1].conv1, name="weight", amount=comp)
    prune.random_unstructured(model.layer4[1].conv2, name="weight", amount=comp)
    prune.random_unstructured(model.layer2[0].shortcut[0], name="weight", amount=comp)
    prune.random_unstructured(model.layer3[0].shortcut[0], name="weight", amount=comp)
    prune.random_unstructured(model.layer4[0].shortcut[0], name="weight", amount=comp)
    #'''

    model.to(device)

    '''
    prune.remove(model.conv1, 'weight')
    prune.remove(model.layer1[0].conv1, 'weight')
    prune.remove(model.layer1[0].conv2, "weight")
    prune.remove(model.layer1[1].conv1, "weight")
    prune.remove(model.layer1[1].conv2, "weight")
    prune.remove(model.layer2[0].conv1, "weight")
    prune.remove(model.layer2[0].conv2, "weight")
    prune.remove(model.layer2[1].conv1, "weight")
    prune.remove(model.layer2[1].conv2, "weight")
    prune.remove(model.layer3[0].conv1, "weight")
    prune.remove(model.layer3[0].conv2, "weight")
    prune.remove(model.layer3[1].conv1, "weight")
    prune.remove(model.layer3[1].conv2, "weight")
    prune.remove(model.layer4[0].conv1, "weight")
    prune.remove(model.layer4[0].conv2, "weight")
    prune.remove(model.layer4[1].conv1, "weight")
    prune.remove(model.layer4[1].conv2, "weight")
    prune.remove(model.layer2[0].shortcut[0], "weight")
    prune.remove(model.layer3[0].shortcut[0], "weight")
    prune.remove(model.layer4[0].shortcut[0], "weight")
    '''

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    
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
