import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from old import model
import util
#from sketchedsgd import SketchedSGD, SketchedSum, SketchedModel
#from error_SGD import ErrorFeedbackSGD

class ResNet18:
    def __init__(self, topk_rate, comp_rate, num_sketches=1, learning_rate=0.001, name="resnet", 
                save_dir="saves", out_dir="out", resume_epoch=None, dataset="cifar10", save_every=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()

        if dataset == "imagenet":
            self.image_dim = 224
            self.classes = 1000
        else:
            self.image_dim = 32
            self.classes = 10

        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1
        self.topk_rate = topk_rate
        self.learn_rate = learning_rate
        self.save_dir = save_dir
        self.out_dir = out_dir
        self.save_every = save_every

        #self.net = model.SketchNet(comp_rate, self.image_dim, self.classes, num_sketches=num_sketches)
        self.net = model.SketchNet18(self.image_dim, self.classes)
        if self.use_cuda:
            self.net = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            #self.net = self.net.cuda()
        # self.net = torch.nn.DataParallel(self.net)
        #self.net = SketchedModel(self.net)
        # self.optimizer = torch.optim.SGD(self.net.module.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        #self.optimizer = ErrorFeedbackSGD(self.net.module.parameters(), self.net.module.sketches, self.topk_rate, lr=learning_rate)

        if resume_epoch is not None:
            self.load_parameters(os.path.join(save_dir, f"{name}_" + str(resume_epoch) + '.pth'))

        self.dataset = dataset
        self.name = name
        self.param_string = f"|topk{topk_rate}|comp{comp_rate}|num-s{num_sketches}|"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)


    def train(self, model, data_loader, num_epochs=75, learning_rate=0.001, batch_size=256, test_each_epoch=False, verbose=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        #batch_size = batch_size * torch.cuda.device_count()
        #if batch_size==0:
        #    batch_size=32
        #self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #self.net.train()
        #resnet18
        model.train()

        #self.optimizer = SketchedSGD(self.optimizer, k=80000, accumulateError=True, p1=0, p2=2)
        #summer = SketchedSum(self.optimizer, self.net, c=240000, r=1, numWorkers=10)

        #if self.dataset == "imagenet":
        #    train_transform = transforms.Compose([
        #        transforms.RandomResizedCrop(self.image_dim),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #    ])
        #    train_dataset = datasets.ImageNet("/fs/cml-datasets/ImageNet/ILSVRC2012", 
        #        split='train', transform=train_transform)
        #else:
        #    train_transform = transforms.Compose([
        #        util.Cutout(num_cutouts=2, size=8, p=0.8),
        #        transforms.RandomCrop(self.image_dim, padding=4),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #    ])
        #    train_dataset = datasets.CIFAR10('data/cifar', 
        #        train=True, download=True, transform=train_transform)
        #print(batch_size)
        #data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()
        e_loss = []
        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            
            train_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                #self.optimizer.zero_grad()
                optimizer.zero_grad()
                #outputs = self.net.forward(images)
                #resnet18
                outputs = model.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                
                #loss = summer(loss)

                loss.backward()
                optimizer.step()
                #self.optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct
                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct / batch_total, epoch_correct / epoch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)
            
            train_loss = train_loss / len(data_loader)
            e_loss.append(train_loss)
            self.train_accuracies.append(epoch_correct / epoch_total)
            if verbose:
                progress_bar.new_line()
            
            print('made it here')
            with open(os.path.join(self.out_dir, self.name + self.param_string + "train.txt"), 'a') as f:
                f.write(f"{self.train_accuracies[-1]}\n")

            if test_each_epoch:
                test_accuracy = self.test()
                self.test_accuracies.append(test_accuracy)
                with open(os.path.join(self.out_dir, self.name + self.param_string + "test.txt"), 'a') as f:
                    f.write(f"{test_accuracy}\n")
                #if verbose:
                print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every epoch
            #if epoch % self.save_every == 0:
            #    self.save_parameters(epoch, directory=self.save_dir, name=self.name)
        total_loss = sum(e_loss) / len(e_loss)  
        #return self.net.state_dict()
        #resnet18
        return model.state_dict(), total_loss
        
    def test(self, batch_size=None):
        """Tests the network.

        """
        self.net.eval()
        if self.dataset == "imagenet":
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            test_dataset = datasets.ImageNet("/fs/cml-datasets/ImageNet/ILSVRC2012",
                split='val', transform=test_transform)
            if batch_size is None:
                batch_size = 64
        else:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_dataset = datasets.CIFAR10('data/cifar', 
                train=False, download=True, transform=test_transform)
            if batch_size is None:
                batch_size = 256

        batch_size = batch_size * torch.cuda.device_count()
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()
        test_acc = correct/total
        with open(os.path.join(self.out_dir, self.name + self.param_string + "test.txt"), 'a') as f:
                    f.write(f"{test_acc}\n")

        #self.net.train()
        return correct / total

    def save_parameters(self, epoch, directory, name):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }, os.path.join(directory, f"{name}_" + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.start_epoch = checkpoint['epoch']

