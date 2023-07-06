import sys
import numpy as np
from PIL import Image
import os
import shutil


class Recorder(object):
    def __init__(self, folderName, size, rank, args, cr):
        self.record_epoch_times = list()
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_loss = list()
        self.record_training_acc = list()
        self.record_test_acc = list()
        self.rank = rank
        self.saveFolderName = folderName + '/' + args.name + '-' + args.dataset + '-' + str(size)\
                              + 'workers-' + str(cr) + 'cr'

        if rank == 0:
            if not os.path.isdir(self.saveFolderName):
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(args) + '\n')
            else:
                shutil.rmtree(self.saveFolderName)
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(args) + '\n')

    def add_batch_stats(self, comp_time, train_acc, loss):
        self.record_comp_times.append(comp_time)
        self.record_training_acc.append(train_acc)
        self.record_loss.append(loss)

    def add_epoch_stats(self, test_acc, epoch_time, comm_time):
        self.record_test_acc.append(test_acc)
        self.record_comm_times.append(comm_time)
        self.record_epoch_times.append(epoch_time)

    def save_epoch_stats(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc.log', self.record_test_acc,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comm-time.log', self.record_comm_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_epoch_times,
                   delimiter=',')

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comp-time.log', self.record_comp_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.record_loss,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc.log', self.record_training_acc,
                   delimiter=',')


class ProgressBar:
    """
    Prints a progress bar to the standard output (very similar to Keras).
    """

    def __init__(self, width=30):
        """
        Parameters
        ----------
        width : int
            The width of the progress bar (in characters)
        """
        self.width = width

    def update(self, max_value, current_value, info):
        """Updates the progress bar with the given information.

        Parameters
        ----------
        max_value : int
            The maximum value of the progress bar
        current_value : int
            The current value of the progress bar
        info : str
            Additional information that will be displayed to the right of the progress bar
        """
        progress = int(round(self.width * current_value / max_value))
        bar = '=' * progress + '.' * (self.width - progress)
        prefix = '{}/{}'.format(current_value, max_value)

        prefix_max_len = len('{}/{}'.format(max_value, max_value))
        buffer = ' ' * (prefix_max_len - len(prefix))

        sys.stdout.write('\r {} {} [{}] - {}'.format(prefix, buffer, bar, info))
        sys.stdout.flush()

    def new_line(self):
        print()


class Cutout(object):
    """
    Implements Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, num_cutouts, size, p=0.5):
        """
        Parameters
        ----------
        num_cutouts : int
            The number of cutouts
        size : int
            The size of the cutout
        p : float (0 <= p <= 1)
            The probability that a cutout is applied (similar to keep_prob for Dropout)
        """
        self.num_cutouts = num_cutouts
        self.size = size
        self.p = p

    def __call__(self, img):

        height, width = img.size

        cutouts = np.ones((height, width))

        if np.random.uniform() < 1 - self.p:
            return img

        for i in range(self.num_cutouts):
            y_center = np.random.randint(0, height)
            x_center = np.random.randint(0, width)

            y1 = np.clip(y_center - self.size // 2, 0, height)
            y2 = np.clip(y_center + self.size // 2, 0, height)
            x1 = np.clip(x_center - self.size // 2, 0, width)
            x2 = np.clip(x_center + self.size // 2, 0, width)

            cutouts[y1:y2, x1:x2] = 0

        cutouts = np.broadcast_to(cutouts, (3, height, width))
        cutouts = np.moveaxis(cutouts, 0, 2)
        img = np.array(img)
        img = img * cutouts
        return Image.fromarray(img.astype('uint8'), 'RGB')
