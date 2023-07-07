import matplotlib.pyplot as plt
import numpy as np
import os


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 1000, low_percentile = 1, high_percentile = 99):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(x, y, num_runs, num_dots, mylegend,ls='-', lw=3, transparency=0.2):
    assert(x.ndim==1)
    assert(x.size==num_dots)
    assert(y.ndim==2)
    assert(y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    plt.plot(x, y_mean, 'o-', label=mylegend, linestyle=ls, linewidth=lw) #, label=r'$\alpha$={}'.format(alpha))
    plt.fill_between(x, y_min, y_max, alpha=transparency)
    return


def unpack_data(directory_path, datatype='test-acc.log', epochs=200, num_workers=10):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    data = np.zeros((epochs, num_workers)) * np.nan
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                if count >= num_workers:
                    break
                j = int(file.split('-')[0][1:])
                with open(directory_path + '/' + file, 'r') as f:
                    i = 0
                    for line in f:
                        data[i, j] = line
                        i += 1
                count += 1
    return data


def data_size(directory_path, datatype='train-acc.log', numworkers=10):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    dataset_sizes = [0 for i in range(numworkers)]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                j = int(file.split('-')[0][1:])
                with open(directory_path + '/' + file, 'r') as f:
                    dataset_sizes[j] = len(f.readlines())
    return dataset_sizes


if __name__ == '__main__':
    colors = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    # number of test runs
    ntest = 3
    # specify number of workers
    num_workers = 10
    folder = 'output/cifar100-noniid/'
    prune_folder = 'output/weight-prune-cifar10/'
    crs = [0.1, 0.25, 0.5, 1]
    labels = ['90% Compression', '75% Compression', '50% Compression', 'No Compression']
    plot_test_acc = True
    plot_train_acc = False
    noniid = True
    prune = False
    dataset = 'cifar100'
    cifar_num_data = 50000
    epochs = 200
    batch_size = 128
    data_per_client = int(cifar_num_data / num_workers)
    total_iterations = int(np.ceil(data_per_client/batch_size) * epochs)


    if not prune:

        if noniid:
            ds = [[] for i in range(len(crs))]
            for trial in range(1, ntest + 1):
                for i, cr in enumerate(crs):
                    if cr < 1:
                        path = folder + 'run' + str(trial) + '-sketch-' + dataset + '-' + str(num_workers) \
                               + 'workers-' + str(cr) + 'cr'
                    else:
                        path = folder + 'run' + str(trial) + '-nosketch-' + dataset + '-' + str(num_workers) \
                               + 'workers-' + str(cr) + 'cr'
                    ds[i].append(data_size(path, numworkers=num_workers))
            total_iterations = np.max(np.array(ds).flatten())

        if plot_test_acc:
            iters = np.arange(1, epochs + 1)
        elif plot_train_acc:
            ones = np.arange(0, 10)
            tens = np.arange(10, 100)[::10]
            hundreds = np.arange(100, 1000)[::100]
            thousands = np.arange(1000, total_iterations)[::1000]
            iters = np.concatenate((ones, tens, hundreds, thousands))
            # iters = np.append(iters, total_iterations-10)

        accs = [[] for i in range(len(crs))]
        for trial in range(1, ntest+1):
            for i, cr in enumerate(crs):
                if cr < 1:
                    path = folder + 'run' + str(trial) + '-sketch-' + dataset + '-' + str(num_workers) + 'workers-' + str(cr) + 'cr'
                else:
                    path = folder + 'run' + str(trial) + '-nosketch-' + dataset + '-' + str(num_workers) + 'workers-' + str(cr) + 'cr'
                if plot_test_acc:
                    data = unpack_data(path, num_workers=10)
                    data = np.nanmean(data, axis=1, keepdims=True)
                elif plot_train_acc:
                    data = unpack_data(path, num_workers=num_workers, epochs=total_iterations, datatype='train-acc.log')
                    data = data[iters, :]
                    data = np.nanmean(data, axis=1, keepdims=True)

                accs[i].append(data)

        plt.figure()
        for i, cr in enumerate(crs):
            stacked_acc = np.hstack(accs[i])
            y_mean, y_min, y_max = generate_confidence_interval(stacked_acc.T)
            plt.plot(iters, y_mean, label=labels[i], color=colors[i])
            plt.fill_between(iters, y_min, y_max, alpha=0.2, color=colors[i])

            print(cr)
            print(y_mean)
            print('=======')

        if plot_test_acc:

            plt.legend(loc='lower right')
            plt.xlim([1, 200])
            plt.ylim([0.0, 0.6])
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.xscale("log")
            plt.grid()
            # plt.show()
            plt.savefig('cifar100-10device-noniid-alpha1.pdf')
            # plt.savefig('cifar100-10device-iid.pdf')

        elif plot_train_acc:
            # plt.legend(loc='lower right')
            plt.legend(loc='upper left')
            plt.xlim([1, total_iterations-1000])
            # plt.xlim([1, 7000])
            plt.ylim([0.0, 1])
            plt.xlabel('Iterations')
            plt.ylabel('Average Train Accuracy Across Devices')
            plt.xscale("log")
            plt.grid()
            plt.show()
            # plt.savefig('cifar100-10device-noniid-alpha1-trainacc.pdf')
            # plt.savefig('cifar100-10device-iid-trainacc.pdf')

    else:
        crs = [0.1, 0.25, 0.5]
        labels = ['90% Weight Prune', '75% Weight Prune', '50% Weight Prune']
        labels_c = ['90% Comfetch', '75% Comfetch', '50% Comfetch']
        colors_c = ['purple', 'black', 'yellow', 'orange']

        if plot_test_acc:
            iters = np.arange(1, epochs + 1)
        elif plot_train_acc:
            ones = np.arange(0, 10)
            tens = np.arange(10, 100)[::10]
            hundreds = np.arange(100, 1000)[::100]
            thousands = np.arange(1000, total_iterations)[::1000]
            iters = np.concatenate((ones, tens, hundreds, thousands))

        accs = [[] for i in range(len(crs))]
        accs_c = [[] for i in range(len(crs))]
        for trial in range(1, ntest + 1):
            for i, cr in enumerate(crs):
                path = prune_folder + 'run' + str(trial) + '-prune-' + dataset + '-' + str(
                    num_workers) + 'workers-' + str(cr) + 'cr'
                path_c = folder + 'run' + str(trial) + '-sketch-' + dataset + '-' + str(
                    num_workers) + 'workers-' + str(cr) + 'cr'
                if plot_test_acc:
                    data = unpack_data(path, num_workers=10)
                    data = np.nanmean(data, axis=1, keepdims=True)
                    data_c = unpack_data(path_c, num_workers=10)
                    data_c = np.nanmean(data_c, axis=1, keepdims=True)
                elif plot_train_acc:
                    data = unpack_data(path, num_workers=num_workers, epochs=total_iterations, datatype='train-acc.log')
                    data = data[iters, :]
                    data = np.nanmean(data, axis=1, keepdims=True)

                    data_c = unpack_data(path_c, num_workers=num_workers, epochs=total_iterations, datatype='train-acc.log')
                    data_c = data_c[iters, :]
                    data_c = np.nanmean(data_c, axis=1, keepdims=True)

                accs[i].append(data)
                accs_c[i].append(data_c)

        plt.figure()
        for i, cr in enumerate(crs):
            stacked_acc = np.hstack(accs[i])
            y_mean, y_min, y_max = generate_confidence_interval(stacked_acc.T)
            plt.plot(iters, y_mean, label=labels[i], color=colors[i])
            plt.fill_between(iters, y_min, y_max, alpha=0.2, color=colors[i])

            print('weighted prune')
            print(cr)
            print(y_mean)
            print('=======')

            #'''
            stacked_acc_c = np.hstack(accs_c[i])
            y_mean_c, y_min_c, y_max_c = generate_confidence_interval(stacked_acc_c.T)
            plt.plot(iters, y_mean_c, label=labels_c[i], color=colors_c[i])
            plt.fill_between(iters, y_min_c, y_max_c, alpha=0.2, color=colors_c[i])

            print(y_mean_c)
            print('=======')
            #'''

        if plot_test_acc:

            plt.legend(loc='lower right')
            plt.xlim([1, 200])
            # plt.ylim([0.1, 0.9])
            plt.ylim([0.0, 0.6])
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            plt.xscale("log")
            plt.grid()
            # plt.show()
            # plt.savefig('cifar100-10device-noniid-alpha1.pdf')
            plt.savefig('cifar100-weight-prune-comparison-iid.pdf')

        elif plot_train_acc:
            # plt.legend(loc='lower right')
            plt.legend(loc='upper left')
            plt.xlim([1, total_iterations - 1000])
            # plt.xlim([1, 7000])
            plt.ylim([0.1, 1])
            plt.xlabel('Iterations')
            plt.ylabel('Average Train Accuracy Across Devices')
            plt.xscale("log")
            plt.grid()
            # plt.show()
            # plt.savefig('cifar100-10device-noniid-alpha1-trainacc.pdf')
            # plt.savefig('cifar100-10device-iid-trainacc.pdf')

