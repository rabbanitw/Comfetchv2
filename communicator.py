import torch
import numpy as np
import time
from mpi4py import MPI


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


class Communicator:

    def __init__(self, size, comm, device):
        self.comm = comm
        self.size = size
        self.device = device
        self.tensor_list = list()
        self.send_buffer = None
        self.recv_buffer = None

    def average(self):

        self.comm.Barrier()
        tic = time.time()

        self.comm.Allreduce(self.send_buffer, self.recv_buffer, op=MPI.SUM)

        self.comm.Barrier()
        toc = time.time()

        return toc - tic

    def sync_models(self, model):

        # prepare model to be communicated
        self.prepare(model)

        # averaging across all devices
        _ = self.average()

        # uniform averaging amongst all clients
        self.recv_buffer = torch.from_numpy(self.recv_buffer / self.size)

        # reset local models to be the averaged model
        self.reset_model()

    def prepare(self, model):

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param)

        # necessary preprocess (flatten tensors)
        self.send_buffer = flatten_tensors(self.tensor_list).cpu().detach().numpy()
        self.recv_buffer = np.zeros_like(self.send_buffer)

    def reset_model(self):
        uft = unflatten_tensors(self.recv_buffer, self.tensor_list)
        for f, t in zip(uft, self.tensor_list):
            t = t.to(self.device)
            with torch.no_grad():
                t.set_(f)

    def communicate(self, model):

        # prepare model to be communicated
        self.prepare(model)

        # averaging across all devices
        comm_time = self.average()

        # uniform averaging amongst all clients
        self.recv_buffer = torch.from_numpy(self.recv_buffer / self.size)

        # reset local models to be the averaged model
        self.reset_model()

        return comm_time
