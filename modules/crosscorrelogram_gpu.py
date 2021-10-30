import math
import numpy as np
from numba import cuda


@cuda.jit
def computeAutoCC(histogram, spike, Begin=-55, End=55):
    '''
    Gpu version of autocc
    '''
    Start = 0
    N_spike = len(spike)
    n_bin = int(End - Begin)
    thread_idx, thread_idy = cuda.grid(2)

    if thread_idx < N_spike and thread_idy < N_spike:
        delta = spike[thread_idx] - spike[thread_idy]
        if delta > Begin and delta != 0 and (delta-Begin) < n_bin:
            cuda.atomic.add(histogram, int(math.floor(delta-Begin)), 1)

@cuda.jit
def computeCC(histogram, spikeR, spikeT, Begin=-55, End=55):

    n_dim = int(End - Begin)
    Start = 0
    N_spike_x = len(spikeR)
    N_spike_y = len(spikeT)
    thread_idx, thread_idy = cuda.grid(2)

    if thread_idx < N_spike_x and thread_idy < N_spike_y:

        delta = (spikeT[thread_idx] - spikeR[thread_idy]) - Begin
        if delta > 0 and delta < n_dim:
            cuda.atomic.add(histogram, int(math.floor(delta)), 1)
