import os
import h5py
import argparse
import numpy as np


def save(data, key):
    out_dir = 'spmot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, '{}_{}'.format('spmot', key))
    print('Save {}'.format(out_path))
    np.save(out_path, data)

# load h5py file
hf = h5py.File(os.path.join('spmot.hdf5'), 'r')
keys = hf.keys()

for key in keys:
    if isinstance(hf[key], h5py.Dataset):
        data = np.array(hf.get(key))
        save(data, key)