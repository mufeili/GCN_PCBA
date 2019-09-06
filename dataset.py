import os
import numpy as np
import pickle
import torch
from dgl.data.chem.utils import smile_to_bigraph
from itertools import accumulate

from utils import mkdir_p

class PCBADataset(object):
    """
    Parameters
    ----------
    file : str
        Path to data file. The first line of the file contains 'smiles task1 task2...'.
        Each rest line contains one datapoint in the form of 'smiles label_for_task1
        label_for_task2...'.
    chunk_size : int
        For working with graph neural networks with DGL, we need to construct DGLGraph
        for each molecule and featurize their atoms and/or bonds. Since the construction
        itself is not cheap, we want to perform graph construction only once and save the
        constructed graphs for later use. Different from fingerprint features, we cannot
        save them in a txt file but need to serialize them with pickle. If the dataset is
        too large, we cannot save them into a single file as they might be too large to
        load into memory at once. For a workaround, we save the graphs into multiple files,
        where each one can contain up to chunk_size graphs.
    """
    def __init__(self, file='data/pcba.txt', chunk_size=10000):
        dir_processed = 'data/processed'
        self.chunk_size = chunk_size
        if os.path.isdir(dir_processed):
            preprocessed = True
        else:
            preprocessed = False
            mkdir_p(dir_processed)
            self.chunk_id = 0
            self.graphs = []

        with open(file, 'r') as f:
            headline = f.readline().strip()
            self.tasks = headline.split('\t')[1:]
            self.n_tasks = len(self.tasks)
            self.num_mols = 0

            while True:
                line = f.readline()
                if not line:
                    break
                self.num_mols += 1
                if not preprocessed:
                    print('Processing molecule {:d}'.format(self.num_mols))
                    self._process_molecule(line)

        if not preprocessed:
            self._dump_chunk()

    def _process_molecule(self, line):
        """
        Parameters
        ----------
        line : str
            SMILES and task labels for a molecule
        """
        elements = line.strip().split('\t')
        smile = elements[0]
        labels = np.array(elements[1:]).astype(np.float32)
        is_nan = np.isnan(labels)
        labels[is_nan] = 0
        labels = torch.from_numpy(labels)
        mask = torch.from_numpy(
            (~is_nan).astype(np.float32))

        g = smile_to_bigraph(smile)
        g.smile = smile
        g.mask = mask
        g.labels = labels
        self.graphs.append(g)
        if len(self.graphs) == self.chunk_size:
            self._dump_chunk()

    def _dump_chunk(self):
        """Save a chunk of molecules."""
        if len(self.graphs) == 0:
            return

        with open('data/processed/chunk_{:d}.pkl'.format(self.chunk_id), 'wb') as f:
            pickle.dump(self.graphs, f)
        self.graphs = []
        self.chunk_id += 1

    def __len__(self):
        """
        Returns
        -------
        int
            Number of molecules in the dataset
        """
        return self.num_mols

    def __getitem__(self, item):
        """Get the datapoint with idx item

        Returns
        -------
        str
            Molecule in SMILES
        DGLGraph
            DGLGraph for the molecule
        float32 torch tensor of shape (1, num_tasks)
            Labels for the molecule
        float32 torch tensor of shape (1, num_tasks)
            Binary mask for indicating the existence of labels
        """
        chunk_id = item // self.chunk_size
        with open('data/processed/chunk_{:d}.pkl'.format(chunk_id), 'rb') as f:
            graphs = pickle.load(f)
            id_in_chunk = item % self.chunk_size
            g = graphs[id_in_chunk]
        return g.smile, g, g.labels, g.mask

class Subset(object):
    """Subset of a dataset at specified indices
    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """
    def __init__(self, dataset, indices, num_tasks=128):
        self.dataset = dataset
        self.indices = indices
        self.num_tasks = num_tasks
        self._task_pos_weights = None

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)

    @property
    def task_pos_weights(self):
        """We perform a weight balancing to address the imbalance
        of positive and negative samples.

        Returns
        -------
        self._task_pos_weights : torch float32 tensor of shape (self.num_tasks,)
            self._task_pos_weights[i] gives the weight on positive samples of task
            i in loss computation
        """
        if self._task_pos_weights is None:
            print('Start computing positive weights...')
            task_total_count = torch.zeros(self.num_tasks)
            task_pos_count = torch.zeros(self.num_tasks)

            for i in self.indices:
                print('Processing training molecule {:d}/{:d}'.format(i+1, len(self)))
                s_i, g_i, label_i, mask_i = self.dataset[i]
                task_total_count += mask_i
                task_pos_count += label_i

            weights = torch.ones(self.num_tasks)
            for t in range(self.num_tasks):
                pos_count_t = float(task_pos_count[t])
                if pos_count_t > 0:
                    weights[t] = (float(task_total_count[t]) - pos_count_t) / pos_count_t
            self._task_pos_weights = weights

        return self._task_pos_weights

def split_dataset(dataset, frac_list=None):
    """
    Parameters
    ----------
    dataset
    frac_list : list of three floats
        The proportion of data to use for training, validation and test.
        If None, we will use [0.8, 0.1, 0.1].

    Returns
    -------
    list
        Consists of three subsets for training, validation and test
    """
    if frac_list is None:
        frac_list = [0.6, 0.2, 0.2]

    frac_list = np.array(frac_list)
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(
            np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    indices = np.arange(num_data)
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]
