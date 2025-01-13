"""
This module provides utility functions and classes for working with PyTorch
workflows in UF3.
"""
from typing import List, Collection, Callable
import torch
import pandas as pd
from uf3.representation import process

class HDF5Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        table_names (list): A list of table names to load from the HDF5 file.
        subset (Collection): A subset of keys to load from the HDF5 file.
        load_fn (Callable): A function to load data from the HDF5 file.
        **load_fn_kwargs: Additional keyword arguments to pass to the load function
            when loading data.
    Returns:
        A PyTorch Dataset object.
    """
    def __init__(self,
                 filename: str,
                 table_names: List[str],
                 subset: Collection = None,
                 load_fn: Callable = process.load_feature_db,
                 **load_fn_kwargs,
                 ):
        self.filename = filename
        self.table_names = table_names
        self.subset = subset
        self.load_fn = load_fn
        self.load_fn_kwargs = load_fn_kwargs

    def __len__(self):
        return len(self.table_names)

    def __getitem__(self,
                    idx: int):
        table_name = self.table_names[idx]
        #df = process.load_feature_db(self.filename, table_name, sparse_hdf5=self.sparse_hdf5)
        df = self.load_fn(self.filename, table_name, **self.load_fn_kwargs)
        if self.subset is not None:
            keys = df.index.unique(level=0).intersection(self.subset)
            if len(keys) == 0:
                return None  # Skip if no keys found
        return df


def hdf5_dataloader(filename: str,
                    table_names: List[str],
                    subset: Collection = None,
                    batch_size: int = 1,
                    shuffle: bool = False,
                    drop_last: bool = False,
                    num_workers: int = 0,
                    load_fn: Callable = process.load_feature_db,
                    **load_fn_kwargs,
                    ) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for loading data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        table_names (list): A list of table names to load from the HDF5 file.
        subset (Collection): A subset of keys to load from the HDF5 file.
        batch_size (int): The batch size.
        sparse_hdf5 (bool): Whether the HDF5 features are stored as sparse matrices.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last batch if it is smaller
            than the batch size.
        num_workers (int): The number of workers to use for loading data.
        load_fn (Callable): A function to load data from the HDF5 file.
        **load_fn_kwargs: Additional keyword arguments to pass to the load function
            when loading data.

    Returns:
        A PyTorch DataLoader object.
    """
    def collate_fn(batch):
        return [item for item in batch if item is not None]
    dataset = HDF5Dataset(filename, table_names, subset, load_fn, **load_fn_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             drop_last=drop_last,
                                             collate_fn=collate_fn)
    return dataloader