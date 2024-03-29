# GCN on PCBA

This is an example for multi-label molecule classification with DGL on a large dataset. We assume the dataset to be too 
large to be loaded into memory at once.

## Dependencies

1. PyTorch
2. DGL
3. Scikit-learn
4. RDKit

## Dataset and Task

[PCBA](http://moleculenet.ai/datasets-1) is a subset from PubChem BioAssay and consists of measured biological 
activities of small molecules generated by high-throughput screening. It contains 437929 molecules for 128 binary 
classification tasks. We use a subset of 170000 molecules.

## Example Usage

`python main.py` with options

```
-m {GCN,GAT}, Model to use
-c CHUNK_SIZE, Number of preprocessed molecules in each pickle file. Default to be 1.
-n NUM_EPOCHS, Max number of epochs to train the model. Default to be 100.
```

## Reference Numbers

With early stopping, the training of GCN takes 64 epochs, 100 minutes. The test prc-auc score is 0.1431. For reference,
the numbers reported in MoleculeNet is 0.136.
