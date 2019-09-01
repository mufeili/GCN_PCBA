import datetime
import time
import torch

from dgl import model_zoo
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import PCBADataset, split_dataset
from utils import Meter, EarlyStopping, collate_molgraphs, set_random_seed

def update_default_configure(args):
    default_configure = {
        'batch_size': 128,
        'lr': 1e-3,
        'patience': 10,
        'atom_data_field': 'h'
    }
    args.update(default_configure)

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        atom_feats, labels, mask = atom_feats.to(args['device']), \
                                   labels.to(args['device']), \
                                   mask.to(args['device'])
        logits = model(bg, atom_feats)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (mask != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, mask)
    train_roc_auc = train_meter.roc_auc_averaged_over_tasks()
    print('epoch {:d}/{:d}, training roc-auc score {:.4f}'.format(
        epoch + 1, args['num_epochs'], train_roc_auc))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            atom_feats = bg.ndata.pop(args['atom_data_field'])
            atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
            logits = model(bg, atom_feats)
            eval_meter.update(logits, labels, mask)
    return eval_meter.roc_auc_averaged_over_tasks()

def main(args):
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed()

    dataset = PCBADataset(chunk_size=args['chunk_size'])
    trainset, valset, testset = split_dataset(dataset)
    train_loader = DataLoader(trainset, batch_size=args['batch_size'], collate_fn=collate_molgraphs)
    val_loader = DataLoader(valset, batch_size=args['batch_size'], collate_fn=collate_molgraphs)
    test_loader = DataLoader(testset, batch_size=args['batch_size'], collate_fn=collate_molgraphs)

    if args['model'] == 'GCN':
        model = model_zoo.chem.GCNClassifier(in_feats=74,
                                             gcn_hidden_feats=[128, 128],
                                             classifier_hidden_feats=128,
                                             n_tasks=dataset.n_tasks)
    elif args['model'] == 'GAT':
        model = model_zoo.chem.GATClassifier(in_feats=74,
                                             gat_hidden_feats=[64, 64],
                                             num_heads=[4, 4],
                                             classifier_hidden_feats=128,
                                             n_tasks=dataset.n_tasks)

    loss_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(
        trainset.task_pos_weights).to(args['device']), reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    stopper = EarlyStopping(patience=args['patience'])
    model.to(args['device'])

    t0 = time.time()

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_roc_auc = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_roc_auc, model)
        print('epoch {:d}/{:d}, validation roc-auc score {:.4f}, best validation roc-auc score {:.4f}'.format(
            epoch + 1, args['num_epochs'], val_roc_auc, stopper.best_score))
        if early_stop:
            break

    t1 = time.time()
    print('It took {} to finish training for an epoch.'.format(datetime.timedelta(seconds=t1 - t0)))

    stopper.load_checkpoint(model)
    test_roc_auc = run_an_eval_epoch(args, model, test_loader)
    print('test roc-auc score {:.4f}'.format(test_roc_auc))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Molecule Classification')
    parser.add_argument('-m', '--model', type=str, choices=['GCN', 'GAT'],
                        help='Model to use')
    parser.add_argument('-c', '--chunk-size', type=int, default=1,
                        help='Number of preprocessed molecules in each pickle file')
    parser.add_argument('-n', '--num-epochs', type=int, default=1,
                        help='Max number of epochs to train the model')
    args = parser.parse_args().__dict__
    update_default_configure(args)

    main(args)
