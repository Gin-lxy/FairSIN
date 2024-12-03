from dataset import *
from model import *
from utils import *
from evaluation import *
import tensorlayerx as tlx
from tensorlayerx import nn
import tensorlayerx.optimizers as optim
import tensorlayerx.dataflow as dataflow
from tensorlayerx.nn import Module, Linear
import argparse
from tqdm import tqdm
import warnings
import math
import os
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tensorlayerx.optimizers.lr_scheduler import ExponentialDecayLR

class MLP(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = Linear(in_features=input_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.lin2 = Linear(in_features=hidden_dim, out_features=output_dim)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h):
        h = self.lin1(h)
        h = self.lin2(h)
        return h

def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = tlx.losses.BinaryCrossEntropy()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = optim.Adam(discriminator.trainable_weights, learning_rate=args.d_lr, weight_decay=args.d_wd)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = optim.Adam(classifier.trainable_weights, learning_rate=args.c_lr, weight_decay=args.c_wd)

    if args.encoder == 'MLP':
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = optim.Adam(encoder.trainable_weights, learning_rate=args.e_lr, weight_decay=args.e_wd)
    elif args.encoder == 'GCN':
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = optim.Adam(encoder.trainable_weights, learning_rate=args.e_lr, weight_decay=args.e_wd)
    elif args.encoder == 'GIN':
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = optim.Adam(encoder.trainable_weights, learning_rate=args.e_lr, weight_decay=args.e_wd)
    elif args.encoder == 'SAGE':
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = optim.Adam(encoder.trainable_weights, learning_rate=args.e_lr, weight_decay=args.e_wd)

    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        new_adj = tlx.ops.load(args.dataset+'_hadj.pt')
    else:
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        print('sample begin')
        new_adj = np.zeros_like(data.adj)
        for i in tqdm(range(data.adj.shape[0])):
            neighbor = tlx.ops.nonzero(data.adj[i])
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            new_adj[i, h_nei_idx] = 1
        print('select done')
        tlx.ops.save(new_adj, args.dataset+'_hadj.pt')

    new_adj = tlx.convert_to_tensor(new_adj)
    new_adj_sp = new_adj.numpy()
    new_adj_sp = sp.coo_matrix(new_adj)
    new_adj_sp = sp.csr_matrix((new_adj_sp.data, (new_adj_sp.row, new_adj_sp.col)), shape=data.adj.shape)
    data.adj = data.adj + sp.eye(data.adj.shape[0]) + args.delta * new_adj_sp
    adj_norm = sys_normalized_adjacency(data.adj)
    adj_norm_sp = sparse_mx_to_tlx_tensor(adj_norm)
    data.adj_norm_sp = adj_norm_sp

    deg = np.sum(new_adj.numpy(), axis=1)
    deg = tlx.convert_to_tensor(deg)
    print('node avg degree:', data.edge_index.shape[1]/data.adj.shape[0], ' heteroneighbor degree mean:', deg.float().mean(), ' node without heteroneghbor:', (deg == 0).sum())

    for count in pbar:
        tlx.set_seed(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf

        for epoch in range(args.epochs):
            # train classifier
            for epoch_c in range(args.c_epochs):
                classifier.set_train()
                encoder.set_train()
                optimizer_c.clear_grad()
                optimizer_e.clear_grad()

                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = classifier(h)

                loss_c = criterion(output[data.train_mask], tlx.expand_dims(data.y[data.train_mask], axis=1).to(args.device))

                loss_c.backward()
                optimizer_e.apply_gradients(zip(encoder.trainable_weights, encoder.trainable_weights))
                optimizer_c.apply_gradients(zip(classifier.trainable_weights, classifier.trainable_weights))

            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(data.x, classifier, discriminator, encoder, data, args)

            print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'], 'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'], 'tradeoff:', auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']))

            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                print('best_val_tradeoff', epoch)
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--d_lr', type=float, default=0.002)
    parser.add_argument('--d_wd', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0.0001)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0.0001)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--m_epoch', type=int, default=100)

    args = parser.parse_args()
    args.device = tlx.device('GPU' if tlx.ops.gpu_device_count() > 0 else 'CPU')
    tlx.set_device(args.device)
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 2-1  # binary classes are 0,1

    args.train_ratio, args.val_ratio = tlx.convert_to_tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), tlx.convert_to_tensor([
        (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = tlx.ops.maximum(args.train_ratio) / args.train_ratio, tlx.ops.maximum(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    acc, f1, auc_roc, parity, equality = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc:', round(np.mean(auc_roc) * 100, 2), '±', round(np.std(auc_roc) * 100, 2), sep='')
    print('Acc:', round(np.mean(acc) * 100, 2), '±', round(np.std(acc) * 100, 2), sep='')
    print('f1:', round(np.mean(f1) * 100, 2), '±', round(np.std(f1) * 100, 2), sep='')
    print('parity:', round(np.mean(parity) * 100, 2), '±', round(np.std(parity) * 100, 2), sep='')
    print('equality:', round(np.mean(equality) * 100, 2), '±', round(np.std(equality) * 100, 2), sep='')
