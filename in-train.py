from dataset import *
from model import *
from utils import *
from evaluation import *
import argparse
from tqdm import tqdm
import tensorlayerx as tlx
import warnings
#warnings.filterwarnings('ignore')
import math
import os
import time
from memory_profiler import memory_usage
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

class MLP(tlx.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = tlx.nn.Linear(in_features=input_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.fc2 = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.fc3 = tlx.nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
    
def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = tlx.losses.binary_cross_entropy()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)
   
    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = tlx.optimizers.Adam(lr=args.d_lr, weight_decay=args.d_wd, params=discriminator.trainable_weights)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = tlx.optimizers.Adam(lr=args.c_lr, weight_decay=args.c_wd, params=classifier.trainable_weights)

    if args.encoder == 'MLP':
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd, params=encoder.trainable_weights)
    elif args.encoder == 'GCN':
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd, params=encoder.trainable_weights)
    elif args.encoder == 'GIN':
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd, params=encoder.trainable_weights)
    elif args.encoder == 'SAGE':
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd, params=encoder.trainable_weights)

    # examine if the file exists
    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        # load data
        new_adj = tlx.files.load_pt(args.dataset+'_hadj.pt')
    else:
        # pretrain neighbor predictor
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        # neighbor select
        print('sample begin.')
        # compute heterogeneous neighbor
        new_adj = np.zeros((data.adj.shape[0], data.adj.shape[0])).astype(int)
        for i in tqdm(range(data.adj.shape[0])):
            neighbor = np.nonzero(data.adj[i])[0]
            data.sens = np.array(data.sens, dtype=np.int64)
            mask = (data.sens[neighbor] != data.sens[i])
            h_nei_idx = neighbor[mask]
            new_adj[i, h_nei_idx] = 1
        print('select done.')
        # save data
        tlx.files.save_pt(new_adj, args.dataset+'_hadj.pt')
    
    c_X = data.x
    new_adj = tlx.convert_to_tensor(new_adj)
    # compute degree matrix and heterogeneous neighbor's feature 
    deg = np.sum(new_adj.numpy(), axis=1)
    deg = tlx.convert_to_tensor(deg)
    indices = tlx.nonzero(new_adj)
    values = new_adj[indices[:, 0], indices[:, 1]]
    mat = tlx.sparse.SparseTensor(indices.t(), values, new_adj.shape).float()
    h_X = tlx.ops.sparse_dense_matmul(mat, data.x) / deg.unsqueeze(-1) 
    # examine if any row contains NaN 
    mask = tlx.reduce_any(tlx.isnan(h_X), axis=1)
    # delete rows containing NaN 
    h_X = tlx.boolean_mask(h_X, ~mask)
    c_X = tlx.boolean_mask(c_X, ~mask)
    print('node avg degree:', data.edge_index.shape[1]/data.adj.shape[0], ' heteroneighbor degree mean:', deg.mean(), ' node without heteroneighbor:', tlx.reduce_sum(deg == 0))
    deg_norm = deg
    deg_norm = tlx.where(deg_norm == 0, 1, deg_norm)
    
    model = MLP(len(data.x[0]), args.hidden, len(data.x[0])).to(args.device) 
    optimizer = tlx.optimizers.Adam(lr=args.m_lr, weight_decay=0.001, params=model.trainable_weights)

    indices = np.arange(c_X.shape[0]) # help for check the index after split
    indices_train, indices_test, y_train, y_test = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = tlx.gather(c_X, indices_train), tlx.gather(c_X, indices_test), tlx.gather(h_X, indices_train), tlx.gather(h_X, indices_test)

    for count in pbar:
        tlx.set_seed(count + args.seed)
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # train mlp
            for m_epoch in range(args.m_epoch):
                model.set_train()
                with tlx.GradientTape() as tape:
                    output = model(X_train)
                    train_loss = tlx.losses.mean_squared_error(output, y_train)
                grad = tape.gradient(train_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
                
                model.set_eval()
                output = model(X_test)
                valid_loss = tlx.losses.mean_squared_error(output, y_test)
                
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_mlp_state = model.trainable_weights

            model.load_weights(best_mlp_state)
            model.set_eval()

            # train classifier
            classifier.set_train()
            encoder.set_train()
            for epoch_c in range(args.c_epochs):
                with tlx.GradientTape(persistent=True) as tape:
                    h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                    output = classifier(h)
                    loss_c = tlx.losses.sigmoid_cross_entropy(output[data.train_mask], tlx.expand_dims(data.y[data.train_mask], axis=1))
                grad_e = tape.gradient(loss_c, encoder.trainable_weights)
                grad_c = tape.gradient(loss_c, classifier.trainable_weights)
                optimizer_e.apply_gradients(zip(grad_e, encoder.trainable_weights))
                optimizer_c.apply_gradients(zip(grad_c, classifier.trainable_weights))

            if args.d == 'yes':
                # train discriminator to recognize the sensitive group
                discriminator.set_train()
                encoder.set_train()
                for epoch_d in range(args.d_epochs):
                    with tlx.GradientTape(persistent=True) as tape:
                        h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)
                        loss_d = criterion(output.view(-1), data.x[:, args.sens_idx])
                    grad_d = tape.gradient(loss_d, discriminator.trainable_weights)
                    grad_e = tape.gradient(loss_d, encoder.trainable_weights)
                    optimizer_d.apply_gradients(zip(grad_d, discriminator.trainable_weights))
                    optimizer_e.apply_gradients(zip(grad_e, encoder.trainable_weights))

            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(data.x, classifier, discriminator, encoder, data, args)
            print(epoch, 'Acc:', accs['test'], 'F1:', F1s['test'], 'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'], 'tradeoff:', auc_rocs['test'] + F1s['test'] + accs['test'] - args.alpha * (tmp_parity['test'] + tmp_equality['test']))

            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                print('best_val_tradeoff', epoch)
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val'])
                
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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=10)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=5)
    parser.add_argument('--m_epoch', type=int, default=20)
    parser.add_argument('--d', type=str, default='no')
    parser.add_argument('--m_lr', type=float, default=0.1)

    args = parser.parse_args()
    tlx.set_device(device='CPU')  # 设置为CPU
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 2 - 1  # binary classes are 0,1 

    acc, f1, auc_roc, parity, equality = run(data, args)

    print('======' + args.dataset + args.encoder + '======')
    print('Acc:', round(np.mean(acc) * 100, 2), '±', round(np.std(acc) * 100, 2), sep='')
    print('f1:', round(np.mean(f1) * 100, 2), '±', round(np.std(f1) * 100, 2), sep='')
    print('parity:', round(np.mean(parity) * 100, 2), '±', round(np.std(parity) * 100, 2), sep='')
    print('equality:', round(np.mean(equality) * 100, 2), '±', round(np.std(equality) * 100, 2), sep='')
