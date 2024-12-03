from dataset import *
from model import *
from utils import *
from evaluation import *
import argparse
from tqdm import tqdm
import tensorlayerx as tlx
import warnings
warnings.filterwarnings('ignore')
import math
import os
import time
from memory_profiler import memory_usage

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

class MLP(tlx.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = tlx.layers.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = tlx.layers.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = tlx.layers.Linear(in_features=hidden_dim, out_features=output_dim)
        
    def forward(self, x):
        x = tlx.relu(self.fc1(x))
        x = tlx.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = tlx.losses.BinaryCrossEntropy()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = tlx.ops.to_device(data, args.device)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = tlx.optimizers.Adam(lr=args.d_lr, weight_decay=args.d_wd)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = tlx.optimizers.Adam(lr=args.c_lr, weight_decay=args.c_wd)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)
    elif(args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)

    if os.path.isfile(args.dataset+'_hadj.pt'):
        print('########## sample already done #############')
        new_adj = tlx.ops.load(args.dataset+'_hadj.pt')
    else:
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        print('sample begin')
        new_adj = tlx.zeros((data.adj.shape[0], data.adj.shape[0])).int()
        for i in tqdm(range(data.adj.shape[0])):
            neighbor = tlx.convert_to_tensor(data.adj[i].nonzero()).to(args.device)
            mask = (data.sens[neighbor[1]] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            new_adj[i, h_nei_idx] = 1
        print('select done')
        tlx.ops.save(new_adj, args.dataset+'_hadj.pt')
    
    c_X = data.x
    deg = np.sum(new_adj.numpy(), axis=1)
    deg = tlx.convert_to_tensor(deg).cpu()
    indices = tlx.convert_to_tensor(new_adj.nonzero())
    values = new_adj[indices[:, 0], indices[:, 1]]
    mat = tlx.SparseTensor(indices=indices.T, values=values, dense_shape=new_adj.shape).float().cpu()
    h_X = tlx.ops.sparse_dense_matmul(mat, data.x.cpu()) / deg.unsqueeze(-1)
    mask = tlx.any(tlx.isnan(h_X), axis=1)
    h_X = h_X[~mask].to(args.device)
    c_X = c_X[~mask].to(args.device)
    print('node avg degree:', data.edge_index.shape[1] / data.adj.shape[0], ' heteroneighbor degree mean:', deg.float().mean(), ' node without heteroneghbor:', (deg == 0).sum())
    deg_norm = deg
    deg_norm[deg_norm == 0] = 1
    deg_norm = deg_norm.to(args.device)
    
    model = MLP(len(data.x[0]), args.hidden, len(data.x[0])).to(args.device)
    optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=0)

    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0])
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]

    for count in pbar:
        seed_everything(count + args.seed)
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()
        model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf

        for m_epoch in range(0, args.m_epoch):
            model.train()
            output = model(X_train)
            train_loss = tlx.losses.mean_squared_error(output, y_train)
            train_loss.backward()
            optimizer.step()
            model.eval()
            output = model(X_test)
            valid_loss = tlx.losses.mean_squared_error(output, y_test)

            print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(
                m_epoch,
                train_loss,
                valid_loss
            ))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = model.state_dict()
                print('best_val_loss:', m_epoch)

        model.load_state_dict(best_mlp_state)
        model.eval()

        for epoch in range(0, args.epochs):
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()
                optimizer.zero_grad()

                h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                output = classifier(h)
                loss_c = tlx.losses.sigmoid_cross_entropy(output[data.train_mask], tlx.expand_dims(data.y[data.train_mask], -1))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()

            if args.d == 'yes':
                discriminator.train()
                encoder.train()
                for epoch_d in range(0, args.d_epochs):
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()
                    optimizer.zero_grad()

                    h = encoder(data.x + args.delta * model(data.x), data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_d = criterion(output.view(-1), tlx.cast(data.x[:, args.sens_idx], dtype=tlx.float32))

                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_e.step()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(
                data.x, classifier, discriminator, encoder, data, args)

            print(epoch, 'Acc:', accs['test'], 'F1:', F1s['test'],
                  'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'],'tradeoff:',F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']))

            if F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                print('best_val_tradeoff', epoch)
                best_val_tradeoff = F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
                
            model.load_state_dict(best_mlp_state)

        acc[count] = test_acc
        f1[count] = test_f1
        parity[count] = test_parity
        equality[count] = test_equality

    return acc, f1,  parity, equality

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=10)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.1)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=5)
    parser.add_argument('--m_epoch', type=int, default=200)
    parser.add_argument('--d', type=str, default='no')

    args = parser.parse_args()
    args.device = 'cuda' if tlx.BACKEND == 'torch' else 'cpu'
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data['x'].shape[1], 2-1

    acc, f1,  parity, equality = run(data, args)

    end_time = time.time()
    run_time = end_time - start_time
    print('======' + args.dataset + args.encoder + '======')
    print('Acc:', round(np.mean(acc) * 100,2), '±' ,round(np.std(acc) * 100,2), sep='')
    print('f1:', round(np.mean(f1) * 100,2), '±' ,round(np.std(f1) * 100,2), sep='')
    print('parity:', round(np.mean(parity) * 100,2), '±', round(np.std(parity) * 100,2), sep='')
    print('equality:', round(np.mean(equality) * 100,2), '±', round(np.std(equality) * 100,2), sep='')
    print('run time {} s'.format(run_time))
