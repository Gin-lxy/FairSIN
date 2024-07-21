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

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

class MLP_discriminator(tlx.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = tlx.layers.Linear(out_features=2, in_features=args.hidden-1)
        
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)
        return h

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

    discriminator_ = MLP_discriminator(args).to(args.device)
    optimizer_d_ = tlx.optimizers.Adam(lr=args.d_lr, weight_decay=args.d_wd)
    
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
    
    # Check if the file exists
    if os.path.isfile(args.dataset + '_hadj.pt'):
        print('########## sample already done #############')
        new_adj = tlx.ops.load(args.dataset + '_hadj.pt')
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
        tlx.ops.save(new_adj, args.dataset + '_hadj.pt')
    
    c_X = data.x
    new_adj = new_adj.cpu()
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
    optimizer = tlx.optimizers.Adam(lr=args.m_lr, weight_decay=0)

    from sklearn.model_selection import train_test_split

    indices = np.arange(c_X.shape[0])
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]
    
    test_acc1, test_acc2, test_acc3, test_H_score1, test_H_score2, test_H_score3 = np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)
    for count in pbar:
        seed_everything(count + args.seed)
        model.reset_parameters()
        discriminator.reset_parameters()
        discriminator_.reset_parameters()
        
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

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = model.state_dict()

        model.load_state_dict(best_mlp_state)

        x = data.x
        x = x[:, [i for i in range(data.x.size(1)) if i != args.sens_idx]]
        best_val_loss = math.inf
        encoder.train()
        for epoch_d in range(0, args.d_epochs):
            optimizer_d.zero_grad()
            discriminator.train()
            output = discriminator(x)
            loss_d = tlx.losses.softmax_cross_entropy_with_logits(output[data.train_mask], tlx.cast(data.x[:, args.sens_idx][data.train_mask], dtype=tlx.int64))
            loss_d.backward()
            optimizer_d.step()

            discriminator.eval()
            output = discriminator(x)
            valid_loss = tlx.losses.softmax_cross_entropy_with_logits(output[data.val_mask], tlx.cast(data.x[:, args.sens_idx][data.val_mask], dtype=tlx.int64))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = discriminator.state_dict()
        
        discriminator.load_state_dict(best_mlp_state)
        discriminator.eval()
        output = discriminator(x)
        pred_val1 = (output[data.test_mask][:, 1] > output[data.test_mask][:, 0]).astype(data.y.dtype)
        acc1 = tlx.reduce_sum(tlx.equal(pred_val1, data.x[:, args.sens_idx][data.test_mask])).item() / tlx.reduce_sum(data.test_mask).item()
        probs = tlx.softmax(output, axis=1)
        prob1 = tlx.reduce_mean(probs[:, 0][data.test_mask])
        prob_all1 = tlx.reduce_mean(probs[:, 0])
        H_score1 = tlx.reduce_sum(1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])).item() / tlx.reduce_sum(data.test_mask).item()
        pred_val1 = 1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])

        best_val_loss = math.inf
        discriminator_.reset_parameters()
        encoder.eval()
        emb = tlx.ops.sparse_dense_matmul(data.adj_norm_sp, data.x)
        emb = emb[:, [i for i in range(0, data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ = tlx.losses.softmax_cross_entropy_with_logits(output[data.train_mask], tlx.cast(data.x[:, args.sens_idx][data.train_mask], dtype=tlx.int64))
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = tlx.losses.softmax_cross_entropy_with_logits(output[data.val_mask], tlx.cast(data.x[:, args.sens_idx][data.val_mask], dtype=tlx.int64))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = discriminator_.state_dict()

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val2 = (output[data.test_mask][:, 1] > output[data.test_mask][:, 0]).astype(data.y.dtype)
        acc2 = tlx.reduce_sum(tlx.equal(pred_val2, data.x[:, args.sens_idx][data.test_mask])).item() / tlx.reduce_sum(data.test_mask).item()
        H_score2 = tlx.reduce_sum(1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])).item() / tlx.reduce_sum(data.test_mask).item()
        pred_val2 = 1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])
        probs = tlx.softmax(output, axis=1)
        prob2 = tlx.reduce_mean(probs[:, 0][data.test_mask])
        prob_all2 = tlx.reduce_mean(probs[:, 0])

        best_val_loss = math.inf
        discriminator_.reset_parameters()
        origin_state = discriminator_.state_dict()
        encoder.eval()
        model.eval()
        emb = data.x + args.delta * model(data.x).detach()
        emb = emb[:, [i for i in range(0, data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ = tlx.losses.softmax_cross_entropy_with_logits(output[data.train_mask], tlx.cast(data.x[:, args.sens_idx][data.train_mask], dtype=tlx.int64))
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = tlx.losses.softmax_cross_entropy_with_logits(output[data.val_mask], tlx.cast(data.x[:, args.sens_idx][data.val_mask], dtype=tlx.int64))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = discriminator_.state_dict()

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val3 = (output[data.test_mask][:, 1] > output[data.test_mask][:, 0]).astype(data.y.dtype)
        acc3 = tlx.reduce_sum(tlx.equal(pred_val3, data.x[:, args.sens_idx][data.test_mask])).item() / tlx.reduce_sum(data.test_mask).item()
        H_score3 = tlx.reduce_sum(1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])).item() / tlx.reduce_sum(data.test_mask).item()
        pred_val3 = 1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])

        best_val_loss = math.inf
        discriminator_.load_state_dict(origin_state)
        encoder.eval()
        model.eval()
        emb = data.x + args.delta * model(data.x).detach()
        emb = tlx.ops.sparse_dense_matmul(data.adj_norm_sp, emb) / 2
        emb = emb[:, [i for i in range(0, data.x.size(1)) if i != args.sens_idx]]
        for epoch_d in range(0, args.d_epochs):
            discriminator_.train()
            optimizer_d_.zero_grad()
            output = discriminator_(emb)
            loss_d_ = tlx.losses.softmax_cross_entropy_with_logits(output[data.train_mask], tlx.cast(data.x[:, args.sens_idx][data.train_mask], dtype=tlx.int64))
            loss_d_.backward()
            optimizer_d_.step()

            discriminator_.eval()
            output = discriminator_(emb)
            valid_loss = tlx.losses.softmax_cross_entropy_with_logits(output[data.val_mask], tlx.cast(data.x[:, args.sens_idx][data.val_mask], dtype=tlx.int64))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_mlp_state = discriminator_.state_dict()

        discriminator_.load_state_dict(best_mlp_state)
        discriminator_.eval()
        output = discriminator_(emb)
        pred_val4 = (output[data.test_mask][:, 1] > output[data.test_mask][:, 0]).astype(data.y.dtype)
        acc4 = tlx.reduce_sum(tlx.equal(pred_val4, data.x[:, args.sens_idx][data.test_mask])).item() / tlx.reduce_sum(data.test_mask).item()
        H_score4 = tlx.reduce_sum(1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])).item() / tlx.reduce_sum(data.test_mask).item()
        pred_val4 = 1/2 + 1/2 * tlx.tanh(output[data.test_mask][:, 0] - output[data.test_mask][:, 1])

        print('======' + args.dataset + '======')
        print('score1:', H_score1)
        print('score2:', H_score2)
        print('score3:', H_score3)
        print('score4:', H_score4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--d_lr', type=float, default=0.01)
    parser.add_argument('--d_wd', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.01)
    parser.add_argument('--c_wd', type=float, default=0.0001)
    parser.add_argument('--e_lr', type=float, default=0.01)
    parser.add_argument('--e_wd', type=float, default=0.0001)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--m_epoch', type=int, default=100)
    parser.add_argument('--m_lr', type=float, default=0.01)

    args = parser.parse_args()
    args.device = 'cuda' if tlx.BACKEND == 'torch' else 'cpu'
    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data['x'].shape[1], 2-1

    args.train_ratio, args.val_ratio = tlx.convert_to_tensor([
        tlx.reduce_sum(tlx.equal(data['y'][data['train_mask']], 0)), tlx.reduce_sum(tlx.equal(data['y'][data['train_mask']], 1))]), tlx.convert_to_tensor([
            tlx.reduce_sum(tlx.equal(data['y'][data['val_mask']], 0)), tlx.reduce_sum(tlx.equal(data['y'][data['val_mask']], 1))])
    args.train_ratio, args.val_ratio = tlx.max(args.train_ratio) / args.train_ratio, tlx.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
        tlx.cast(data['y'][data['train_mask']], dtype=tlx.int64)], args.val_ratio[tlx.cast(data['y'][data['val_mask']], dtype=tlx.int64)]

    run(data, args)
