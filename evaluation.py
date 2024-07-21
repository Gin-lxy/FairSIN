import tensorlayerx as tlx
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric

def evaluate(x, classifier, hp, encoder, data, args):
    classifier.set_eval()
    encoder.set_eval()

    with tlx.no_grad():
        h = encoder(data['x'], data['edge_index'], data['adj_norm_sp'])
        output = classifier(h)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (tlx.greater(output[data['val_mask']].squeeze(), 0)).astype(data['y'].dtype)
    pred_test = (tlx.greater(output[data['test_mask']].squeeze(), 0)).astype(data['y'].dtype)

    accs['val'] = tlx.reduce_sum(tlx.equal(pred_val, data['y'][data['val_mask']])).item() / tlx.reduce_sum(data['val_mask']).item()
    accs['test'] = tlx.reduce_sum(tlx.equal(pred_test, data['y'][data['test_mask']])).item() / tlx.reduce_sum(data['test_mask']).item()

    F1s['val'] = f1_score(data['y'][data['val_mask']].numpy(), pred_val.numpy())
    F1s['test'] = f1_score(data['y'][data['test_mask']].numpy(), pred_test.numpy())

    auc_rocs['val'] = roc_auc_score(data['y'][data['val_mask']].numpy(), output[data['val_mask']].numpy())
    auc_rocs['test'] = roc_auc_score(data['y'][data['test_mask']].numpy(), output[data['test_mask']].numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.numpy(), data['y'][data['val_mask']].numpy(), data['sens'][data['val_mask']].numpy())
    paritys['test'], equalitys['test'] = fair_metric(pred_test.numpy(), data['y'][data['test_mask']].numpy(), data['sens'][data['test_mask']].numpy())

    return accs, auc_rocs, F1s, paritys, equalitys
