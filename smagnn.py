#-*- coding: utf-8 -*-
import os
import time
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='Device')
parser.add_argument('--dataset_name', type=str, default='house1to10')
parser.add_argument('--emb_size', type=int, default=32, help='Embeding Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate')
parser.add_argument('--seed', type=int, default=222, help='Random seed')
parser.add_argument('--epoch', type=int, default=80000, help='Epoch')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--neighbours', type=int, default=200, help='Max Neighbours')
parser.add_argument('--val_epoch', type=int, default=50, help='Epoch Interval For Validation')
parser.add_argument('--zero_shot', action='store_true', help='Zero Shot Learning')
parser.add_argument('--ckpt', type=str, help='Checkpoint path')
parser.add_argument('--eval', action='store_true', help='Evaluation')
parser.add_argument('--comment', type=str, default='')
args = parser.parse_args()

data_prefix = 'experiments-data'

if 'cuda' in args.device:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Edge, sub_edge, GELU
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

from common import DATA_EMB_DIC

class GraphDataset(Dataset):
    def __init__(self, val, test=False, edge=None):
        self.val = Edge(val)
        print(len(self.val.edge_set))
        self.test = test
        if edge is None:
            self.edges = edges
        else:
            self.edges = edge

    def __len__(self):
        return len(self.val.edge_list)

    def __getitem__(self, index):
        edge = self.val.edge_list[index]
        left, right, s = [edge[0]], [edge[1]], edge[2]==1
        neighbour = np.array(self.edges.edge_abn[left[0]] + self.edges.edge_abp[left[0]])
        if len(neighbour) == 0:
            left_n = []
        else:
            deg = self.edges.degb[neighbour]
            deg_idx = np.argsort(deg)[::-1]
            neighbour = neighbour[deg_idx].tolist()
            samples = neighbour[:args.neighbours]
            left_n = list(set(samples) - set(right))
        neighbour = np.array(self.edges.edge_ban[right[0]] + self.edges.edge_bap[right[0]])
        if len(neighbour) == 0:
            right_n = []
        else:
            deg = self.edges.dega[neighbour]
            deg_idx = np.argsort(deg)[::-1]
            neighbour = neighbour[deg_idx].tolist()
            samples = neighbour[:args.neighbours]
            right_n = list(set(samples)-set(left))
        sub_0 = sub_edge(left, left_n, self.edges)
        sub_1 = sub_edge(right_n, left_n, self.edges)
        sub_2 = sub_edge(right_n, right, self.edges)
        edge_s = int(s)
        return {
            'left': left,
            'right': right,
            'left_n': left_n,
            'right_n': right_n,
            'sub_0': sub_0,
            'sub_1': sub_1,
            'sub_2': sub_2,
            'edge_s': edge_s
        }

def pad_tensor(batch):
    max_size = [max(tensor.size(dim) for tensor in batch) for dim in range(batch[0].dim())]
    batch_size = len(batch)
    background = torch.zeros([batch_size] + max_size, dtype=batch[0].dtype)
    for i, tensor in enumerate(batch):
        indices = tuple(slice(0, sz) for sz in tensor.size())
        background[i][indices] = tensor
    return background

def collate_fn(batch):
    batched_data = {key: [] for key in batch[0]}
    for item in batch:
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                batched_data[key].append(value)
            elif isinstance(value, list):
                batched_data[key].append(torch.tensor(value))
            else:
                batched_data[key].append(value)
    for key in batched_data:
        if isinstance(batched_data[key][0], torch.Tensor):
            if all(tensor.shape == batched_data[key][0].shape for tensor in batched_data[key]):
                batched_data[key] = torch.stack(batched_data[key])
            else:
                batched_data[key] = pad_tensor(batched_data[key])
        else:
            batched_data[key] = torch.tensor(batched_data[key])

    return batched_data

class Attntion(nn.Module):
    def __init__(self, input_dim, head=4):
        super(Attntion, self).__init__()
        self.bt_pre = nn.Linear(input_dim, 6)
        self.bt_cur = nn.Linear(input_dim, 6)
        self.fcc = nn.Sequential(
            GELU(),
            nn.Linear(8, head),
        )
        self.fcg = nn.Sequential(
            GELU(),
            nn.Linear(8, input_dim),
        )
        self.head = head
        self.dim = input_dim
        self.fuse = nn.Sequential(
            GELU(),
            nn.Linear(12, 4),
        )
        self.ffn = nn.Sequential(
            GELU(),
            nn.Linear(12, 8),
        )

    def forward(self, prev, curr, edges):
        bt_pre = self.bt_pre(prev)
        bt_cur = self.bt_cur(curr)
        shape = (bt_pre.shape[0],bt_pre.shape[1],bt_cur.shape[1],bt_pre.shape[2])
        bt_pre = bt_pre.unsqueeze(2).expand(shape)
        bt_cur = bt_cur.unsqueeze(1).expand(shape)
        c = torch.cat((bt_pre, bt_cur), dim=-1)
        c = self.fuse(c)
        c[edges==0] = 0
        edges1, edges2 = torch.sum(edges, dim=1)+1, torch.sum(edges, dim=2)+1
        edges1[edges1==0] = 1
        edges2[edges2==0] = 1
        d = torch.sum(c, dim=1).unsqueeze(1).expand(c.shape) / edges1.unsqueeze(1).unsqueeze(-1)
        e = torch.max(c, dim=1)[0].unsqueeze(1).expand(c.shape)
        fused = self.ffn(torch.cat((c,d,e),dim=-1))
        c = self.fcc(fused).transpose(-2,-3)
        mask = edges.transpose(-1,-2)
        res_list = []
        for i in range(self.head):
            c_true = c[:,:,:,i].clone()
            if c_true.shape[2] != 1:
                c_true[mask==0] = -9e15
                c_true = F.softmax(c_true, dim=2).clone()
            c_true[mask==0] = 0
            res = torch.bmm(c_true, prev[:,:,self.dim//self.head*i:self.dim//self.head*(i+1)])
            res_list.append(res)
        res = torch.cat(res_list, dim=-1)
        fused = fused.clone()
        fused[edges==0] = 0
        res = res + self.fcg(torch.sum(fused, dim=1) / edges1.unsqueeze(-1))
        return res

class MultiHeadAttLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super(MultiHeadAttLayer, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.weight_curr = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.normal_(self.weight_curr, mean=0, std=0.1)
        self.fcp = nn.Linear(input_dim, output_dim)
        self.fcn = nn.Linear(input_dim, output_dim)
        self.ffn = nn.Sequential(
            GELU(),
            nn.Linear(input_dim, input_dim//2),
            GELU(),
            nn.Linear(input_dim//2, output_dim),
        )
        self.attn = Attntion(input_dim)
        self.lnrom = nn.LayerNorm(input_dim)

    def forward(self, prev_layer_features, current_layer_features, edges):
        positive_features = self.attn(prev_layer_features, current_layer_features, edges==1)
        negative_features = self.attn(prev_layer_features, current_layer_features, edges==-1)
        transformed_agg_features = self.fcp(positive_features) - self.fcn(negative_features)
        current_layer_features = torch.matmul(current_layer_features, self.weight_curr) + transformed_agg_features
        current_layer_features = self.lnrom(current_layer_features)

        return self.ffn(current_layer_features) + current_layer_features
    
class SubGraphLayer(nn.Module):
    def __init__(self, input_dim):
        super(SubGraphLayer, self).__init__()
        self.attn1 = MultiHeadAttLayer(input_dim)
        self.attn2 = MultiHeadAttLayer(input_dim)
        self.attn3 = MultiHeadAttLayer(input_dim)
        self.attnp = MultiHeadAttLayer(input_dim)
        self.attnq = MultiHeadAttLayer(input_dim)
    
    def forward(self, emb_a, emb_an, emb_bn, emb_b, sub_0, sub_1, sub_2):
        sub_1 = sub_1.transpose(1,2)
        sub_11 = sub_2.transpose(1,2).clone()
        sub_11[sub_11!=0] = 1
        sub_22 = sub_0.transpose(1,2).clone()
        sub_22[sub_22!=0] = 1
        x0 = emb_a
        x1 = self.attn1(x0, emb_bn, sub_0)
        x2 = self.attn2(x1, emb_an, sub_1) + self.attnp(emb_a, emb_an, sub_11)
        x3 = self.attn3(x2, emb_b, sub_2) + self.attnq(emb_bn, emb_b, sub_22)
        return x3

class SMAGNN(nn.Module):
    def __init__(self, n, m, emb_size=32):
        super(SMAGNN, self).__init__()

        if not args.zero_shot:
            self.features_a = nn.Parameter(torch.randn((n, emb_size)), requires_grad=True)
            self.features_b = nn.Parameter(torch.randn((m, emb_size)), requires_grad=True)
        self.suba = SubGraphLayer(emb_size)
        self.subb = SubGraphLayer(emb_size)
        self.fcx = nn.Linear(emb_size, emb_size)
        self.fcy = nn.Linear(emb_size, emb_size)
        self.B = nn.Linear(emb_size, emb_size)
        self.C = nn.Sequential(
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size*5, emb_size),
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size, emb_size//4),
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size//4, 1),
        )
        self.emb_size = emb_size

    def get_embeddings(self, a, b, end=0):
        if args.zero_shot:
            emb_a = torch.randn((*a.shape, self.emb_size), device=a.device)
            emb_b = torch.randn((*b.shape, self.emb_size), device=b.device)
        else:
            emb_a = self.features_a[a.long()]
            emb_b = self.features_b[b.long()]
            if end:
                emb_a, emb_b = emb_a.detach(), emb_b.detach()
        emb_a = self.fcx(emb_a)
        emb_b = self.fcy(emb_b)
        return emb_a, emb_b

    def forward(self, left, left_n, right_n, right, sub_0, sub_1, sub_2, **kwargs):
        embed_a, embed_b = self.get_embeddings(left, right, end=1)
        embed_an, embed_bn = self.get_embeddings(right_n, left_n)
        x = self.suba(embed_a, embed_an, embed_bn, embed_b, sub_0, sub_1, sub_2)
        y = self.subb(embed_b, embed_bn, embed_an, embed_a, sub_2.transpose(1,2), sub_1.transpose(1,2), sub_0.transpose(1,2))
        fuse = torch.cat((self.B(x)*y,x,y,embed_a,embed_b),dim=-1)
        fuse = self.C(fuse).squeeze(-1).squeeze(-1)*4
        return torch.sigmoid(fuse)

    def loss(self, pred_y, y):
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        pos_ratio = y.sum() /  y.size()[0]
        weight = torch.where(y > 0.5, 1./pos_ratio, 1./(1-pos_ratio))
        acc = sum((y>0.5) == (pred_y>0.5))/y.shape[0]
        return F.binary_cross_entropy(pred_y, y, weight=weight), acc

def load_data(dataset_name):
    train_file_path = os.path.join(data_prefix, f'{dataset_name}_training.txt')
    val_file_path = os.path.join(data_prefix, f'{dataset_name}_validation.txt')
    test_file_path = os.path.join(data_prefix, f'{dataset_name}_testing.txt')
    # if not os.path.exists(val_file_path):
    #     os.rename(os.path.join(data_prefix, f'{dataset_name}_val.txt'),val_file_path)
    # if not os.path.exists(test_file_path):
    #     os.rename(os.path.join(data_prefix, f'{dataset_name}_test.txt'),test_file_path)

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)

def power(tensor,exponent):
    return torch.sign(tensor) * torch.pow(torch.abs(tensor), exponent)

@torch.no_grad()
def test_and_val(dataloader, model, mode='val', epoch=0):
    model.eval()
    preds_ys, ys = [], []
    for data in dataloader:
        d = {key: value.cuda() for key, value in data.items()}
        pred = model(**d)
        preds_ys.append(pred)
        ys.append(d['edge_s'].float())
    preds = torch.cat(preds_ys, dim=-1).cpu().numpy()
    y = torch.cat(ys, dim=-1).cpu().numpy()
    preds[preds >= 0.5]  = 1
    preds[preds < 0.5] = 0
    auc = roc_auc_score(y, preds)
    f1 = f1_score(y, preds)
    macro_f1 = f1_score(y, preds, average='macro')
    micro_f1 = f1_score(y, preds, average='micro')
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1' : f1,
        f'{mode}_mac' : macro_f1,
        f'{mode}_mic' : micro_f1,
    }
    for k, v in res.items():
        mode ,_, metric = k.partition('_')
        if not args.eval:
            tb_writer.add_scalar(f'{metric}/{mode}', v, epoch)
    return res

def format_res(data, p=1):
    s = {k:f"{v*100:.{p}f}" for k,v in data.items()}
    s = str(s).strip('{}').replace("'",'')
    return s

def split(edgelist, u, v, n, m):
    mask1 = np.isin(edgelist[:, 0], u) & np.isin(edgelist[:, 1], v)
    mask2 = ~np.isin(edgelist[:, 0], u) & ~np.isin(edgelist[:, 1], v)
    edgelist1 = edgelist[mask1]
    edgelist2 = edgelist[mask2]

    sorted_u = np.sort(u)
    sorted_v = np.sort(v)
    edgelist1[:, 0] = [np.where(sorted_u == x)[0][0] for x in edgelist1[:, 0]]
    edgelist1[:, 1] = [np.where(sorted_v == x)[0][0] for x in edgelist1[:, 1]]

    u_complement = np.setdiff1d(np.arange(n), u)
    v_complement = np.setdiff1d(np.arange(m), v)
    sorted_u_complement = np.sort(u_complement)
    sorted_v_complement = np.sort(v_complement)
    edgelist2[:, 0] = [np.where(sorted_u_complement == x)[0][0] for x in edgelist2[:, 0]]
    edgelist2[:, 1] = [np.where(sorted_v_complement == x)[0][0] for x in edgelist2[:, 1]]
    return edgelist1, edgelist2

def run():
    global edges
    train_edgelist, val_edgelist, test_edgelist = load_data(args.dataset_name)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    print(set_a_num, set_b_num)
    edges = Edge(train_edgelist, set_a_num, set_b_num)

    model = SMAGNN(n=set_a_num, m=set_b_num, emb_size=args.emb_size)
    if args.eval:
        setup_seed(args.seed)
        checkpoint = torch.load(args.ckpt, weights_only=True)
        if args.zero_shot:
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if 'feature' not in k}
            model.load_state_dict(filtered_checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint)
        model = model.cuda()
        dataset_test = GraphDataset(test_edgelist, test=1, edge=edges)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True)
        res = test_and_val(dataloader_test, model, mode='test', epoch=args.epoch)
        print(format_res(res))
        return
    model = model.cuda()

    params = [[], []]
    for name, param in model.named_parameters():
        if name.startswith('features'):
            params[0].append(param)
        else:
            params[1].append(param)
    param_groups = [
        {'params': params[0], 'lr': args.lr*1},
        {'params': params[1], 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler_slow = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    dataset = GraphDataset(train_edgelist, edge=edges)
    dataset_val = GraphDataset(val_edgelist, test=1, edge=edges)
    dataset_test = GraphDataset(test_edgelist, test=1, edge=edges)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=16, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True)
    
    epoch = 0
    res_best = dict()
    best_auc = 0.0
    tm = time.time()
    while epoch < args.epoch:
        for data in dataloader:
            data = {key: value.cuda() for key, value in data.items()}
            model.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred_y = model(**data)
                train_y = data['edge_s'].float()
                loss = model.loss(pred_y, train_y)[0]
                tb_writer.add_scalar('loss0/train', float(loss), epoch)

                loss.backward()
                optimizer.step()
            
            epoch += 1
            if epoch%100 == 0:
                print(epoch, (time.time()-tm)/3600)
            if epoch%args.val_epoch == 0:
                model.eval()
                with torch.set_grad_enabled(False):
                    val_res = test_and_val(dataloader_val, model, mode='val', epoch=epoch)
                    if val_res['val_auc']>best_auc:
                        best_auc = val_res['val_auc']
                        res_best.update(val_res)
                        torch.save(model.state_dict(), f'./res/{args.dataset_name}_{epoch}.pth')

                        test_res = test_and_val(dataloader_test, model, mode='test', epoch=epoch)
                        res_best.update(test_res)
                        print(f'\r\033[Kepoch: {epoch}\t{format_res(test_res)}')
                        with open(f'./txt/{dataset_name}_{comment}.txt','a') as fh:
                            fh.write(f'{epoch}\t{format_res(res_best,p=4)}\n')
                    print(f'\r\033[Kepoch: {epoch}\tval_auc: {val_res["val_auc"]*100:.1f}\tbest_auc: {best_auc*100:.1f}', end='')

            if epoch%50 == 0:
                if epoch < 6000:
                    scheduler.step()
                elif epoch < 30000:
                    scheduler_slow.step()
            
            if epoch >= args.epoch:
                break
    print()

if __name__ == "__main__":
    dataset_name = args.dataset_name
    if args.eval:
        for i in range(0,5):
            args.dataset_name = dataset_name+'-'+str(i+1)
            run()
        exit()
    if args.comment:
        comment = args.comment
    else:
        comment = str(int(time.time())%10000000)
    with open(f'./code/{dataset_name}_{comment}.py','a') as fh:
        with open(__file__,'r') as f:
            fh.write(f.read())
            fh.write('\n\n\n\n\n\n')
    for i in range(0,5):
        hyper_params = dict(vars(args))
        del hyper_params['device']
        hyper_params = "~".join([f"{k}-{v}" for k,v in hyper_params.items()])
        tb_writer = SummaryWriter(log_dir=f'./logs/{hyper_params}')
        with open(f'./txt/{dataset_name}_{comment}.txt','a') as fh:
            if i==0:
                fh.write(hyper_params)
            fh.write(f'\n{i}\n')
        print(f'training: {i}')
        
        setup_seed(args.seed)
        args.dataset_name = dataset_name+'-'+str(i+1)
        run()