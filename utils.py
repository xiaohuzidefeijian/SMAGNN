import math
import torch
import numpy as np
import torch.nn as nn

class Edge:
    def __init__(self, edge_list):
        self.n, self.m = edge_list[:,0].max() + 1, edge_list[:,1].max() + 1
        self.pos = []
        self.neg = []
        self.edge_abp = [[] for _ in range(self.n)]
        self.edge_bap = [[] for _ in range(self.m)]
        self.edge_abn = [[] for _ in range(self.n)]
        self.edge_ban = [[] for _ in range(self.m)]
        self.edge_abns = [set() for _ in range(self.n)]
        self.edge_abps = [set() for _ in range(self.n)]
        self.edge_set = set()
        self.edge_pro = dict()
        self.edge_list = edge_list
        self.dega = np.zeros([self.n])
        self.degb = np.zeros([self.m])
        for a,b,s in edge_list:
            self.edge_set.add((a,b))
            self.dega[a]+=1
            self.degb[b]+=1
            if s==1:
                self.pos.append((a,b))
                self.edge_pro[(a,b)] = 1
                self.edge_abp[a].append(b)
                self.edge_abps[a].add(b)
                self.edge_bap[b].append(a)
            if s==-1:
                self.neg.append((a,b))
                self.edge_pro[(a,b)] = -1
                self.edge_abn[a].append(b)
                self.edge_abns[a].add(b)
                self.edge_ban[b].append(a)

def sub_edge(a_node, b_node, edge):
    sub_edge_list = [[], []]
    map_a = {j:i for i,j in enumerate(a_node)}
    map_b = {j:i for i,j in enumerate(b_node)}
    n, m = len(a_node), len(b_node)
    sb = set(b_node)
    for an in a_node:
        for p in edge.edge_abps[an] & sb:
            sub_edge_list[0].append((map_a[an],map_b[p]))
        for p in edge.edge_abns[an] & sb:
            sub_edge_list[1].append((map_a[an],map_b[p]))
    adj = torch.zeros((n, m), dtype=torch.long)
    for i in range(2):
        sub_edge_list[i] = torch.tensor(sub_edge_list[i], dtype=torch.long)
        if len(sub_edge_list[i]):
            adj[sub_edge_list[i][:,0],sub_edge_list[i][:,1]] = -1 if i else 1
    return adj

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
