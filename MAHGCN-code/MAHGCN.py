import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

dir_path = r"C:\Users\Faruk\Code\CCIR_Project\MAHGCN-code"

class MAHGCN(nn.Module):
    """Option A: The Original MAHGCN"""
    def __init__(self, act, drop_p, layer, degree_normalize=False):
        super(MAHGCN, self).__init__()
        self.layer=layer
        self.net_gcn_down1 = GCN(1000, 1, act, drop_p, degree_normalize)
        self.net_pool1 = AtlasMap(1000, 400, drop_p)
        self.net_gcn_down2 = GCN(400, 1, act, drop_p, degree_normalize)
        self.net_pool2 = AtlasMap(400, 300, drop_p)
        self.net_gcn_down3 = GCN(300, 1, act, drop_p, degree_normalize)
        self.net_pool3 = AtlasMap(300, 200, drop_p)
        self.net_gcn_down4 = GCN(200, 1, act, drop_p, degree_normalize)


    def forward(self, g1, g2, g3, g4, h):
        # g1: 200-parcel, g2: 300-parcel, g3: 400-parcel, g4: 1000-parcel
        if self.layer == 4:
            h = self.net_gcn_down1(g4, h)
            downout1 = h
            h = self.net_pool1(h)
            h = self.net_gcn_down2(g3, torch.diag(h))
            downout2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, torch.diag(h))
            downout3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, torch.diag(h))
            hh = torch.cat((downout1, downout2, downout3, h))
        elif self.layer == 3:
            h = self.net_gcn_down2(g3, h)
            downout2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, torch.diag(h))
            downout3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, torch.diag(h))
            hh = torch.cat((downout2, downout3, h))
        elif self.layer == 2:
            h = self.net_gcn_down3(g2, h)
            downout3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, torch.diag(h))
            hh = torch.cat((downout3, h))
        return hh


class MAHGCNWide(nn.Module):
    """Option C: each ROI carries k features. No diag trick. Fewer params."""
    def __init__(self, act, drop_p, layer, hidden_dim, degree_normalize=False):
        super().__init__()
        self.layer = layer
        k = hidden_dim

        # in_dim differs depending on whether a layer is the *first* called
        # (gets identity N×N input) or comes after a pool (gets [N, k]).
        down1_in = 1000
        down2_in = 400 if layer == 3 else k
        down3_in = 300 if layer == 2 else k
        down4_in = k

        self.net_gcn_down1 = GCN(down1_in, k, act, drop_p, degree_normalize)
        self.net_pool1 = AtlasMap(1000, 400, drop_p)
        self.net_gcn_down2 = GCN(down2_in, k, act, drop_p, degree_normalize)
        self.net_pool2 = AtlasMap(400, 300, drop_p)
        self.net_gcn_down3 = GCN(down3_in, k, act, drop_p, degree_normalize)
        self.net_pool3 = AtlasMap(300, 200, drop_p)
        self.net_gcn_down4 = GCN(down4_in, k, act, drop_p, degree_normalize)

        total_nodes = {2: 500, 3: 900, 4: 1900}[layer]
        self.output_dim = total_nodes * k

    def forward(self, g1, g2, g3, g4, h):
        if self.layer == 4:
            h = self.net_gcn_down1(g4, h); d1 = h
            h = self.net_pool1(h)
            h = self.net_gcn_down2(g3, h); d2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d1, d2, d3, h), dim=0)
        elif self.layer == 3:
            h = self.net_gcn_down2(g3, h); d2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d2, d3, h), dim=0)
        elif self.layer == 2:
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d3, h), dim=0)
        return hh.flatten()


class MAHGCNWideDiag(nn.Module):
    """Option B: k features per ROI + per-channel diag weighting (synthesis).
    Preserves the original inductive bias while expanding capacity."""
    def __init__(self, act, drop_p, layer, hidden_dim, degree_normalize=False):
        super().__init__()
        self.layer = layer
        k = hidden_dim

        # First GCN in the chain (gets identity input) stays as regular GCN.
        # Subsequent GCNs (get [N, k]) use GCNDiagMulti.
        if layer == 4:
            self.net_gcn_down1 = GCN(1000, k, act, drop_p, degree_normalize)
            self.net_gcn_down2 = GCNDiagMulti(400, k, k, drop_p, degree_normalize)
            self.net_gcn_down3 = GCNDiagMulti(300, k, k, drop_p, degree_normalize)
            self.net_gcn_down4 = GCNDiagMulti(200, k, k, drop_p, degree_normalize)
        elif layer == 3:
            self.net_gcn_down1 = GCN(1000, k, act, drop_p, degree_normalize)  # unused
            self.net_gcn_down2 = GCN(400, k, act, drop_p, degree_normalize)
            self.net_gcn_down3 = GCNDiagMulti(300, k, k, drop_p, degree_normalize)
            self.net_gcn_down4 = GCNDiagMulti(200, k, k, drop_p, degree_normalize)
        elif layer == 2:
            self.net_gcn_down1 = GCN(1000, k, act, drop_p, degree_normalize)  # unused
            self.net_gcn_down2 = GCN(400, k, act, drop_p, degree_normalize)   # unused
            self.net_gcn_down3 = GCN(300, k, act, drop_p, degree_normalize)
            self.net_gcn_down4 = GCNDiagMulti(200, k, k, drop_p, degree_normalize)

        self.net_pool1 = AtlasMap(1000, 400, drop_p)
        self.net_pool2 = AtlasMap(400, 300, drop_p)
        self.net_pool3 = AtlasMap(300, 200, drop_p)

        total_nodes = {2: 500, 3: 900, 4: 1900}[layer]
        self.output_dim = total_nodes * k

    def forward(self, g1, g2, g3, g4, h):
        # identical flow to MAHGCNWide
        if self.layer == 4:
            h = self.net_gcn_down1(g4, h); d1 = h
            h = self.net_pool1(h)
            h = self.net_gcn_down2(g3, h); d2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d1, d2, d3, h), dim=0)
        elif self.layer == 3:
            h = self.net_gcn_down2(g3, h); d2 = h
            h = self.net_pool2(h)
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d2, d3, h), dim=0)
        elif self.layer == 2:
            h = self.net_gcn_down3(g2, h); d3 = h
            h = self.net_pool3(h)
            h = self.net_gcn_down4(g1, h)
            hh = torch.cat((d3, h), dim=0)
        return hh.flatten()

class GATEdgeLayer(nn.Module):
    """Multi-head GAT with FC edge-weight term in attention.

    e_ij = LeakyReLU( a_src·z_i + a_dst·z_j + edge_w * fc_ij )
    alpha_ij = softmax_j(e_ij)
    h_i' = ELU( sum_j alpha_ij * z_j )   [concatenated across heads]

    a_src·z_i + a_dst·z_j is the standard decomposition of a^T[z_i||z_j]
    (avoids materialising the [N,N,H,2D] concat tensor — same result).
    edge_w is a learned per-head scalar on the FC value.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.W      = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst  = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.edge_w = nn.Parameter(torch.ones(num_heads))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.leaky = nn.LeakyReLU(0.2)
        self.act   = nn.ELU()

    def forward(self, g, h):
        # g: [B, N, N], h: [B, N, feat_dim]
        B, N, _ = h.shape
        z = self.W(h).view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D]
        e_src = (z * self.a_src).sum(-1)                          # [B, N, H]
        e_dst = (z * self.a_dst).sum(-1)                          # [B, N, H]
        e = self.leaky(
            e_src.unsqueeze(2) + e_dst.unsqueeze(1)               # [B, N, N, H]
            + g.unsqueeze(-1) * self.edge_w                        # [B, N, N, H]
        )
        e = e.masked_fill((g == 0).unsqueeze(-1), -1e9)
        alpha = self.drop(F.softmax(e, dim=2))                    # [B, N, N, H]
        out = torch.einsum('bijh,bjhd->bihd', alpha, z)           # [B, N, H, D]
        return self.act(out.reshape(B, N, -1))


class MAHGCNGATEdge(nn.Module):
    """Hierarchical multi-head GAT with edge-weight attention,
    following the same atlas-scale hierarchy as MAHGCNWide/WideDiag."""
    def __init__(self, act, drop_p, layer, hidden_dim, num_heads, num_gat_layers=1):
        super().__init__()
        self.layer = layer
        k = hidden_dim * num_heads   # total features per node

        down2_in = 400 if layer == 3 else k
        down3_in = 300 if layer == 2 else k

        self.net_gat_down1 = nn.ModuleList([GATEdgeLayer(1000 if i == 0 else k, k, num_heads, drop_p) for i in range(num_gat_layers)])
        self.net_gat_down2 = nn.ModuleList([GATEdgeLayer(down2_in if i == 0 else k, k, num_heads, drop_p) for i in range(num_gat_layers)])
        self.net_gat_down3 = nn.ModuleList([GATEdgeLayer(down3_in if i == 0 else k, k, num_heads, drop_p) for i in range(num_gat_layers)])
        self.net_gat_down4 = nn.ModuleList([GATEdgeLayer(k, k, num_heads, drop_p) for _ in range(num_gat_layers)])

        self.net_pool1 = AtlasMap(1000, 400, drop_p)
        self.net_pool2 = AtlasMap(400,  300, drop_p)
        self.net_pool3 = AtlasMap(300,  200, drop_p)

        total_nodes = {2: 500, 3: 900, 4: 1900}[layer]
        self.output_dim = total_nodes * k
        
    @staticmethod
    def forward_gat_layers(g, h, net_gat_layers):
        for gat_layer in net_gat_layers:
            h = gat_layer(g, h)
        
        return h

    def forward(self, g1, g2, g3, g4, h):
        if self.layer == 4:
            h = self.forward_gat_layers(g4, h, self.net_gat_down1); d1 = h
            h = self.net_pool1(h)
            h = self.forward_gat_layers(g3, h, self.net_gat_down2); d2 = h
            h = self.net_pool2(h)
            h = self.forward_gat_layers(g2, h, self.net_gat_down3); d3 = h
            h = self.net_pool3(h)
            h = self.forward_gat_layers(g1, h, self.net_gat_down4)
            hh = torch.cat((d1, d2, d3, h), dim=0)
        elif self.layer == 3:
            h = self.forward_gat_layers(g3, h, self.net_gat_down2); d2 = h
            h = self.net_pool2(h)
            h = self.forward_gat_layers(g2, h, self.net_gat_down3); d3 = h
            h = self.net_pool3(h)
            h = self.forward_gat_layers(g1, h, self.net_gat_down4)
            hh = torch.cat((d2, d3, h), dim=0)
        elif self.layer == 2:
            h = self.forward_gat_layers(g2, h, self.net_gat_down3); d3 = h
            h = self.net_pool3(h)
            h = self.forward_gat_layers(g1, h, self.net_gat_down4)
            hh = torch.cat((d3, h), dim=0)
        return hh.flatten()


def build_mahgcn(mode, act, drop_p, hidden_dim, num_heads=1, degree_normalize=False, num_gat_layers=1):
    """Factory: pick MAHGCN variant by config string."""
    if mode == "original":
        m = MAHGCN(act, drop_p, layer, degree_normalize)
        m.output_dim = {2: 500, 3: 900, 4: 1900}[layer]
        return m
    elif mode == "wide":
        return MAHGCNWide(act, drop_p, layer, hidden_dim, degree_normalize)
    elif mode == "wide_diag":
        return MAHGCNWideDiag(act, drop_p, layer, hidden_dim, degree_normalize)
    elif mode == "gat":
        return MAHGCNGATEdge(act, drop_p, layer, hidden_dim, num_heads, num_gat_layers)
    else:
        raise ValueError(f"Unknown gcn_mode: {mode!r}")

def build_single_res_fmri(mode, act, drop_p, ROInum, hidden_dim, num_heads=1, degree_normalize=False, num_gnn_layers=1):
    if mode == "gcn":
        return VanilleGCN(ROInum, hidden_dim, act, drop_p, degree_normalize, num_gnn_layers)
    if mode == "gat":
        return VanillaGAT(ROInum, hidden_dim, drop_p, num_heads, num_gnn_layers)
    raise ValueError(f"Unknown gcn_mode for single-resolution fMRI: {mode!r}")

class VanillaGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_p, num_heads, num_gat_layers=1):
        super().__init__()
        self.output_dim = hidden_dim * num_heads   # total features per node, concatenated heads

        self.net_gat_layers = nn.ModuleList([GATEdgeLayer(in_dim if i == 0 else self.output_dim, self.output_dim, num_heads, drop_p) for i in range(num_gat_layers)])
        
    @staticmethod
    def forward_gat_layers(g, h, net_gat_layers):
        for gat_layer in net_gat_layers:
            h = gat_layer(g, h)
        return h

    def forward(self, g_matrix, h):
        # g_matrix: [B, N, N], h: [B, N, feat_dim]
        h = self.forward_gat_layers(g_matrix, h, self.net_gat_layers)
        return h.mean(dim=-2)  # [B, output_dim]

class VanilleGCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p=0.3, degree_normalize=False, num_layers=1):
        super(VanilleGCN, self).__init__()
        self.proj = nn.ModuleList([nn.Linear(in_dim if i == 0 else out_dim, out_dim) for i in range(num_layers)])
        self.num_layers = num_layers
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.degree_normalize = degree_normalize
        self.output_dim = out_dim

    def forward(self, g, h):
        # g: [B, N, N], h: [B, N, feat_dim]
        if self.degree_normalize:
            deg = g.sum(dim=-1).clamp(min=1e-8).pow(-0.5)  # [B, N]
            g = deg.unsqueeze(-1) * g * deg.unsqueeze(-2)   # [B, N, N]
        for i in range(self.num_layers):
            h = self.drop(h)
            h = torch.bmm(g, h)   # [B, N, feat_dim]
            h = self.proj[i](h)   # [B, N, out_dim]
            h = self.act(h)
        return h.mean(dim=-2)     # [B, out_dim]

class AtlasMap(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = rf'{dir_path}/interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '.npy'
        Map = np.load(filename)
        #Map[Map<0.50] =0
        #Map[Map>= 0.50] = 1
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.T
        if h.dim() > 1 and h.shape[1] == 1:
            h = h.squeeze(-1)
        #h = torch.diag(h)
        return h


class AtlasMap_mean(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap_mean, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = './interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '.npy'
        Map = np.load(filename)
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        Map = Map / torch.sum(Map, axis=0)
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        return h

class AtlasMap_max(nn.Module):

    def __init__(self, indim, outdim, p):
        super(AtlasMap_max, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        dim=h.shape
        filename = './interlayermapping/mapping_' + str(self.indim) + 'to' + str(
            self.outdim) + '.npy'
        Map = np.load(filename)
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()

        h = h.T * Map
        h = torch.max(h, axis=0).values
        h = torch.reshape(h,(dim[0],self.outdim))
        h = h.T
        h = torch.squeeze(h)
        return h

class AtlasMap_th(nn.Module):

    def __init__(self, indim, outdim, p, th):
        super(AtlasMap_th, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.th=th

    def forward(self, h):
        #h = torch.diag(h)
        #h = self.drop(h)
        h = h.T
        filename = './interlayermapping/mapping_'+str(self.indim) +'to' + str(self.outdim)+ '.npy'
        Map = np.load(filename)
        Map[Map<self.th] =0
        Map[Map>= self.th] = 1
        Map = torch.tensor(Map)
        Map = Map.float()
        Map = Map.cuda()
        h = torch.matmul(h, Map)
        h = h.T
        h = torch.squeeze(h)
        #h = torch.diag(h)
        return h


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        #self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        #self.unpools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            #self.down_gcns.append(GCN(dim, dim, act, drop_p))
            #self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            #self.unpools.append(Unpool(dim, dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))
        #self.proj = nn.Linear(dim, out_dim)

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        #org_h = h
        #print(h.shape)
        #print(self.dim)
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            #adj_ms.append(g)
            h1=torch.cat([h1, h], dim=0)
            #indices_list.append(idx)

        #for hh in down_outs:
            #h = torch.cat([h, hh],dim=0)
        #print(h.shape)
        #for i in range(self.l_n):
        #    up_idx = self.l_n - i - 1
        #    g, idx = adj_ms[up_idx], indices_list[up_idx]
        #    g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
        #    h = self.up_gcns[i](g, h)
        #    h = h.add(down_outs[up_idx])
        #    hs.append(h)
        #h = h.add(org_h)
        #hs.append(h)
        #h = self.proj(h)
        return h1


class GraphDiif(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphDiif, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)

        self.pools.append(DiffPool(500, 400))
        self.pools.append(DiffPool(400, 300))
        self.pools.append(DiffPool(300, 200))
        self.pools.append(DiffPool(200, 100))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))

    def forward(self, g, h):
        hnext = self.top_gcn(g, h)
        h1 = hnext
        for i in range(self.l_n):
            g, h = self.pools[i](g, h, hnext)
            #print(g.shape)
            #print(h.shape)
            h=torch.diag(torch.squeeze(h))
            hnext = self.down_gcns[i](g, h)
            #print(hnext.shape)
            h1=torch.cat([h1, hnext], dim=0)

        return h1
    def Loss(self):
        L=0
        for i in range(self.l_n):
            L=L+self.pools[i].Loss()
        return L

class DiffPool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiffPool, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gcn=GCN(self.in_dim, self.out_dim, nn.ReLU())
        self.softmax = nn.Softmax()
        self.assign = torch.zeros(in_dim, out_dim)
        self.g = torch.zeros(in_dim, in_dim)

    def forward(self, g, h, hnext):
        self.g = g
        self.assign=self.gcn(g,h)
        #print(assign.shape)
        self.assign=self.softmax(self.assign)
        newh = torch.matmul(torch.transpose(self.assign, 0, 1), hnext)
        newg = torch.transpose(self.assign, 0, 1) @ g @ self.assign
        return newg, newh
    def Loss(self):
        loss_LP=torch.norm(torch.cat([self.g,torch.matmul(self.assign,torch.transpose(self.assign, 0, 1))]))/self.in_dim
        loss_en=0
        eps = 1e-7
        NPassign=F.relu(self.assign)
        for row in range(self.in_dim):
            loss_en=loss_en-torch.sum(NPassign[row,:]*torch.log(NPassign[row,:])+eps)
        loss_en=loss_en/self.in_dim
        return (loss_LP+loss_en)/self.in_dim

class GraphSAG(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphSAG, self).__init__()
        self.ks = ks
        self.top_gcn = GCN(in_dim, out_dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        #self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        #self.unpools = nn.ModuleList()
        self.dim=dim
        self.l_n = len(ks)
        for i in range(self.l_n):
            #self.down_gcns.append(GCN(dim, dim, act, drop_p))
            #self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(SAGPool(ks[i], dim, drop_p))
            #self.unpools.append(Unpool(dim, dim, drop_p))

        self.down_gcns.append(GCN(400, 1, act, drop_p))
        self.down_gcns.append(GCN(300, 1, act, drop_p))
        self.down_gcns.append(GCN(200, 1, act, drop_p))
        self.down_gcns.append(GCN(100, 1, act, drop_p))
        #self.proj = nn.Linear(dim, out_dim)

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        #org_h = h
        #print(h.shape)
        #print(self.dim)
        h = self.top_gcn(g, h)
        h1 = h
        for i in range(self.l_n):
            g, h, idx = self.pools[i](g, h)
            h = self.down_gcns[i](g, torch.diag(torch.squeeze(h)))
            #adj_ms.append(g)
            h1=torch.cat([h1, h], dim=0)
            #indices_list.append(idx)

        #for hh in down_outs:
            #h = torch.cat([h, hh],dim=0)
        #print(h.shape)
        #for i in range(self.l_n):
        #    up_idx = self.l_n - i - 1
        #    g, idx = adj_ms[up_idx], indices_list[up_idx]
        #    g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
        #    h = self.up_gcns[i](g, h)
        #    h = h.add(down_outs[up_idx])
        #    hs.append(h)
        #h = h.add(org_h)
        #hs.append(h)
        #h = self.proj(h)
        return h1

class SAGPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(SAGPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.in_dim = in_dim
        self.proj = GCN(self.in_dim, 1, nn.ReLU(), p)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(g, Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p=0.3, degree_normalize=False):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.degree_normalize = degree_normalize

    def forward(self, g, h):
        h = self.drop(h)
        if self.degree_normalize:
            deg = g.sum(dim=1).pow(-0.5)
            D_inv_sqrt = torch.diag(deg)
            g = D_inv_sqrt @ g @ D_inv_sqrt
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h
    

    
class GCNDiagMulti(nn.Module):
    """GCN that preserves per-source diag weighting per feature channel.
    out[i, c] = sum_j w[j, c] * g[i, j] * h[j, c]
    """
    def __init__(self, num_nodes, in_dim, out_dim, drop_p, degree_normalize=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.proj = nn.Linear(num_nodes * in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()
        self.degree_normalize = degree_normalize

    def forward(self, g, h):
        h = self.drop(h)
        if self.degree_normalize:
            deg = g.sum(dim=1).pow(-0.5)
            D = torch.diag(deg)
            g = D @ g @ D
        # weighted[i, j, c] = g[i, j] * h[j, c]
        weighted = g.unsqueeze(-1) * h.unsqueeze(0)        # [N, N, k]
        weighted = weighted.reshape(self.num_nodes, -1)     # [N, N*k]
        return self.act(self.proj(weighted))

class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g



    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
