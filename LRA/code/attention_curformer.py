
import torch
import torch.nn as nn
import math
import einops


class CURAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.select_number = config["select_number"]
        self.select_type = config["select_type"]
        self.seq_len = config["max_seq_len"]

        if "inv_coeff_init_option" in config:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

    def top_k_sum_selection(self, T, select_number, mask=None):
        B, H, N, D = T.shape
        device = T.device
        nt = torch.tensor((B, H, select_number, D), device=device)
        index = torch.tensor((B, H, select_number),
                             dtype=torch.long, device=device)
        # TODO clean le code
        if self.select_type == "embed":
            somme = T[:, :, 1:, 0]
        elif self.select_type == "random":
            somme = torch.rand(B, H, N-1, device=device)

        '''if mask is not None:
            somme = somme.masked_fill(~mask, -torch.finfo(somme.dtype).max)'''
        top = torch.topk(input=somme, k=select_number-1,
                         dim=-1).indices + 1
        top = torch.cat(
            (top, torch.zeros(B, H, 1, device=device).int()), dim=-1)
        index, _ = torch.sort(top, -1)
        index_shift = einops.rearrange(index, 'b h n -> (b h n)')
        shift = torch.arange(0, B*H*N, N, device=device)
        shift = torch.repeat_interleave(shift, select_number)
        index_shift = index_shift + shift
        nt = torch.index_select(
            einops.rearrange(T, 'b h n d -> (b h n) d'),
            0,
            index_shift
        )
        nt = einops.rearrange(nt, '(b h n) d -> b h n d',
                              b=B, h=H, n=select_number)
        return nt, index

    def m_matrix_composition(self, C, R_indexes):
        B, H, N, M = C.shape
        device = C.device
        nm = torch.tensor((B, H, M, M), device=device)
        index_shift = einops.rearrange(R_indexes, 'b h n -> (b h n)')
        shift = torch.arange(0, B*H*N, N, device=device)
        shift = torch.repeat_interleave(shift, M)
        index_shift = index_shift + shift
        nm = torch.index_select(
            einops.rearrange(C, 'b h n m -> (b h n) m'),
            0,
            index_shift
        )
        nm = einops.rearrange(
            nm, '(b h m1) m2 -> b h m1 m2', b=B, h=H, m1=M, m2=M)
        return nm

    def forward(self, Q, K, V, mask):

        #Q = Q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        #K = K * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        nc, c_index = self.top_k_sum_selection(
            T=K, select_number=self.select_number, mask=mask)
        nr, r_index = self.top_k_sum_selection(
            T=Q, select_number=self.select_number, mask=mask)
        c = Q @ nc.transpose(-1, -2)
        r = nr @ K.transpose(-1, -2)
        m = self.m_matrix_composition(c, r_index)

        kernel_1 = torch.nn.functional.softmax(
            c, dim=-1
        )
        kernel_2 = torch.nn.functional.softmax(
            m, dim=-1
        )
        kernel_3 = torch.nn.functional.softmax(
            r, dim=-1
        )
        kernel_2_inv = self.iterative_inv(kernel_2)
        X = torch.matmul(kernel_1, kernel_2_inv)
        X = torch.matmul(X, torch.matmul(kernel_3, V))

        return X

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-
                              1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV,
                             15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_selectec_col_row={self.select_number}, seq_len={self.seq_len}'
