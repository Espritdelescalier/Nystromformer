import torch
import torch.nn as nn
import math
import einops


class CURAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.seq_len = config["max_seq_len"]

        self.index_select = None
        self.func_select = None
        self.submatrix_extraction = None
        self.absolute = False
        self.copy_rv = "copy_rv" in config
        self.select_number = config["select_number"] if "select_number" in config else 64
        self.select_type = config["select_type"] if "select_type" in config else "random"
        self.num_iter = config["num_iter"] if "num_iter" in config else 4
        self.cls_token = config["pooling_mode"] == "CLS"

        self.init_cur(config["select_mode"] if "select_mode" in config else "default")

    def init_cur(self, select_mode):

        if select_mode == 'same_k':
            self.index_select = self.index_select_same_on_k
        elif select_mode == 'same_q':
            self.index_select = self.index_select_same_on_q
        elif select_mode == 'inverted':
            self.index_select = self.index_select_different_inverted
        else:
            self.index_select = self.index_select_different

        self.submatrix_extraction = self.submatrix_extraction_rearrange

        if self.select_type == "abs":
            self.absolute = True

        if self.select_type == "step":
            self.func_select = self.step_selection
            self.submatrix_extraction = self.submatrix_extraction_step
        elif self.select_type == "topmin":
            self.func_select = self.top_min_k_sum_selection
        else:
            self.func_select = self.top_k_sum_selection

    def index_select_same_on_k(self, q, k, select_number, mask=None):
        index, index_shift = self.func_select(
            T=k, select_number=select_number, mask=mask)
        return index, index_shift, index, index_shift

    def index_select_same_on_q(self, q, k, select_number, mask=None):
        index, index_shift = self.func_select(
            T=q, select_number=select_number, mask=mask)
        return index, index_shift, index, index_shift

    def index_select_different(self, q, k, select_number, mask=None):
        index_k, index_shift_k = self.func_select(
            T=k, select_number=select_number, mask=mask)
        index_q, index_shift_q = self.func_select(
            T=q, select_number=select_number, mask=mask)
        return index_q, index_shift_q, index_k, index_shift_k

    def index_select_different_inverted(self, q, k, select_number, mask=None):
        index_k, index_shift_k = self.func_select(
            T=q, select_number=select_number, mask=mask)
        index_q, index_shift_q = self.func_select(
            T=k, select_number=select_number, mask=mask)
        return index_q, index_shift_q, index_k, index_shift_k

    def step_selection(self, T, select_number, mask=None):
        with torch.no_grad():
            B, H, N, D = T.shape

            if N < select_number:
                select_number = N

            pas = N // select_number

            imax = pas * select_number

            index = torch.arange(0, imax, pas, device=T.device)
            return index, index

    def top_k_sum_selection(self, T, select_number, mask=None):
        with torch.no_grad():
            B, H, N, D = T.shape
            device = T.device

            N2 = N

            if self.cls_token:
                T = T[:, :, 1:, :]
                N2 = N - 1
                if mask is not None:
                    mask = mask[:, 1:]

            if self.select_type == "embed":
                somme = T[:, :, :, 0]
            elif self.select_type == "random":
                somme = torch.rand(B, H, N2, device=device)
            else:
                somme = torch.sum((T.abs() if self.absolute else T), -1)

            if mask is not None:
                somme = somme.masked_fill(
                    ~mask[:, None, :].to(torch.bool), -torch.finfo(somme.dtype).max)
            index = torch.argsort(
                somme, dim=-1, descending=True)[:, :, :select_number]
            if self.cls_token:
                index = index + 1
                index[:, :, -1] = torch.tensor([0],
                                               device=device).expand_as(index[:, :, -1])
            # index = torch.topk(input=somme, k=select_number,dim=-1).indices
            # index, _ = torch.sort(index, -1)
            index_shift = einops.rearrange(index, 'b h n -> (b h n)')
            shift = torch.arange(0, B * H * N, N, device=device)
            shift = torch.repeat_interleave(shift, select_number)
            index_shift = index_shift + shift
            return index, index_shift

    def submatrix_extraction_rearrange(self, T, select_number, index_shift):
        B, H, N, D = T.shape
        nt = torch.index_select(
            einops.rearrange(T, 'b h n d -> (b h n) d'),
            0,
            index_shift
        )
        nt = einops.rearrange(nt, '(b h n) d -> b h n d',
                              b=B, h=H, n=select_number)
        return nt

    def submatrix_extraction_step(self, T, select_number, index_shift):

        nt = torch.index_select(
            T,
            2,
            index_shift
        )

        return nt

    def top_min_k_sum_selection(self, T, select_number, mask=None):
        # TODO REPLACE
        with torch.no_grad():
            B, H, N, D = T.shape
            device = T.device

            somme = torch.sum(T, -1)

            """if mask is not None:
                somme = somme.masked_fill(
                    ~mask[:, None, :].to(torch.bool), -torch.finfo(somme.dtype).max)"""

            index = torch.argsort(somme, dim=-1)
            index = torch.cat((index[:, :, :select_number // 2], index[:, :, -select_number // 2:]), dim=-1)

            index_shift = einops.rearrange(index, 'b h n -> (b h n)')
            shift = torch.arange(0, B * H * N, N, device=device)
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

    def forward(self, Q, K, V, mask):
        B, H, N, D = Q.shape
        Q = Q / math.sqrt(self.head_dim)

        index_q, shift_q, index_k, shift_k = self.index_select(Q, K, self.select_number, mask=mask)

        nc = self.submatrix_extraction(K, self.select_number, shift_k)
        nr = self.submatrix_extraction(Q, self.select_number, shift_q)

        c = Q @ nc.transpose(-1, -2)
        r = nr @ K.transpose(-1, -2)

        # print(torch.count_nonzero(mask, dim=1), mask.numel(), mask.shape)
        r = r.masked_fill(
            ~mask[:, None, None, :].to(torch.bool),
            -torch.finfo(Q.dtype).max
        )

        kernel_1 = torch.nn.functional.softmax(c, dim=-1)
        u = self.submatrix_extraction(kernel_1, self.select_number, shift_q)
        kernel_3 = torch.nn.functional.softmax(r, dim=-1)
        kernel_2_inv = self.iterative_inv(u)

        RV = torch.matmul(kernel_3, V)

        X = torch.matmul(kernel_1, torch.matmul(kernel_2_inv, RV))

        if self.copy_rv:
            shift = torch.arange(0, B * H * N * D, N * D, device=Q.device)
            shift = torch.repeat_interleave(shift, self.select_number * D)
            if self.select_type == 'step':
                index_shift = ((torch.repeat_interleave(index_q * D, D)
                                + torch.arange(D, device=Q.device)
                                .expand(index_q.numel(), D).flatten())
                               .expand(B * H, D * self.select_number).flatten() + shift)
            else:
                index_shift = (torch.repeat_interleave(index_q * D, D)
                               + torch.arange(D, device=Q.device).expand(index_q.numel(), D).flatten() + shift)

            X.put_(index_shift.reshape_as(RV), RV)

        return X

    def iterative_inv(self, mat):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(self.num_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_selectec_col_row={self.select_number}, seq_len={self.seq_len}'
