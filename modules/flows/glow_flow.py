import torch
import torch.nn as nn
from modules.flows.flow import Flow
from overrides import overrides
import numpy as np
import math
from tools.utils import init_linear_layer
from scipy.stats import ortho_group
from scipy.linalg import lu
import torch.nn.functional as F

def logsumexp(x, dim=None):
    """
    Args:
    x: A pytorch tensor (any dimension will do)
    dim: int or None, over which to perform the summation. `None`, the
    default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))

class LinearFlow(Flow):
    def __init__(self, args):
        super(LinearFlow, self).__init__()

        self.args = args
        self.emb_dim = args.emb_dim

        self.W = nn.Parameter(torch.Tensor(self.emb_dim, self.emb_dim))
        nn.init.orthogonal_(self.W)

    @overrides
    def forward(self, x: torch.tensor):
        y, _ = self.backward(x, require_log_probs=False, forward=True)
        return y

    @overrides
    def backward(self, y: torch.tensor, require_log_probs=True, forward=False):
        if not forward:
            x_prime = y.mm(self.W)
        else:
            x_prime = y.mm(torch.inverse(self.W))

        if require_log_probs:
            _, log_abs_det = torch.slogdet(self.W)
            log_probs = log_abs_det
        else:
            log_probs = torch.tensor(0)
        return x_prime, log_probs

class SplitBlockFlow(Flow):
    def __init__(self, args):
        super(SplitBlockFlow, self).__init__()

        self.args = args
        self.emb_dim = args.emb_dim
        self.block_type = args.latent_glow_split_block_type
        self.n_layers = args.latent_glow_split_block_internal_layers
        self.hidden_dim = args.latent_glow_split_block_internal_dim
        self.min_gain = torch.tensor(args.latent_glow_split_min_gain)
        self.max_gain = torch.tensor(args.latent_glow_split_max_gain)


        self.W_hids, self.b_hids = [], []
        cur_in_dim = self.emb_dim
        for i in range(self.n_layers):
            cur_W = nn.Parameter(torch.Tensor(cur_in_dim, self.hidden_dim))
            nn.init.kaiming_uniform_(cur_W)
            self.register_parameter("W_hid_%s" % i, cur_W)
            self.W_hids.append(cur_W)
            cur_b = torch.zeros(self.hidden_dim)
            self.register_parameter("b_hid_%s" % i, cur_b)
            self.b_hids.append(cur_b)
            cur_in_dim = self.hidden_dim
        
        self.W_loc = nn.Parameter(torch.zeros(cur_in_dim, self.emb_dim))
        self.b_loc = nn.Parameter(torch.zeros(self.emb_dim))
        if self.block_type == "affine":
            self.gain_bias_init = torch.log((1-self.min_gain) / (self.max_gain-1))
            self.gain_range = self.max_gain - self.min_gain
            self.W_gain = nn.Parameter(torch.zeros(cur_in_dim, self.emb_dim))
            self.b_gain = nn.Parameter(torch.ones(self.emb_dim) * self.gain_bias_init)
        

    @overrides
    def forward(self, x: torch.tensor):
        y, _ = self.backward(x, require_log_probs=False, forward=True)
        return y

    @overrides
    def backward(self, y: torch.tensor, require_log_probs=True, forward=False):
        h = y
        for i in range(self.n_layers):
            h = h.mm(self.W_hids[i]) + self.b_bids[i]
            h = nn.functional.relu(h)
        loc = h.mm(self.W_loc) + self.b_loc
        if self.block_type == "affine":
            gain = h.mm(self.W_gain) + self.b_gain
            gain = nn.functional.sigmoid(gain) * self.gain_range + self.min_gain
            if not forward:
                x_prime = y * gain + loc
            else:
                x_prime = (y - loc) / gain
        else:
            if not forward:
                x_prime = y + loc
            else:
                x_prime = y - loc

        if require_log_probs and (self.block_type == "affine"):
            log_abs_det = torch.log(torch.abs(gain)).sum(dim=1)
            log_probs = log_abs_det
        else:
            log_probs = torch.tensor(0)
        return x_prime, log_probs


class GlowFlow_batch(Flow):
    def __init__(self, args):
        # freqs: np array of length ``size of word vocabulary''
        super(GlowFlow_batch, self).__init__()

        self.args = args

        self.emb_dim = args.emb_dim
        self.n_cond_blocks = args.latent_glow_cond_blocks
        self.n_shared_blocks = args.latent_glow_shared_blocks
        self.sd = torch.tensor(args.latent_glow_split_prior_sd)

        self.variance = self.sd * self.sd
        self.variance_matrix = torch.ones(1, self.emb_dim) * self.variance

        self.src_cond_blocks = nn.ModuleList()
        self.tgt_cond_blocks = nn.ModuleList()
        self.shared_blocks = nn.ModuleList()
        i = 0
        while i < self.n_cond_blocks:
            if (i % 2) == 0:
                src_cur_flow = LinearFlow(args)
                tgt_cur_flow = LinearFlow(args)
            else:
                src_cur_flow = SplitBlockFlow(args)
                tgt_cur_flow = SplitBlockFlow(args)
            self.src_cond_blocks.append(src_cur_flow)
            self.tgt_cond_blocks.append(tgt_cur_flow)
            i += 1

        while i < self.n_shared_blocks:
            if (i % 2) == 0:
                cur_flow = LinearFlow(args)
            else:
                cur_flow = SplitBlockFlow(args)
            self.shared_blocks.append(cur_flow)
            i += 1

        self.src_flow = GlowFlowAdaptor(self, "src")
        self.tgt_flow = GlowFlowAdaptor(self, "tgt")


    def cal_fix_var(self, x_prime):
        # x_prime: (batch_size, dim) -> transformed space

        def _l2_distance(x, y):
            # x: N, d, y: M, d
            # return N, M
            return torch.pow(x, 2).sum(1).unsqueeze(1) - 2 * torch.mm(x, y.t()) + torch.pow(y, 2).sum(1)

        x = torch.zeros(1, self.emb_dim)
        if self.args.init_var:
            log_det = torch.log(self.variance).sum()
            d = _l2_distance(x_prime, x) / self.variance.unsqueeze(0)
        else:
            log_det = self.emb_dim * math.log(self.variance)
            d = _l2_distance(x_prime, x) / self.variance

        logprob = -0.5 * (log_det + d + self.emb_dim * math.log(2 * math.pi)) # (batch_size, base_batch_size)
        # (batch_size, )
        logLL = logsumexp(logprob, dim=1)
        return logLL

    def clip_logvar(self, logvar):
        with torch.no_grad():
            logvar.clamp_(self.args.clip_min_log_var, self.args.clip_max_log_var)
            # logvar.clamp_(self.args.clip_min_log_var, 0.)

    def run_blocks(self, blocks, y: torch.tensor, require_log_probs=True, forward=False):
        x_prime, log_probs = y, torch.tensor(0)
        block_iter = reversed(blocks) if forward else blocks
        for block in block_iter:
            x_prime, partial_log_probs = block.backward(x_prime, require_log_probs, forward)
            if require_log_probs:
                log_probs = log_probs + partial_log_probs
        return x_prime, log_probs
    
    def run_src_cond_blocks(self, y: torch.tensor, require_log_probs=True, forward=False):
        return self.run_blocks(self.src_cond_blocks y, require_log_probs, forward)
    def run_tgt_cond_blocks(self, y: torch.tensor, require_log_probs=True, forward=False):
        return self.run_blocks(self.tgt_cond_blocks y, require_log_probs, forward)
    def run_shared_blocks(self, y: torch.tensor, require_log_probs=True, forward=False):
        return self.run_blocks(self.shared_blocks y, require_log_probs, forward)
    def run_src_to_tgt_cond_blocks(self, y: torch.tensor, require_log_probs=True, forward=False):
        x_mid, log_probs = self.run_blocks(self.src_cond_blocks y, require_log_probs, forward)
        x, new_log_probs = self.run_blocks(self.src_cond_blocks x_mid, require_log_probs, not forward)
        return x, log_probs - new_log_probs
    def run_tgt_to_src_cond_blocks(self, y: torch.tensor, require_log_probs=True, forward=False):
        x_mid, log_probs = self.run_blocks(self.tgt_cond_blocks y, require_log_probs, forward)
        x, new_log_probs = self.run_blocks(self.src_cond_blocks x_mid, require_log_probs, not forward)
        return x, log_probs - new_log_probs
    def run_src_to_lat(self, y: torch.tensor, require_log_probs=True, forward=False):
        if not forward:
            x_mid, log_probs = self.run_blocks(self.src_cond_blocks y, require_log_probs, False)
            x, new_log_probs = self.run_blocks(self.shared_blocks x_mid, require_log_probs, False)
        else:
            x_mid, log_probs = self.run_blocks(self.shared_blocks y, require_log_probs, True)
            x, new_log_probs = self.run_blocks(self.src_cond_blocks x_mid, require_log_probs, True)
        return x, log_probs + new_log_probs
    def run_tgt_to_lat(self, y: torch.tensor, require_log_probs=True, forward=False):
        if not forward:
            x_mid, log_probs = self.run_blocks(self.tgt_cond_blocks y, require_log_probs, False)
            x, new_log_probs = self.run_blocks(self.shared_blocks x_mid, require_log_probs, False)
        else:
            x_mid, log_probs = self.run_blocks(self.shared_blocks y, require_log_probs, True)
            x, new_log_probs = self.run_blocks(self.tgt_cond_blocks x_mid, require_log_probs, True)


    @overrides
    def forward(self, x: torch.tensor, from_src=True):
        y, _ = self.backward(x, require_log_probs=False, forward=True, from_src=True)
        return y

    @overrides
    def backward(self, y: torch.tensor, x: torch.tensor=None, x_freqs: torch.tensor=None, require_log_probs=True, var=None, y_freqs=None, forward=False, from_src=True):
        # from other language to this language

        x_prime, log_abs_det = self.run_src_to_lat(y, require_log_probs, forward) if from_src else self.run_tgtto_lat(y, require_log_probs, forward)

        if require_log_probs:
            x = torch.zeros(1, self.emd_dim)
            log_probs = self.cal_fix_var(x_prime)
            log_probs = log_probs + log_abs_det
        else:
            log_probs = torch.tensor(0)
        return x_prime, log_probs

class GlowFlowAdaptor(Flow):
    def __init__(self, cond_flow, direction):
        super(GlowFlowAdaptor, self).__init__()
        self.cond_flow = cond_flow
        self.from_src = (direction == "src")
    
    @overrides
    def forward(self, x: torch.tensor):
        return self.cond_flow.forward(x, from_src=self.from_src)

    @overrides
    def backward(self, y: torch.tensor, x: torch.tensor=None, x_freqs: torch.tensor=None, require_log_probs=True, var=None, y_freqs=None, forward=False):
        return self.cond_flow.backward(y, x, x_freqs:, require_log_probs, var, y_freqs, forward, from_src=self.from_src)

    @property W(self):
        first_cond_block = self.cond_flow.src_cond_blocks[0] if self.from_source, else self.cond_flow.tgt_cond_blocks[0]
        return first_cond_block.W

