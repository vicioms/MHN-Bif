# mostly from https://github.com/raywang4/EqM/blob/main/models.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    





# \omega_k = b^{-2*k/d_model} for k = 0, 1, ..., d_model/2 - 1
# emb_sin = sin(pos * \omega_k) and emb_cos = cos(pos * \omega_k)
def get_1d_pos_emb_from_grid(emb_dim, pos):
    assert emb_dim % 2 == 0, "Embedding dimension must be even for sinusoidal positional embeddings"
    omega = 1.0 / (10000 ** (2 * torch.arange(emb_dim // 2) / emb_dim))
    pos = pos.reshape(-1)
    pos_omega = torch.einsum('n,d->nd', pos, omega)
    emb_sin = pos_omega.sin()
    emb_cos = pos_omega.cos()
    emb = torch.cat([emb_cos, emb_sin], dim=1)
    return emb


def get_2d_pos_emb_from_grid(emb_dim, grid):
    assert emb_dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D sinusoidal positional embeddings"
    emb_h = get_1d_pos_emb_from_grid(emb_dim // 2, grid[0])
    emb_w = get_1d_pos_emb_from_grid(emb_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_2d_pos_emb(emb_dim, grid_size):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    pos_emb = get_2d_pos_emb_from_grid(emb_dim, grid)
    return pos_emb


class EqM(nn.Module):
    def __init__(self, 
                input_size,
                patch_size,
                in_channels,
                hidden_size, # set by config
                depth, # set by config
                num_heads, # set by config
                mlp_ratio, # set by config
                class_dropout_prob,
                num_classes,
                learn_sigma,
                ebm):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.x_embedder = PatchEmbed(img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.ebm = ebm

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_pos_emb(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    

    def forward(self, x0, y, return_act=False, get_energy=False, train=False):
        x0.requires_grad_(True)
        act = []
        x = self.x_embedder(x0) + self.pos_embed
        y = self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, y)                      # (N, T, D)
            act.append(x)
        x = self.final_layer(x, y)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        # explicit energy
        E=0
        if self.ebm == 'l2':
            E = -torch.sum(x**2, dim=(1,2,3))/2
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train, retain_graph=train)[0] 
        if self.ebm == 'dot':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train, retain_graph=train)[0]
        if self.ebm == 'mean':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train, retain_graph=train,)[0]           
        if get_energy:
            return x, -E
        if return_act: 
            return x, act
        return x
    
    def forward_with_cfg(self, x, y, cfg_scale, return_act=False, get_energy=False, train=False, cfg_to_all_channels=True):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, y, return_act=return_act, get_energy=get_energy, train=train)
        if get_energy:
            x, E = model_out
            model_out=x
        if return_act:
            act = model_out[1]
            model_out = model_out[0]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1), act
        

        # in the original version they use, for reproducibility:
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        # i.e. they apply classifier-free guidance to only three channels instead of all channels.
        # we just give the possibility here
    
        if cfg_to_all_channels:
            eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        else:
            eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
        return torch.cat([eps, rest], dim=1)



def EqM_XL_2(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_XL_4(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_XL_8(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_L_2(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_L_4(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_L_8(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_B_2(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=2, num_heads=12, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_B_4(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=4, num_heads=12, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_B_8(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=8, num_heads=12, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_S_1(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=1, num_heads=6, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_S_2(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=2, num_heads=6, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_S_4(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=4, num_heads=6, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)

def EqM_S_8(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=8, num_heads=6, mlp_ratio=4.0, class_dropout_prob=0.1, **kwargs)


EqM_models = {
    'EqM-XL/2': EqM_XL_2,  'EqM-XL/4': EqM_XL_4,  'EqM-XL/8': EqM_XL_8,
    'EqM-L/2':  EqM_L_2,   'EqM-L/4':  EqM_L_4,   'EqM-L/8':  EqM_L_8,
    'EqM-B/2':  EqM_B_2,   'EqM-B/4':  EqM_B_4,   'EqM-B/8':  EqM_B_8,
    'EqM-S/1':  EqM_S_1,   'EqM-S/2':  EqM_S_2,   'EqM-S/4':  EqM_S_4,   'EqM-S/8':  EqM_S_8,
}