import torch

def init_t_xy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t.remainder(end_x)).float()
    t_y = t.div(end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_axial_cis(dim, end_x, end_y, theta = 100.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim))
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x, freqs_y = torch.outer(t_x, freqs), torch.outer(t_y, freqs)
    freqs_cis_x,freqs_cis_y = torch.polar(torch.ones_like(freqs_x), freqs_x), torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1).cuda()

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim > 1, "Input tensor x must have at least 2 dimensions."
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 3) + [x.shape[-3], x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[1], x.shape[-1]):
        shape = [1] * ndim
        shape[1] = x.shape[1]
        shape[-1] = x.shape[-1]
    else:
        raise ValueError("Shape of freqs_cis does not match x in any expected pattern.")
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_complex, xk_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)), torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_rotated, xk_rotated = torch.view_as_real(xq_complex * freqs_cis_broadcast).flatten(-2), torch.view_as_real(xk_complex * freqs_cis_broadcast).flatten(-2)
    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)
