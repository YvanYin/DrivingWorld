import torch
import torch.nn.functional as F

def top_k_sampling(logits, top_k=30, temperature_k=1.0):
    """
    Sampling from top-k.
    In LLM, lowering the temperature can reduce the semantic inconsistency problem in long-context generation.
    Args:
        logits: [B, 1, N], N is the codebook size.
    """
    assert len(logits.shape) == 3
    logits_topk = logits / temperature_k
    v, _ = torch.topk(logits, min(top_k, logits_topk.size(-1)))
    logits_topk[logits_topk < v[:, :, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits_topk, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs[:, 0, :], num_samples=1)
    return idx_next

def top_p_sampling(logits, top_p=0.9, temperature_p=1.0, filter_value=-float('Inf')):
    """
    Keep the top tokens with cumulative probability >= top_p (nucleus filtering), see https://arxiv.org/abs/1904.09751.
    In LLM, lowering the temperature can reduce the semantic inconsistency problem in long-context generation.
    Args:
        logits: [B, 1, N], N is the codebook size.
    """
    B, _, N = logits.shape
    logits_top_p = logits[:, 0, :] / temperature_p
    sorted_logits, sorted_indices = torch.sort(logits_top_p, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    for i in range(B):
        indices_to_remove = sorted_indices[i:i+1, ...][sorted_indices_to_remove[i:i+1, ...]]
        logits_top_p[i:i+1, indices_to_remove] = filter_value

    probabilities = F.softmax(logits_top_p, dim=-1)
    idx_next = torch.multinomial(probabilities, 1)
    return idx_next

def pk_sampling(logits, top_k=30, temperature_k=1.0, top_p=0.9, temperature_p=1.0, filter_value=-float('Inf')):
    assert len(logits.shape) == 3
    logits_topk = logits / temperature_k
    v, _ = torch.topk(logits, min(top_k, logits_topk.size(-1)))
    logits_topk[logits_topk < v[:, :, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    logits = F.softmax(logits_topk, dim=-1)
    
    B, _, N = logits.shape
    logits_top_p = logits[:, 0, :] / temperature_p
    sorted_logits, sorted_indices = torch.sort(logits_top_p, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    for i in range(B):
        indices_to_remove = sorted_indices[i:i+1, ...][sorted_indices_to_remove[i:i+1, ...]]
        logits_top_p[i:i+1, indices_to_remove] = filter_value

    probabilities = F.softmax(logits_top_p, dim=-1)
    idx_next = torch.multinomial(probabilities, 1)
    return idx_next
