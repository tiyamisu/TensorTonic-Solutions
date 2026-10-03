import torch

def kda_recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, decay_logits: torch.Tensor, write_strength: torch.Tensor, output_gate_logits: torch.Tensor, output_projection: torch.Tensor, initial_state: torch.Tensor, g_min: float = -5.0, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing sequence outputs and the final recurrent state.
    """
    state = initial_state.clone()
    outputs = []
    for position in range(query.shape[1]):
        query_t = query[:, position]
        key_t = key[:, position]
        value_t = value[:, position]
        beta_t = write_strength[:, position]
        alpha_t = torch.exp(g_min * torch.sigmoid(decay_logits[:, position]))
        decayed = alpha_t.unsqueeze(-1) * state
        recalled = (key_t.unsqueeze(-1) * decayed).sum(dim=-2)
        erase = beta_t.unsqueeze(-1) * key_t.unsqueeze(-1) * recalled.unsqueeze(-2)
        write = beta_t.unsqueeze(-1) * key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
        state = decayed - erase + write
        read = (query_t.unsqueeze(-1) * state).sum(dim=-2)
        normalized = read / torch.sqrt(read.square().mean(dim=-1, keepdim=True) + eps)
        gated = torch.sigmoid(output_gate_logits[:, position]) * normalized
        merged = gated.reshape(gated.shape[0], -1)
        outputs.append(merged @ output_projection.transpose(0, 1))
    return {"outputs": torch.stack(outputs, dim=1), "final_state": state}
