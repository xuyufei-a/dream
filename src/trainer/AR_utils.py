import torch
from typing import Literal

def create_temporal_mask(index: torch.tensor):
    """
    Create a temporal mask for sequence modeling. Noticed mask token may share the same index, but they are not supposed to attend to each other.

    Args:
        index (torch.tensor): A 2D tensor of shape (batch_size, seq_len) representing the temporal indices. 

    Returns:
        torch.tensor: A 3D tensor of shape (batch_size, seq_len, seq_len) where each element (b, i, j) is True if j is temporally preceding to i or j == i, else False.
    """
    mask = index.unsqueeze(1) < index.unsqueeze(2)
    mask = torch.logical_or(mask, torch.eye(index.size(1), device=index.device).unsqueeze(0).bool()).unsqueeze(1)

    return mask

def create_temporal_index(input_ids: torch.tensor, attention_mask: torch.tensor, mask_token_id: int, bos_token_id: int=None):
    """
    Create a temporal index for sequence modeling. Each position is assigned an index based on its order in the sequence, with masked tokens sharing the same index.

    Args:
        input_ids (torch.tensor): A 2D tensor of shape (batch_size, seq_len) representing the input token IDs.
        attention_mask (torch.tensor): A 2D tensor of shape (batch_size, seq_len) representing the attention mask.
        mask_token_id (int): The token ID used for masking.

    Returns:
        torch.tensor: A 2D tensor of shape (batch_size, seq_len) where each element represents the temporal index of the corresponding token in the input_ids.
    """
    batch_size, seq_len = input_ids.size()
    temporal_index = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)

    input_pos = torch.logical_and(attention_mask.bool(), input_ids != mask_token_id)
    if bos_token_id is not None:
        bos_pos = input_ids == bos_token_id
        input_pos = torch.logical_and(input_pos, ~bos_pos) 

    for b in range(batch_size):
        input_sum = input_pos[b].sum().item()
        re_permutation = torch.randperm(input_sum, device=input_ids.device)
        temporal_index[b][input_pos[b]] = re_permutation
        temporal_index[b][~input_pos[b]] = input_sum + 1

        if bos_token_id is not None:
            temporal_index[b][bos_pos[b]] = -1

        # TODO unattended padding token are regarded as mask token
        # but we assume their loss are masked out during training

    return temporal_index

def create_bi_mask(input_ids: torch.tensor, attention_mask: torch.tensor, mask_token_id: int, L: int):
    """
    create bi-directional mask for masked tokens in blocks of length L
    """
    is_mask = input_ids == mask_token_id
    is_mask_cumsum = torch.cumsum(is_mask.int(), dim=-1) - 1

    gid = (is_mask_cumsum // L + 1) * is_mask
    gid_q = gid.unsqueeze(-1)
    gid_k = gid.unsqueeze(-2)

    group_bi_mask = (gid_q == gid_k) & (gid_q > 0)

    return group_bi_mask

def create_temporal_mask_wrapper(input_ids: torch.tensor, attention_mask: torch.tensor, mask_token_id: int, bos_token_id: int=None, enable_c2c_mask: bool=False, enable_c2m_mask: bool=False, enable_m2m_mask: bool=False, return_index: bool=False, temporal_index: torch.tensor=None, enable_pad_diagnal: bool=True, block_length: int=1):
    """
    a wrapper function to create temporal causal mask and its variants
    """
    # a specific optimization for c2c mask only
    assert enable_c2c_mask and not enable_m2m_mask
    if enable_c2c_mask and not enable_m2m_mask:
        input_pos = torch.logical_and(attention_mask.bool(), input_ids != mask_token_id)
        temporal_attention_mask = torch.logical_and(attention_mask.unsqueeze(-1), input_pos.unsqueeze(-2))

        if enable_c2m_mask:
            temporal_attention_mask = torch.logical_or(temporal_attention_mask, temporal_attention_mask.permute(0, 2, 1))
        if block_length > 1:
            group_bi_mask = create_bi_mask(input_ids, attention_mask, mask_token_id, block_length)
            temporal_attention_mask = torch.logical_or(temporal_attention_mask, group_bi_mask)

        temporal_attention_mask = torch.logical_or(temporal_attention_mask, torch.eye(input_ids.size(1), device=input_ids.device).unsqueeze(0).bool()).unsqueeze(1)
        if not enable_pad_diagnal:
            temporal_attention_mask = torch.logical_and(temporal_attention_mask,
                                                    torch.logical_and(
                                                        attention_mask.unsqueeze(1).unsqueeze(-2),
                                                        attention_mask.unsqueeze(1).unsqueeze(-1),
                                                    ))
            # if padding token didn't attend to any token, may cause issue for sdpa. see https://github.com/pytorch/pytorch/issues/110213

        if return_index:
            return temporal_attention_mask, None
        else:
            return temporal_attention_mask

    if temporal_index is None:
        temporal_index = create_temporal_index(input_ids, attention_mask, mask_token_id, bos_token_id)
    temporal_attention_mask = create_temporal_mask(temporal_index)

    if enable_c2c_mask or enable_c2m_mask or enable_m2m_mask:
        c_pos = torch.logical_and(attention_mask.bool(), input_ids != mask_token_id).unsqueeze(1)
        m_pos = torch.logical_and(attention_mask.bool(), input_ids == mask_token_id).unsqueeze(1)

        # unsqueeze(-1) for row and unsqueeze(-2) for column
        if enable_c2c_mask:
            c2c_mask = torch.logical_and(c_pos.unsqueeze(-1), c_pos.unsqueeze(-2))
            temporal_attention_mask = torch.logical_or(temporal_attention_mask, c2c_mask)
        
        if enable_c2m_mask:
            c2m_mask = torch.logical_and(c_pos.unsqueeze(-1), m_pos.unsqueeze(-2))
            temporal_attention_mask = torch.logical_or(temporal_attention_mask, c2m_mask)

        if enable_m2m_mask:
            m2m_mask = torch.logical_and(m_pos.unsqueeze(-1), m_pos.unsqueeze(-2))
            temporal_attention_mask = torch.logical_or(temporal_attention_mask, m2m_mask)

    if return_index:
        return temporal_attention_mask, temporal_index
    else:
        return temporal_attention_mask

if __name__ == "__main__":
    input_ids = torch.tensor([[-1, -1,2,3,-1,-1,0,0],[1,2,3,-1,-1,-1,-1, -1]])
    attention_mask = torch.tensor([[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1]])

    mask_token_id = -1

    t1 = create_temporal_mask_wrapper(input_ids, attention_mask, mask_token_id, enable_c2c_mask=True, enable_m2m_mask=False, enable_c2m_mask=False, block_length=1)
    t2 = create_temporal_mask_wrapper(input_ids, attention_mask, mask_token_id, enable_c2c_mask=True, enable_m2m_mask=False, enable_c2m_mask=True, block_length=1)
    t3 = create_temporal_mask_wrapper(input_ids, attention_mask, mask_token_id, enable_c2c_mask=True, enable_m2m_mask=False, enable_c2m_mask=False, block_length=2)
    t4 = create_temporal_mask_wrapper(input_ids, attention_mask, mask_token_id, enable_c2c_mask=True, enable_m2m_mask=False, enable_c2m_mask=True, block_length=2)

    print(t1)
    print(t2)
    print(t3)
    print(t4)