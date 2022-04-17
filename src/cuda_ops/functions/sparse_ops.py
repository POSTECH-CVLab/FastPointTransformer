import torch
from torch.autograd import Function

import cuda_sparse_ops


class DotProduct(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, kq_map):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_map.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.kkk = pos_enc.shape[0]
    ctx.save_for_backward(query, pos_enc, kq_map)
    cuda_sparse_ops.dot_product_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc,
                                        out_F, kq_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, kq_map = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_backward(ctx.m, ctx.h, ctx.kkk, ctx.c, query, pos_enc,
                                         kq_map, grad_query, grad_pos, grad_out_F)
    return grad_query, grad_pos, None, None

dot_product_cuda = DotProduct.apply


class ScalarAttention(Function):
  @staticmethod
  def forward(ctx, weight, value, out_F, kq_indices):
    assert (weight.is_contiguous() and value.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = value.shape
    ctx.save_for_backward(weight, value, kq_indices)
    cuda_sparse_ops.scalar_attention_forward(ctx.m, ctx.h, ctx.c, weight, value, out_F,
                                             kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    weight, value, kq_indices = ctx.saved_tensors
    grad_weight = torch.zeros_like(weight)
    grad_value = torch.zeros_like(value)
    cuda_sparse_ops.scalar_attention_backward(ctx.m, ctx.h, ctx.c, weight, value,
                                              kq_indices, grad_weight, grad_value,
                                              grad_out_F)
    return grad_weight, grad_value, None, None

scalar_attention_cuda = ScalarAttention.apply
