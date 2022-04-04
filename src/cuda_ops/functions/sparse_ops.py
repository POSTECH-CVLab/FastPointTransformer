import torch
from torch.autograd import Function

import cuda_sparse_ops


class VectorAttention(Function):
  @staticmethod
  def forward(ctx, query, value, kernel, output_feature, kernel_map):
    assert (query.is_contiguous() and value.is_contiguous() and kernel.is_contiguous()
            and output_feature.is_contiguous())
    ctx.n, ctx.c = output_feature.shape
    ctx.k, ctx.s = kernel.shape

    in_out_arr = []
    with torch.no_grad():
      for k, in_out in kernel_map.items():
        in_out[0] = in_out[0] * ctx.k + k
        in_out_arr.append(in_out)

      in_out_arr = torch.cat(in_out_arr, -1)
      argsort = torch.argsort(in_out_arr[1])
      in_out_arr = in_out_arr[:, argsort]
      bin_cnt = torch.bincount(in_out_arr[1])
      in_cnt = torch.cumsum(bin_cnt, 0)

      in_out = in_out_arr[0].int().contiguous()
      in_cnt = in_cnt.int().contiguous()

    ctx.save_for_backward(query, value, kernel, in_out, in_cnt)
    cuda_sparse_ops.vector_attention_forward(ctx.n, ctx.c, ctx.k, ctx.s, in_out, in_cnt,
                                             query, value, kernel, output_feature)
    return output_feature

  @staticmethod
  def backward(ctx, grad_output):
    query, value, kernel, in_out, in_cnt = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_value = torch.zeros_like(value)
    grad_kernel = torch.zeros_like(kernel)
    cuda_sparse_ops.vector_attention_backward(ctx.n, ctx.c, ctx.k, ctx.s, in_out,
                                              in_cnt, query, value, kernel, grad_output,
                                              grad_query, grad_value, grad_kernel)
    return grad_query, grad_value, grad_kernel, None, None

vector_attention_cuda = VectorAttention.apply


class DotProductWithKey(Function):
  @staticmethod
  def forward(ctx, query, key, pos_enc, out_F, kq_map):
    assert (query.is_contiguous() and key.is_contiguous() and pos_enc.is_contiguous()
            and out_F.is_contiguous())
    ctx.m = kq_map.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.kkk = pos_enc.shape[0]
    ctx.save_for_backward(query, key, pos_enc, kq_map)
    cuda_sparse_ops.dot_product_with_key_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, query,
                                                 key, pos_enc, out_F, kq_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, key, pos_enc, kq_map = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_with_key_backward(ctx.m, ctx.h, ctx.kkk, ctx.c, query,
                                                  key, pos_enc, kq_map, grad_query,
                                                  grad_key, grad_pos, grad_out_F)
    return grad_query, grad_key, grad_pos, None, None

dot_product_with_key_cuda = DotProductWithKey.apply


class DecomposedDotProductWithKey(Function):
  @staticmethod
  def forward(ctx, query, key, pos_intra, pos_inter, out_F, kq_map):
    assert (query.is_contiguous() and key.is_contiguous()
            and pos_intra.is_contiguous() and pos_inter.is_contiguous()
            and out_F.is_contiguous())
    ctx.m = kq_map.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.kkk = pos_inter.shape[0]
    ctx.save_for_backward(query, key, pos_intra, pos_inter, kq_map)
    cuda_sparse_ops.decomposed_dot_product_with_key_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, query,
                                                 key, pos_intra, pos_inter, out_F, kq_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, key, pos_intra, pos_inter, kq_map = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_pos_intra = torch.zeros_like(pos_intra)
    grad_pos_inter = torch.zeros_like(pos_inter)
    cuda_sparse_ops.decomposed_dot_product_with_key_backward(ctx.m, ctx.h, ctx.kkk, ctx.c, query,
                                                  key, pos_intra, pos_inter, kq_map, grad_query,
                                                  grad_key, grad_pos_intra, grad_pos_inter, grad_out_F)
    return grad_query, grad_key, grad_pos_intra, grad_pos_inter, None, None

decomposed_dot_product_with_key_cuda = DecomposedDotProductWithKey.apply


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


class DotProductSample(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, qr_indices):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = qr_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, pos_enc, qr_indices)
    cuda_sparse_ops.dot_product_forward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                        out_F, qr_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, qr_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_backward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                         qr_indices, grad_query, grad_pos, grad_out_F)
    return grad_query, grad_pos, None, None

dot_product_sample_cuda = DotProductSample.apply


class DotProductKey(Function):
  @staticmethod
  def forward(ctx, key, pos_enc, out_F, k_map):
    assert (key.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = len(k_map)
    ctx.kkk, ctx.h, ctx.c = pos_enc.shape
    ctx.save_for_backward(key, pos_enc, k_map)
    cuda_sparse_ops.dot_product_key_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, key, pos_enc,
                                            out_F, k_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    key, pos_enc, k_map = ctx.saved_tensors
    grad_key = torch.zeros_like(key)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_key_backward(ctx.m, ctx.h, ctx.kkk, ctx.c, key, pos_enc,
                                             k_map, grad_key, grad_pos, grad_out_F)
    return grad_key, grad_pos, None, None

dot_product_key_cuda = DotProductKey.apply


class AddSumSquares(Function):
  @staticmethod
  def forward(ctx, ss_key, ss_pos, out_F, k_map):
    assert (ss_key.is_contiguous() and ss_pos.is_contiguous() and out_F.is_contiguous())
    ctx.m = len(k_map)
    ctx.kkk, ctx.h = ss_pos.shape
    ctx.save_for_backward(ss_key, ss_pos, k_map)
    cuda_sparse_ops.add_sum_squares_forward(ctx.m, ctx.h, ctx.kkk, ss_key, ss_pos,
                                            out_F, k_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    ss_key, ss_pos, k_map = ctx.saved_tensors
    grad_ss_key = torch.zeros_like(ss_key)
    grad_ss_pos = torch.zeros_like(ss_pos)
    cuda_sparse_ops.add_sum_squares_backward(ctx.m, ctx.h, ctx.kkk, ss_key, ss_pos,
                                             k_map, grad_ss_key, grad_ss_pos,
                                             grad_out_F)
    return grad_ss_key, grad_ss_pos, None, None

add_sum_squares_cuda = AddSumSquares.apply


class DotProductIntra(Function):
  @staticmethod
  def forward(ctx, pos_enc, out_F, kq_indices):
    assert (pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = pos_enc.shape
    ctx.save_for_backward(pos_enc, kq_indices)
    cuda_sparse_ops.dot_product_intra_forward(ctx.m, ctx.h, ctx.c, pos_enc, out_F,
                                              kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    pos_enc, kq_indices = ctx.saved_tensors
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_intra_backward(ctx.m, ctx.h, ctx.c, pos_enc, kq_indices,
                                               grad_pos, grad_out_F)
    return grad_pos, None, None

dot_product_intra_cuda = DotProductIntra.apply


class DotProductIntraInter(Function):
  @staticmethod
  def forward(ctx, query, intra_pos_enc, inter_pos_enc, out_F, kq_map):
    assert (query.is_contiguous() and intra_pos_enc.is_contiguous()
            and inter_pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_map.shape[1]
    ctx.kkk, ctx.h, ctx.c = inter_pos_enc.shape
    ctx.save_for_backward(query, intra_pos_enc, inter_pos_enc, kq_map)
    cuda_sparse_ops.dot_product_intra_inter_forward(ctx.m, ctx.h, ctx.kkk, ctx.c, query,
                                                    intra_pos_enc, inter_pos_enc, out_F,
                                                    kq_map)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, intra_pos_enc, inter_pos_enc, kq_map = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_intra_pos = torch.zeros_like(intra_pos_enc)
    grad_inter_pos = torch.zeros_like(inter_pos_enc)
    cuda_sparse_ops.dot_product_intra_inter_backward(ctx.m, ctx.h, ctx.kkk, ctx.c,
                                                     query, intra_pos_enc,
                                                     inter_pos_enc, kq_map, grad_query,
                                                     grad_intra_pos, grad_inter_pos,
                                                     grad_out_F)
    return grad_query, grad_intra_pos, grad_inter_pos, None, None

dot_product_intra_inter_cuda = DotProductIntraInter.apply


class DirectDotProduct(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, kq_indices):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, pos_enc, kq_indices)
    cuda_sparse_ops.direct_dot_product_forward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                               out_F, kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, kq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.direct_dot_product_backward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                                kq_indices, grad_query, grad_pos,
                                                grad_out_F)
    return grad_query, grad_pos, None, None

direct_dot_product_cuda = DirectDotProduct.apply


class DirectDotProductWithKey(Function):
  @staticmethod
  def forward(ctx, query, key, pos_enc, out_F, kq_indices):
    assert (query.is_contiguous() and key.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, key, pos_enc, kq_indices)
    cuda_sparse_ops.direct_dot_product_with_key_forward(ctx.m, ctx.h, ctx.c, query, key, pos_enc,
                                               out_F, kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, key, pos_enc, kq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.direct_dot_product_with_key_backward(ctx.m, ctx.h, ctx.c, query, key, pos_enc,
                                                kq_indices, grad_query, grad_key, grad_pos,
                                                grad_out_F)
    return grad_query, grad_key, grad_pos, None, None

direct_dot_product_with_key_cuda = DirectDotProductWithKey.apply


class DirectDotProductShared(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, kq_indices):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, pos_enc, kq_indices)
    cuda_sparse_ops.direct_dot_product_shared_forward(ctx.m, ctx.h, ctx.c, query,
                                                      pos_enc, out_F, kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, kq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.direct_dot_product_shared_backward(ctx.m, ctx.h, ctx.c, query,
                                                       pos_enc, kq_indices, grad_query,
                                                       grad_pos, grad_out_F)
    return grad_query, grad_pos, None, None

direct_dot_product_shared_cuda = DirectDotProductShared.apply


class DirectDotProductSharedWithKey(Function):
  @staticmethod
  def forward(ctx, query, key, pos_enc, out_F, kq_indices):
    assert (query.is_contiguous() and key.is_contiguous() and pos_enc.is_contiguous()
            and out_F.is_contiguous())
    ctx.m = kq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, key, pos_enc, kq_indices)
    cuda_sparse_ops.direct_dot_product_shared_with_key_forward(
        ctx.m, ctx.h, ctx.c, query, key, pos_enc, out_F, kq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, key, pos_enc, kq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.direct_dot_product_shared_with_key_backward(
        ctx.m, ctx.h, ctx.c, query, key, pos_enc, kq_indices, grad_query, grad_key,
        grad_pos, grad_out_F)
    return grad_query, grad_key, grad_pos, None, None

direct_dot_product_shared_with_key_cuda = DirectDotProductSharedWithKey.apply


class DotProductSampleShared(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, sq_indices):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = sq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, pos_enc, sq_indices)
    cuda_sparse_ops.dot_product_sample_shared_forward(ctx.m, ctx.h, ctx.c, query,
                                                      pos_enc, out_F, sq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, sq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_sample_shared_backward(ctx.m, ctx.h, ctx.c, query,
                                                       pos_enc, sq_indices, grad_query,
                                                       grad_pos, grad_out_F)
    return grad_query, grad_pos, None, None

dot_product_sample_shared_cuda = DotProductSampleShared.apply


class DotProductSample(Function):
  @staticmethod
  def forward(ctx, query, pos_enc, out_F, sq_indices):
    assert (query.is_contiguous() and pos_enc.is_contiguous() and out_F.is_contiguous())
    ctx.m = sq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, pos_enc, sq_indices)
    cuda_sparse_ops.dot_product_sample_forward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                               out_F, sq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, pos_enc, sq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_sample_backward(ctx.m, ctx.h, ctx.c, query, pos_enc,
                                                sq_indices, grad_query, grad_pos,
                                                grad_out_F)
    return grad_query, grad_pos, None, None

dot_product_sample_cuda = DotProductSample.apply


class DotProductSampleWithKey(Function):
  @staticmethod
  def forward(ctx, query, key, pos_enc, out_F, skq_indices):
    assert (query.is_contiguous() and key.is_contiguous() and pos_enc.is_contiguous()
            and out_F.is_contiguous())
    ctx.m = skq_indices.shape[1]
    _, ctx.h, ctx.c = query.shape
    ctx.save_for_backward(query, key, pos_enc, skq_indices)
    cuda_sparse_ops.dot_product_sample_with_key_forward(ctx.m, ctx.h, ctx.c, query, key,
                                                        pos_enc, out_F, skq_indices)
    return out_F

  @staticmethod
  def backward(ctx, grad_out_F):
    query, key, pos_enc, skq_indices = ctx.saved_tensors
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_pos = torch.zeros_like(pos_enc)
    cuda_sparse_ops.dot_product_sample_with_key_backward(ctx.m, ctx.h, ctx.c, query,
                                                         key, pos_enc, skq_indices,
                                                         grad_query, grad_key, grad_pos,
                                                         grad_out_F)
    return grad_query, grad_key, grad_pos, None, None

dot_product_sample_with_key_cuda = DotProductSampleWithKey.apply


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
