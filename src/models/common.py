import torch
import torch.nn as nn
import MinkowskiEngine as ME


class MinkowskiLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape, eps, elementwise_affine,
        )

    def forward(self, x):
        if isinstance(x, ME.TensorField):
            return ME.TensorField(
                self.ln(x.F),
                coordinate_field_map_key=x.coordinate_field_map_key,
                coordinate_manager=x.coordinate_manager,
                quantization_mode=x.quantization_mode,
            )
        elif isinstance(x, ME.SparseTensor):
            return ME.SparseTensor(
                self.ln(x.F),
                cooridnate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
        else:
            return self.ln(x)

    def __repr__(self):
        s = "(normalized_shape={}, eps={}, elementwise_affine={})".format(
            self.ln.normalized_shape,
            self.ln.eps,
            self.ln.elementwise_affine
        )
        return self.__class__.__name__ + s


@torch.no_grad()
def downsample_points(points, tensor_map, field_map, size):
    down_points = ME.MinkowskiSPMMAverageFunction().apply(
        tensor_map, field_map, size, points
    )
    _, counts = torch.unique(tensor_map, return_counts=True)
    return down_points, counts.unsqueeze_(1).type_as(down_points)


@torch.no_grad()
def stride_centroids(points, counts, rows, cols, size):
    stride_centroids = ME.MinkowskiSPMMFunction().apply(rows, cols, counts, size, points)
    ones = torch.ones(size[1], dtype=points.dtype, device=points.device)
    stride_counts = ME.MinkowskiSPMMFunction().apply(rows, cols, ones, size, counts)
    stride_counts.clamp_(min=1)
    return torch.true_divide(stride_centroids, stride_counts), stride_counts


def downsample_embeddings(embeddings, inverse_map, size, mode="avg"):
    assert len(embeddings) == size[1]
    assert mode in ["avg", "max"]
    if mode == "max":
        in_map = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiDirectMaxPoolingFunction().apply(
            in_map, inverse_map, embeddings, size[0]
        )
    else:
        cols = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiSPMMAverageFunction().apply(
            inverse_map, cols, size, embeddings
        )
    return down_embeddings