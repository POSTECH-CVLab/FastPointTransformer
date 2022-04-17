import torch
import MinkowskiEngine as ME


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