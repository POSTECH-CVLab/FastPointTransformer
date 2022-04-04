import gin


@gin.configurable
def logged_hparams(keys):
    C = dict()
    for k in keys:
        C[k] = gin.query_parameter(f"{k}")
    return C


def load_from_pl_state_dict(model, pl_state_dict):
    state_dict = {}
    for k, v in pl_state_dict.items():
        state_dict[k[6:]] = v
    model.load_state_dict(state_dict)
    return model