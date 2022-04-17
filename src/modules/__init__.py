from .segmentation import LitSegMinkowskiModule

modules = [LitSegMinkowskiModule]
modules_dict = {m.__name__: m for m in modules}


def get_lightning_module(name):
    assert (
        name in modules_dict.keys()
    ), f"{name} not in {modules_dict.keys()}"
    return modules_dict[name]