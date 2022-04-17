import logging

from src.models.spvcnn import SPVCNN
import src.models.resnet as resnets
import src.models.resunet as resunets
import src.models.fast_point_transformer as transformers

MODELS = [SPVCNN]


def add_models(module):
    MODELS.extend([getattr(module, a) for a in dir(module) if "Net" in a or "Transformer" in a])


add_models(resnets)
add_models(resunets)
add_models(transformers)


def get_model(name):
    """Creates and returns an instance of the model given its class name."""
    # Find the model class from its name
    all_models = MODELS
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        logging.info(f"Invalid model index. You put {name}. Options are:")
        # Display a list of valid model names
        for model in all_models:
            logging.info("\t* {}".format(model.__name__))
        return None
    model_class = mdict[name]
    return model_class