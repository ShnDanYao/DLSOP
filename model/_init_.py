from .oracle import ORACLE
from .sbilinear import sBCNN

def set_module(model_name,model_params):
    return globals()[model_name](**model_params)


def get_module(model_name):
    return globals()[model_name]
