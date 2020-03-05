import tensorflow as tf


def identity_grad(_op, grad):
    """Returns the incoming gradient unmodified."""
    return grad
