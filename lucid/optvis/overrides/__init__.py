"""Override gradient implementations easily.

Example usages:

Fully default overrides
```
  from lucid.optvis import overrides
  #...
  with overrides.standard_overrides():
      T = render.import_model(model, param_t, param_t)
  #...use imported graph
```

Specific subsets of overrides
```
  from lucid.optvis import overrides
  with overrides.gradient_override_map(overrides.pooling_overrides_map):
    T = render.import_model(model, param_t, param_t)
  #...use imported graph
```

Custom overrides
```
  from lucid.optvis import overrides
  with overrides.gradient_override_map({"MaxPool": overrides.avg_smoothed_maxpool_grad}):
    T = render.import_model(model, param_t, param_t)
  #...use imported graph
```

"""
from lucid.optvis.overrides.gradient_override import gradient_override_map, use_gradient
from lucid.optvis.overrides.identity_grad import identity_grad
from lucid.optvis.overrides.redirected_relu_grad import (
    redirected_relu_grad,
    redirected_relu6_grad,
)
from lucid.optvis.overrides.smoothed_maxpool_grad import avg_smoothed_maxpool_grad

pooling_overrides_map = {"MaxPool": avg_smoothed_maxpool_grad}

relu_overrides_map = {"Relu": redirected_relu_grad, "Relu6": redirected_relu6_grad}

default_overrides_map = {**pooling_overrides_map, **relu_overrides_map}


def relu_overrides():
    return gradient_override_map(relu_overrides_map)


def pooling_overrides():
    return gradient_override_map(pooling_overrides_map)


def default_overrides():
    return gradient_override_map(default_overrides_map)


def linearization_overrides():
    return gradient_override_map({"Relu": identity_grad, **pooling_overrides_map})

