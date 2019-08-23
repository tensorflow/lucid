import tensorflow as tf
import logging
from urllib import parse


log = logging.getLogger(__name__)


def tensor_by_name(deferred_tensor, graph=None):
    graph = graph or tf.get_default_graph()
    import_scope = deferred_tensor.model.import_scope
    tensor = graph.get_tensor_by_name(f"{import_scope}/{deferred_tensor.identifier}")
    return tensor


def conversion_func(value, dtype=None, name=None, as_ref=False):
    return value.t


class TensorCollection(dict):
    """Clean scope to add slug-named tensor properties.
    Decided against using __dir__ protocol as that's not used
    by Intellisense.
    """

    def add(self, dt):
        """Makes value accessible as both dict entry and as a property"""
        assert dt.slug not in self
        super().__setitem__(dt.slug, dt)
        super().__setitem__(dt.identifier, dt)
        self.__dict__[dt.slug] = dt


class DeferredTensor:
    def __init__(
        self, model, identifier, depth, shape, op_type, rank, ignored_prefix=""
    ):
        self.model = model
        self.identifier = identifier
        self.depth = depth
        self._shape = shape
        self.op_type = op_type
        self.rank = rank
        self.ignored_prefix = ignored_prefix
        self._slices = []

    def __str__(self):
        return self.slug

    def __repr__(self):
        return f"DeferredTensor <{self.model.name}:{self.slug}>"

    @property
    def shape(self):
        if self._slices:
            log.warn(
                f"Deferred Tensor {self.slug} has been sliced; shape is likely inaccurate!"
            )
            return None
        else:
            return self._shape

    @property
    def slug(self):
        slug = self.identifier
        while slug.startswith(self.ignored_prefix):
            slug = slug[len(self.ignored_prefix) :].strip("/")
        if slug.endswith(":0"):
            shortened = slug[:-2]
        unslashed = shortened.replace("/", "_")
        uncoloned = unslashed.replace(":", "__")
        lowered = uncoloned.lower()
        return parse.quote(lowered, safe="_")

    @property
    def t(self, graph=None):
        tensor = tensor_by_name(self, graph=graph)
        for slice in self._slices:
            tensor = tensor[slice]
        return tensor

    def channel(self, index):
        if index >= self.depth:
            raise IndexError(
                f"{self} has a depth of {self.depth}, but channel {index} was requested."
            )
        else:
            return self[..., index]

    def __getitem__(self, slice):
        self._slices.append(slice)
        return self


tf.register_tensor_conversion_function(DeferredTensor, conversion_func)
