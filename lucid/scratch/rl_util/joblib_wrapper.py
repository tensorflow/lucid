from lucid.misc.io.saving import nullcontext
from lucid.misc.io.loading import load_using_loader
from lucid.misc.io.writing import write_handle


def load_joblib(url_or_handle, *, cache=None, **kwargs):
    import joblib

    return load_using_loader(
        url_or_handle,
        decompressor=nullcontext,
        loader=joblib.load,
        cache=cache,
        **kwargs
    )


def save_joblib(value, url_or_handle, **kwargs):
    import joblib

    if hasattr(url_or_handle, "write") and hasattr(url_or_handle, "name"):
        joblib.dump(value, url_or_handle, **kwargs)
    else:
        with write_handle(url_or_handle) as handle:
            joblib.dump(value, handle, **kwargs)
