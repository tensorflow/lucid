import time
import pytest
import numpy as np
from lucid.misc.io.saving import save, CaptureSaveContext, batch_save
from lucid.misc.io.loading import load
from lucid.misc.io.scoping import io_scope, current_io_scopes
from concurrent.futures import ThreadPoolExecutor
import os.path
import io
import tensorflow as tf


dictionary = {"key": "value"}
dictionary_json = """{
  "key": "value"
}"""
array1 = np.eye(10, 10)
array2 = np.dstack([np.eye(10, 10, k=i - 1) for i in range(3)])


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def test_save_json():
    path = "./tests/fixtures/generated_outputs/dictionary.json"
    _remove(path)
    save(dictionary, path)
    assert os.path.isfile(path)
    content = io.open(path, "rt").read()
    assert content == dictionary_json


def test_save_npy():
    path = "./tests/fixtures/generated_outputs/array.npy"
    _remove(path)
    save(array1, path)
    assert os.path.isfile(path)
    re_read_array = np.load(path)
    assert np.array_equal(array1, re_read_array)


def test_save_npz_array():
    path = "./tests/fixtures/generated_outputs/arrays.npz"
    _remove(path)
    save([array1, array2], path)
    assert os.path.isfile(path)
    re_read_arrays = np.load(path)
    assert all(arr in re_read_arrays for arr in ("arr_0", "arr_1"))
    assert np.array_equal(array1, re_read_arrays["arr_0"])
    assert np.array_equal(array2, re_read_arrays["arr_1"])


def test_save_npz_dict():
    path = "./tests/fixtures/generated_outputs/arrays.npz"
    _remove(path)
    arrays = {"array1": array1, "array2": array2}
    save(arrays, path)
    assert os.path.isfile(path)
    re_read_arrays = np.load(path)
    assert all(arr in re_read_arrays for arr in list(arrays))
    assert np.array_equal(arrays["array1"], re_read_arrays["array1"])


def test_save_image_png():
    path = "./tests/fixtures/generated_outputs/rgbeye.png"
    _remove(path)
    save(array2, path)
    assert os.path.isfile(path)


def test_save_image_jpg():
    path = "./tests/fixtures/generated_outputs/rgbeye.jpg"
    _remove(path)
    save(array2, path)
    assert os.path.isfile(path)


def test_save_array_txt():
    path = "./tests/fixtures/generated_outputs/multiline.txt"
    _remove(path)
    stringarray = ["Line {:d}".format(i) for i in range(10)]
    save(stringarray, path)
    assert os.path.isfile(path)


def test_save_txt():
    path = "./tests/fixtures/generated_outputs/multiline.txt"
    _remove(path)
    string = "".join(["Line {:d}\n".format(i) for i in range(10)])
    save(string, path)
    assert os.path.isfile(path)


def test_save_named_handle():
    path = "./tests/fixtures/generated_outputs/rgbeye.jpg"
    _remove(path)
    with io.open(path, "wb") as handle:
        save(array2, handle)
    assert os.path.isfile(path)


def test_save_compressed_npy():
    uncompressed_path = "./tests/fixtures/generated_outputs/array.npy"
    _remove(uncompressed_path)
    save(array2, uncompressed_path)
    compressed_path = "./tests/fixtures/generated_outputs/array.npy.xz"
    _remove(compressed_path)
    save(array2, compressed_path)
    assert os.path.isfile(uncompressed_path)
    assert os.path.isfile(compressed_path)
    re_read_array = load(compressed_path)
    assert np.array_equal(array2, re_read_array)
    uncompressed_size = os.path.getsize(uncompressed_path)
    compressed_size = os.path.getsize(compressed_path)
    assert compressed_size < uncompressed_size


def test_save_load_pickle():
    path = "./tests/fixtures/generated_outputs/some_data.pickle"
    data = {
        'test': [1, 2, 3, "some string"],
        'numpy_values': array2
    }
    _remove(path)
    with io.open(path, "wb") as handle:
        with pytest.raises(ValueError):
            save(data, handle)
        save(data, handle, allow_unsafe_formats=True)
    assert os.path.isfile(path)
    with pytest.raises(ValueError):
        loaded_data = load(path)
    loaded_data = load(path, allow_unsafe_formats=True)
    assert loaded_data['test'] == data['test']
    assert np.array_equal(loaded_data['numpy_values'], data['numpy_values'])


def test_unknown_extension():
    with pytest.raises(ValueError):
        save({}, "test.unknown")


def test_unknown_compressor():
    with pytest.raises(ValueError):
        save(array2, "test.npy.gz")  # .gz is not currently supported, only xy


def test_save_protobuf():
    path = "./tests/fixtures/generated_outputs/graphdef.pb"
    _remove(path)
    with tf.Graph().as_default():
        a = tf.Variable(42)
        graphdef = a.graph.as_graph_def()
    save(graphdef, path)
    assert os.path.isfile(path)


def test_write_scope_compatibility():
    path = "./tests/fixtures/generated_outputs/write_scope_compatibility.txt"
    _remove(path)

    with io_scope("./tests/fixtures/generated_outputs"):
        save("test content", 'write_scope_compatibility.txt')

    assert os.path.isfile(path)


def test_capturing_saves():
    path = "./tests/fixtures/generated_outputs/test_capturing_saves.txt"
    _remove(path)
    context = CaptureSaveContext()

    with context, io_scope("./tests/fixtures/generated_outputs"):
        save("test", "test_capturing_saves.txt")

    captured = context.captured_saves
    assert len(captured) == 1
    assert "type" in captured[0]
    assert captured[0]["type"] == "txt"


def test_threadlocal_io_scopes():
    """ This tests that scopes are thread local and they don't clobber each other when different threads are competing"""
    def _return_io_scope(io_scope_path):
        with io_scope(io_scope_path):
            time.sleep(np.random.uniform(0.05, 0.1))
            return current_io_scopes()[-1]

    n_tasks = 16
    n_workers = 8
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_return_io_scope, f'gs://test-{i}'): f'gs://test-{i}' for i in range(n_tasks)}
        results = [f.result() for f in futures]
        assert results == list(futures.values())


def test_batch_saves():
    save_ops = [(str(i), f"write_batch_{i}.txt") for i in range(5)]
    [_remove(f"./tests/fixtures/generated_outputs/write_batch_{i}.txt") for i in range(5)]

    context = CaptureSaveContext()
    with context, io_scope("./tests/fixtures/generated_outputs"):
        results = batch_save(save_ops)
        assert len(results) == 5

    assert len(context.captured_saves) == 5
    assert context.captured_saves[0]['type'] == 'txt'
    print(context.captured_saves)
    assert 'write_batch_' in context.captured_saves[0]['url']
    assert all([os.path.isfile(f"./tests/fixtures/generated_outputs/write_batch_{i}.txt") for i in range(5)])
