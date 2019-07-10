# lucid/misc/io

These modules' names end on 'ing' so the three main public methods can stay as
simple verbs ('load', 'save', and 'show') without creating namespace conflicts
if you do need to import the module or some of the lower level methods.

## API

### load

### show

### save

Used to write files. File types are inferred from the file extension given in the destination string.

```python
from lucid.misc.io import save

save(np.random.rand(100), "random.npy")
```

Can save locally or to GCS.

```python
save(random, "gs://bucket/test/random.npy")
```

Gracefully supports the following file extensions: 

```
.png
.jpg
.jpeg
.npy
.npz
.json
.txt
.pb
```

### write_handle

If you want to save a filetype that is not supported, you can import a write handler from `lucid.misc.io.writing` and pass that to most other save implementations. Here's an example with saving a matplotlib plot.

```python
from lucid.misc.io.writing import write_handle

r = np.random.rand(1000, 1000)
plt.figure(figsize=(10, 10))
plt.scatter(x=r[:,1],y=r[:,0], s=2)
plt.show
with write_handle("test2.png") as handle:
    plt.savefig(handle)

```

### io_scope

You can scope a bunch of save/load calls to a destination.

```python
from lucid.misc.io import save, io_scope

with io_scope("gs://bucket/test/folder/"):
    for op in ops:
        save(op, "op.npy")
```
