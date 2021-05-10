## tf-x-transformers
[![PyPI version](https://badge.fury.io/py/tf-x-transformers.svg)](https://badge.fury.io/py/tf-x-transformers)

tf2.0 version of x-transformers

## Install
```bash
$ pip install tf-x-transformers
```
## Usage
Encoder-only (BERT-like)

```python
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tf_x_transformers import TransformerWrapper, Encoder

model = TransformerWrapper(num_tokens=20000,
                           max_seq_len=1024,
                           attn_layers=Encoder(dim=512, depth=12, heads=8))

x = tnp.random.randint(0, 256, (1, 1024))
mask = tf.cast(tf.ones_like(x), dtype=tf.bool)

output = model(x, mask=mask)  # (1, 1024, 20000)

print(output.shape)
```

## Compare
```python
compare.ipynb
mean difference tensor(5.0120e-07)
max difference tensor(1.3351e-05)
```

## Reference 
https://github.com/lucidrains/x-transformers
