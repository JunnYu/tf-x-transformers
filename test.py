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