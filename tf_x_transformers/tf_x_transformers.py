from tf_fast_api import *
from tf_x_transformers.tf_entmax import entmax15
from tf_x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import math
from functools import partial
from inspect import isfunction
from collections import namedtuple
from einops import rearrange, repeat

# K.set_learning_phase(1)

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates',
                           ['pre_softmax_attn', 'post_softmax_attn'])

LayerIntermediates = namedtuple('Intermediates',
                                ['hiddens', 'attn_intermediates'])

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def not_equals(val):
    def inner(x):
        return x != val

    return inner


def equals(val):
    def inner(x):
        return x == val

    return inner


def max_neg_value(tensor):
    return -tnp.finfo(tensor.dtype).max


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val, )


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix):], x[1]),
            tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def get_initializer(initializer_range: float = 0.02):
    return keras.initializers.TruncatedNormal(stddev=initializer_range)


def glu(tensor):
    assert tensor.size(-1) % 2 == 0, "tensor.size(-1) % 2 must be 0"
    x, gate = tensor.chunk(2, axis=-1)
    return x * gate.sigmoid()


class Identity(layers.Layer):
    def call(self, inputs):
        return inputs


class DepthWiseConv1d(layers.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel_size,
                 padding=0,
                 stride=1,
                 bias=True,
                 groups=False):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = keras.Sequential([
            layers.Conv1D(filters=dim_in,
                          kernel_size=kernel_size,
                          padding=padding,
                          groups=dim_in,
                          stride=stride,
                          data_format="channels_first",
                          use_bias=bias),
            layers.Conv1D(filters=dim_out, kernel_size=1)
        ])

    def call(self, x):
        return self.net(x)


class AbsolutePositionalEmbedding(layers.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = layers.Embedding(
            max_seq_len, dim, embeddings_initializer=get_initializer(0.02))

    def call(self, x):
        n = tf.range(x.size(1))
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(layers.Layer):
    def __init__(self, dim, sin_cos_cross=False):
        super().__init__()
        self.dim = dim
        self.sin_cos_cross = sin_cos_cross

    def call(self, inputs, seq_dim=1, offset=0):
        inv_freq = 10000**(-tf.range(0, self.dim, 2, dtype=K.floatx()) /
                           self.dim)
        position_ids = tf.range(inputs.size(seq_dim),
                                dtype=K.floatx()) + offset
        sinusoid_inp = tf.einsum('n,d->nd', position_ids, inv_freq)
        if self.sin_cos_cross:
            embed = tf.stack(
                [sinusoid_inp.sin(), sinusoid_inp.cos()],
                axis=-1).flatten(1, 2)
        else:
            embed = tf.concat(
                [sinusoid_inp.sin(), sinusoid_inp.cos()], axis=-1)

        return embed[None, :, :]


class RelativePositionBias(layers.Layer):
    def __init__(
        self,
        scale,
        causal=False,
        num_buckets=32,
        heads=8,
        max_distance=128,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = layers.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  causal=True,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).astype(n.dtype) * num_buckets
            n = n.abs()
        else:
            n = n.clamp(min=0)  #tf.maximum(n, tf.zeros_like(n))

        max_exact = num_buckets // 2

        is_small = n < max_exact

        val_if_large = max_exact + (tf.math.log(n / max_exact) /
                                    math.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).astype(n.dtype)

        # val_if_large = tf.minimum(val_if_large, num_buckets - 1)
        val_if_large = val_if_large.clamp(max=num_buckets - 1)

        ret += tf.where(is_small, n, val_if_large)
        return ret

    def call(self, qk_dots):
        i, j = qk_dots.shape[-2:]
        q_pos = tf.range(i)
        k_pos = tf.range(j)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rel_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        values = self.relative_attention_bias(rel_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class RotaryEmbedding(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, max_seq_len):
        inv_freq = 1. / (10000**(tf.range(0, self.dim, 2, dtype=K.floatx()) /
                                 self.dim))
        t = tf.range(max_seq_len, dtype=K.floatx())
        freqs = tf.einsum('i , j -> i j', t, inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        return rearrange(emb, 'n d -> () () n d')


def rotate_half(x):
    x1, x2 = x.chunk(2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.size(-2)
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# classes


class Scale(layers.Layer):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def call(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(layers.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = self.add_weight(shape=(1, ), initializer='zeros', name='g')

    def call(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.g = self.add_weight(shape=(1, ), initializer='ones', name='g')

    def build(self, input_shape):
        self.scale = input_shape[self.axis]**-0.5
        return super().build(input_shape)

    def call(self, x):
        norm = x.norm(axis=-1, keepdims=True) * self.scale
        return x / norm.clamp(min=self.epsilon) * self.g


class RMSNorm(layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-8):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.g = self.add_weight(shape=(1, ), initializer='ones', name='g')

    def build(self, input_shape):
        self.scale = input_shape[self.axis]**-0.5
        return super().build(input_shape)

    def call(self, x):
        norm = x.norm(axis=-1, keepdims=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(layers.Layer):
    def call(self, x, residual):
        return x + residual


class GRUGating(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gru = layers.GRUCell(dim)

    def call(self, x, residual):
        gated_output, _ = self.gru(x, residual)
        return gated_output.reshape_as(x)


# feedforward


class GEGLU(layers.Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = layers.Dense(dim_out * 2)

    def call(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * tf.nn.gelu(gate)


class FeedForward(layers.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = layers.Dense(
            inner_dim, activation=tf.nn.gelu) if not glu else GEGLU(inner_dim)

        self.net = keras.Sequential(
            [project_in,
             layers.Dropout(dropout),
             layers.Dense(dim_out)])

    def call(self, x):
        return self.net(x)


# attention.


class Attention(layers.Layer):
    def __init__(self,
                 dim,
                 dim_head=DEFAULT_DIM_HEAD,
                 heads=8,
                 causal=False,
                 mask=None,
                 talking_heads=False,
                 sparse_topk=None,
                 use_entmax15=False,
                 num_mem_kv=0,
                 dropout=0.,
                 on_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads

        self.to_q = layers.Dense(inner_dim, use_bias=False)
        self.to_k = layers.Dense(inner_dim, use_bias=False)
        self.to_v = layers.Dense(inner_dim, use_bias=False)
        self.dropout = layers.Dropout(dropout)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = self.add_weight(shape=(heads, heads),
                                                    initializer="normal",
                                                    name="pre_softmax_proj")
            self.post_softmax_proj = self.add_weight(shape=(heads, heads),
                                                     initializer="normal",
                                                     name="post_softmax_proj")

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else tf.nn.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = self.add_weight(shape=(heads, num_mem_kv, dim_head),
                                         initializer="normal",
                                         name="mem_k")
            self.mem_v = self.add_weight(shape=(heads, num_mem_kv, dim_head),
                                         initializer="normal",
                                         name="mem_v")

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = layers.Dense(
            dim * 2, activation=glu) if on_attn else layers.Dense(dim)

    def call(self,
             x,
             context=None,
             mask=None,
             context_mask=None,
             rel_pos=None,
             sinusoidal_emb=None,
             rotary_pos_emb=None,
             prev_attn=None,
             mem=None):
        b, n, _, h, talking_heads, has_context = *x.size(
        ), self.heads, self.talking_heads, exists(context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = tf.concat((mem, k_input), axis=-2)
            v_input = tf.concat((mem, v_input), axis=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.size(-2) - q_input.size(-2)
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.size(-1)
            (ql, qr), (kl, kr) = map(lambda t: (t[..., :l], t[..., l:]),
                                     (q, k))
            ql, kl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb),
                         (ql, kl))
            q = tf.concat((ql, qr), axis=-1)
            k = tf.concat((kl, kr), axis=-1)

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: tf.ones((b, n)).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: tf.ones((b, k.size(-1))).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask & k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b),
                               (self.mem_k, self.mem_v))
            k = tf.concat((mem_k, k), axis=-2)
            v = tf.concat((mem_v, v), axis=-2)
            if exists(input_mask):
                input_mask = tf.pad(
                    input_mask,
                    paddings=[[0, 0], [0, 0], [0, 0], [self.num_mem_kv,
                                                       0]],  # last 2 dim
                    constant_values=True)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots

        if talking_heads:
            dots = tf.einsum('b h i j, h k -> b k i j', dots,
                             self.pre_softmax_proj)

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots = tf.where(input_mask, dots, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = tf.range(i)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(
                r, 'j -> () () () j')

            mask = tf.pad(mask,
                          paddings=[[0, 0], [0, 0], [0, 0], [j - i, 0]],
                          constant_values=False)
            dots = tf.where(mask, mask_value, dots)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.size(-1):
            top, _ = dots.topk(self.sparse_topk)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots = tf.where(mask, mask_value, dots)
            del mask

        attn = self.attn_fn(dots, axis=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        if talking_heads:
            attn = tf.einsum('b h i j, h k -> b k i j', attn,
                             self.post_softmax_proj)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        intermediates = Intermediates(pre_softmax_attn=pre_softmax_attn,
                                      post_softmax_attn=post_softmax_attn)

        return self.to_out(out), intermediates


class AttentionLayers(layers.Layer):
    def __init__(self,
                 dim,
                 depth,
                 heads=8,
                 causal=False,
                 cross_attend=False,
                 only_cross=False,
                 use_scalenorm=False,
                 use_rmsnorm=False,
                 use_rezero=False,
                 rel_pos_bias=False,
                 rel_pos_num_buckets=32,
                 rel_pos_max_distance=128,
                 position_infused_attn=False,
                 rotary_pos_emb=False,
                 rotary_emb_dim=None,
                 custom_layers=None,
                 sandwich_coef=None,
                 par_ratio=None,
                 residual_attn=False,
                 cross_residual_attn=False,
                 macaron=False,
                 pre_norm=True,
                 gate_residual=False,
                 **kwargs):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)
        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.tf_layers = []

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(
            dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(
            rotary_emb_dim) if rotary_pos_emb else None

        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'
        self.rel_pos = RelativePositionBias(
            scale=dim_head**0.5,
            causal=causal,
            heads=heads,
            num_buckets=rel_pos_num_buckets,
            max_distance=rel_pos_max_distance) if rel_pos_bias else None

        self.pre_norm = pre_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        norm_class = ScaleNorm if use_scalenorm else layers.LayerNormalization
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, axis=-1, epsilon=1e-5)

        norm_fn = Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f', ) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(
                default_block
            ) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f', ) * (par_width -
                                                   len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f', ) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a', ) * sandwich_coef + default_block * (
                depth - sandwich_coef) + ('f', ) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim,
                                  heads=heads,
                                  causal=causal,
                                  **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.tf_layers.append([norm_fn(), layer, residual_fn])

    def call(self,
             x,
             context=None,
             mask=None,
             context_mask=None,
             mems=None,
             return_hiddens=False):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(*map(
                lambda m: (m.size(1) if exists(m) else 0) + x.size(1), mems))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
                zip(self.layer_types, self.tf_layers)):
            is_last = ind == (len(self.tf_layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == 'a':
                out, inter = block(x,
                                   mask=mask,
                                   sinusoidal_emb=self.pia_pos_emb,
                                   rel_pos=self.rel_pos,
                                   rotary_pos_emb=rotary_pos_emb,
                                   prev_attn=prev_attn,
                                   mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x,
                                   context=context,
                                   mask=mask,
                                   context_mask=context_mask,
                                   prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates)

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class ViTransformerWrapper(layers.Layer):
    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 attn_layers,
                 num_classes=None,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        assert isinstance(attn_layers,
                          Encoder), 'attention layers must be an Encoder'
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size)**2

        self.patch_size = patch_size

        self.pos_embedding = self.add_weight(shape=(1, num_patches + 1, dim),
                                             initializer="normal",
                                             name="pos_embedding")
        self.patch_to_embedding = layers.Dense(dim)
        self.cls_token = self.add_weight(shape=(1, 1, dim),
                                         initializer="normal",
                                         name="cls_token")
        self.dropout = layers.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.mlp_head = FeedForward(
            dim, dim_out=num_classes,
            dropout=dropout) if exists(num_classes) else None

    def call(self, img, return_embeddings=False):
        p = self.patch_size

        x = rearrange(img,
                      'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=p,
                      p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        return self.mlp_head(x[:, 0])


class TransformerWrapper(layers.Layer):
    def __init__(self,
                 *,
                 num_tokens,
                 max_seq_len,
                 attn_layers,
                 emb_dim=None,
                 max_mem_len=0.,
                 emb_dropout=0.,
                 num_memory_tokens=None,
                 tie_embedding=False,
                 use_pos_emb=True):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len

        self.token_emb = layers.Embedding(
            num_tokens, emb_dim, embeddings_initializer=get_initializer(0.02))
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (
            use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = layers.Dropout(emb_dropout)

        self.project_emb = layers.Dense(dim) if emb_dim != dim else Identity()
        self.attn_layers = attn_layers
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.to_logits = layers.Dense(
            num_tokens) if not tie_embedding else lambda t: tf.matmul(
                t, self.token_emb.embeddings, transpose_b=True)

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = self.add_weight(shape=(num_memory_tokens,
                                                        dim),
                                                 initializer="normal",
                                                 name="memory_tokens")

            # let funnel encoder know number of memory tokens, if specified
            # TODO: think of a cleaner solution
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def call(self,
             x,
             return_embeddings=False,
             mask=None,
             return_mems=False,
             return_attn=False,
             mems=None,
             **kwargs):
        b, n, num_mem = *x.shape, self.num_memory_tokens
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = tf.concat((mem, x), axis=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = tf.pad(mask, [[0, 0], [0, 0], [num_mem, 0]], value=True)

        x, intermediates = self.attn_layers(x,
                                            mask=mask,
                                            mems=mems,
                                            return_hiddens=True,
                                            **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(
                map(lambda pair: tf.concat(pair, axis=-2), zip(
                    mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(
                map(lambda t: t[..., -self.max_mem_len:, :], new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates))
            return out, attn_maps

        return out


class ContinuousTransformerWrapper(layers.Layer):
    def __init__(self,
                 *,
                 max_seq_len,
                 attn_layers,
                 dim_in=None,
                 dim_out=None,
                 emb_dropout=0.,
                 use_pos_emb=True):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len) if (
            use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = layers.Dropout(emb_dropout)

        self.project_in = layers.Dense(dim) if exists(dim_in) else Identity()

        self.attn_layers = attn_layers
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.project_out = layers.Dense(dim_out) if exists(
            dim_out) else Identity()

    def call(self,
             x,
             return_embeddings=False,
             mask=None,
             return_attn=False,
             mems=None,
             **kwargs):

        x = self.project_in(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x,
                                            mask=mask,
                                            mems=mems,
                                            return_hiddens=True,
                                            **kwargs)
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates))
            return out, attn_maps

        return out


class XTransformer(layers.Layer):
    def __init__(self, *, dim, tie_token_emb=False, **kwargs):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'],
                                              enc_kwargs)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop(
            'num_memory_tokens', None)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'],
                                              dec_kwargs)

        self.encoder = TransformerWrapper(**enc_transformer_kwargs,
                                          attn_layers=Encoder(dim=dim,
                                                              **enc_kwargs))

        self.decoder = TransformerWrapper(**dec_transformer_kwargs,
                                          attn_layers=Decoder(
                                              dim=dim,
                                              cross_attend=True,
                                              **dec_kwargs))

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder)

    def generate(self, seq_in, seq_out_start, seq_len, src_mask=None):
        encodings = self.encoder(seq_in, return_embeddings=True, mask=src_mask)
        return self.decoder.generate(seq_out_start,
                                     seq_len,
                                     context=encodings,
                                     context_mask=src_mask)

    def call(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encoder(src, mask=src_mask, return_embeddings=True)
        out = self.decoder(tgt,
                           context=enc,
                           mask=tgt_mask,
                           context_mask=src_mask)
        return out