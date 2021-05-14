from tf_fast_api import *
from tf_x_transformers.tf_entmax import entmax15

entmax = entmax15  # TODO entmax_beisect
# nucleus


def top_p(logits, thres=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    cum_probs = tf.cumsum(tf.nn.softmax(sorted_logits),
                          axis=-1,
                          exclusive=True)
    logits_masked = tf.where(cum_probs > (1 - thres), 10000, sorted_logits)
    min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True)
    return tf.where(logits >= min_logits, logits, float("-inf"))


# topk


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.size(-1))
    kth_logits = tf.sort(logits, direction="DESCENDING")[..., k - 1:k]
    return tf.where(logits >= kth_logits, logits, float("-inf"))


# entmax


class PytorchCrossEntropyLoss:
    def __init__(self,
                 from_logits: bool = True,
                 reduction: str = "mean",
                 weight: tf.Tensor = None,
                 ignore_index: int = -100,
                 name: str = 'pytorch_cross_entropy_loss'):
        super().__init__()
        self.name = name
        self.from_logits = from_logits
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)

    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred):
        active_loss = tf.not_equal(tf.reshape(y_true, (-1, )),
                                   self.ignore_index)
        y_pred = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.size(-1))),
                                 active_loss)
        y_true = tf.boolean_mask(tf.reshape(y_true, (-1, )), active_loss)
        loss = tf.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=self.from_logits)

        if self.weight is None:
            if self.reduction == "mean":
                loss = tf.reduce_mean(loss)
            elif self.reduction == "sum":
                loss = tf.reduce_sum(loss)
            elif self.reduction == "none":
                pass
            else:
                raise ValueError(
                    "reduction must choose from [mean, sum, none]")
        else:
            values = tf.gather(self.weight, y_true)
            if self.reduction == "mean":
                loss = tf.reduce_sum(values * loss) / tf.reduce_sum(values)
            elif self.reduction == "sum":
                loss = tf.reduce_sum(values * loss)
            elif self.reduction == "none":
                loss = values * loss
            else:
                raise ValueError(
                    "reduction must choose from [mean, sum, none]")

        return loss


class AutoregressiveWrapper(layers.Layer):
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.loss_fn = PytorchCrossEntropyLoss(ignore_index=ignore_index)

    def generate(self,
                 start_tokens,
                 seq_len,
                 eos_token=None,
                 temperature=1.,
                 filter_logits_fn=top_k,
                 filter_thres=0.9,
                 **kwargs):
        was_training = K.learning_phase()
        num_dims = start_tokens.ndim

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.size()

        K.set_learning_phase(0)
        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = tf.fill(out.size(), True)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = tf.nn.softmax(filtered_logits / temperature, axis=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, axis=-1)

            sample = tf.random.categorical(probs, 1)

            out = tf.concat((out, sample), axis=-1)
            mask = tf.pad(mask, [[0, 0], [0, 1]], value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        K.set_learning_phase(int(was_training))

        return out

    def call(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get('mask', None)
        if mask is not None and mask.size(1) == x.size(1):
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(xi, **kwargs)

        loss = self.loss_fn(xo, out)
        return loss
