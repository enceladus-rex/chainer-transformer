from math import sqrt
from collections import namedtuple

from chainer import Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def scaled_dot_product_attention(queries, keys, values, scale=0.1, mask=None):
    x1 = F.matmul(queries, keys, transb=True) / xp.array(scale,
                                                         dtype=keys.dtype)
    x2 = F.where(mask,
                 xp.ones_like(x1.array) *
                 -xp.inf, x1) if mask is not None else x1
    x3 = F.softmax(x2)
    x4 = F.matmul(x3, values)
    return x4


def generate_positional_encoding(start, end, dim):
    positions = xp.arange(start, end)
    stacks = []
    for i in range(dim):
        divisor = 10000**(2 * i / float(dim))
        elements = positions / divisor
        if i % 2 == 0:
            pe = xp.sin(elements, dtype=xp.float32)
        else:
            pe = xp.cos(elements, dtype=xp.float32)
        stacks.append(pe)
    return xp.transpose(xp.stack(stacks))


class BatchApply(Chain):
    def __init__(self, module_constructor, num_batch_dims=None):
        super().__init__()
        self.num_batch_dims = num_batch_dims
        with self.init_scope():
            self.module = module_constructor()

    def forward(self, x):
        num_batch_dims = self.num_batch_dims
        if num_batch_dims is None:
            num_batch_dims = max(0, len(x.shape) - 1)
        original_shape = x.shape
        if len(original_shape) <= 1 or num_batch_dims == 0:
            return self.module(x)
        else:
            merge_dims = original_shape[:num_batch_dims]
            merge_size = 1
            for e in merge_dims:
                merge_size *= e
            merged_shape = (merge_size, ) + original_shape[num_batch_dims:]
            merged_view = F.reshape(x, merged_shape)
            output = self.module(merged_view)
            output_dims = output.shape[1:]
            final = F.reshape(output, merge_dims + output_dims)
            return final


class MultiHeadAttention(Chain):
    def __init__(self, num_heads, model_dim, key_dim, value_dim):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.multi_head_dim = num_heads * value_dim
        with self.init_scope():
            self.head_query_links = ChainList()
            self.head_key_links = ChainList()
            self.head_value_links = ChainList()
            for i in range(num_heads):
                self.head_query_links.append(L.Linear(model_dim, key_dim))
                self.head_key_links.append(L.Linear(model_dim, key_dim))
                self.head_value_links.append(L.Linear(model_dim, value_dim))
            self.output_link = L.Linear(self.multi_head_dim, model_dim)

    def forward(self, queries, keys, values, mask=None):
        heads = []
        for i in range(self.num_heads):
            query_projection = self.head_query_links[i](queries,
                                                        n_batch_axes=2)
            key_projection = self.head_key_links[i](keys, n_batch_axes=2)
            value_projection = self.head_value_links[i](values, n_batch_axes=2)

            head = scaled_dot_product_attention(query_projection,
                                                key_projection,
                                                value_projection,
                                                mask=mask)

            heads.append(head)

        multi_head = F.concat(heads, axis=-1)
        return self.output_link(multi_head, n_batch_axes=2)


class PointwiseFeedForwardNetwork(Chain):
    def __init__(self, model_dim, inner_dim):
        super().__init__()
        with self.init_scope():
            self.lin1 = L.Linear(model_dim, inner_dim)
            self.lin2 = L.Linear(inner_dim, model_dim)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x, n_batch_axes=2)), n_batch_axes=2)


class TransformerEncoderUnit(Chain):
    def __init__(self, num_heads, model_dim, ff_dim, p_drop):
        super().__init__()
        self.p_drop = p_drop
        with self.init_scope():
            kv_dim = model_dim // num_heads
            self.mha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
            self.lnorm1 = BatchApply(L.LayerNormalization)
            self.lnorm2 = BatchApply(L.LayerNormalization)
            self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)

    def forward(self, inputs_unit):
        x1 = F.dropout(self.mha(inputs_unit, inputs_unit, inputs_unit),
                       self.p_drop)
        x2 = self.lnorm1(inputs_unit + x1)
        x3 = F.dropout(self.ff(x2), self.p_drop)
        x4 = self.lnorm2(x2 + x3)
        return x4


class TransformerDecoderUnit(Chain):
    def __init__(self, num_heads, model_dim, ff_dim, p_drop):
        super().__init__()
        self.p_drop = p_drop
        with self.init_scope():
            kv_dim = model_dim // num_heads
            self.mmha = MultiHeadAttention(num_heads, model_dim, kv_dim,
                                           kv_dim)
            self.mha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
            self.lnorm1 = BatchApply(L.LayerNormalization)
            self.lnorm2 = BatchApply(L.LayerNormalization)
            self.lnorm3 = BatchApply(L.LayerNormalization)
            self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)

    def forward(self, inputs_encoding, outputs_unit):
        mask_shape = list(outputs_unit.shape)
        mask_shape[-1] = mask_shape[-2]
        mask = xp.triu(xp.ones(mask_shape, dtype=xp.bool), k=1)
        x1 = F.dropout(
            self.mmha(outputs_unit, outputs_unit, outputs_unit, mask),
            self.p_drop)
        x2 = self.lnorm1(outputs_unit + x1)
        x3 = F.dropout(self.mha(x2, inputs_encoding, inputs_encoding),
                       self.p_drop)
        x4 = self.lnorm2(x2 + x3)
        x5 = F.dropout(self.ff(x4), self.p_drop)
        x6 = self.lnorm3(x4 + x5)
        return x6


class TransformerEncoder(Chain):
    def __init__(self, depth, num_heads, model_dim, ff_dim, p_drop):
        super().__init__()
        with self.init_scope():
            self.unit_links = ChainList()
            for i in range(depth):
                self.unit_links.append(
                    TransformerEncoderUnit(num_heads, model_dim, ff_dim,
                                           p_drop))

    def forward(self, inputs_encoding):
        unit_inputs = [inputs_encoding]
        for unit_link in self.unit_links:
            x = unit_inputs[-1]
            o = unit_link(x)
            unit_inputs.append(o)
        return unit_inputs[-1]


class TransformerDecoder(Chain):
    def __init__(self, depth, num_heads, model_dim, ff_dim, p_drop):
        super().__init__()
        with self.init_scope():
            self.lin1 = L.Linear(model_dim)
            self.unit_links = ChainList()
            for i in range(depth):
                self.unit_links.append(
                    TransformerDecoderUnit(num_heads, model_dim, ff_dim,
                                           p_drop))

    def forward(self, inputs_encoding, outputs_unit):
        unit_inputs = [outputs_unit]
        for unit_link in self.unit_links:
            x = unit_inputs[-1]
            o = unit_link(inputs_encoding, x)
            unit_inputs.append(o)
        unit_output = unit_inputs[-1]
        return F.softmax(self.lin1(unit_output, n_batch_axes=2))


class Transformer(Chain):
    def __init__(self,
                 source_vocab,
                 target_vocab,
                 num_heads=2,
                 encoder_depth=2,
                 decoder_depth=2,
                 ff_dim=512,
                 p_drop=0.1,
                 max_input_length=256,
                 max_output_length=256):
        super().__init__()
        input_model_dim = source_vocab.embedding_size
        output_model_dim = target_vocab.embedding_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.input_model_dim = input_model_dim
        self.output_model_dim = output_model_dim
        self.ff_dim = ff_dim
        self.p_drop = p_drop
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.input_positional_encodings = generate_positional_encoding(
            0, max_input_length, input_model_dim)
        self.output_positional_encodings = generate_positional_encoding(
            0, max_output_length, output_model_dim)
        with self.init_scope():
            self.encoder = TransformerEncoder(encoder_depth, num_heads,
                                              input_model_dim, ff_dim, p_drop)
            self.decoder = TransformerDecoder(decoder_depth, num_heads,
                                              output_model_dim, ff_dim, p_drop)
            self.linear = L.Linear(output_model_dim, target_vocab.vocab_size)

    def forward(self, input_ids, input_masks=None, length=None):
        batch_size, input_length = input_ids.shape[0], input_ids.shape[1]
        input_embeddings = F.embed_id(input_ids, self.source_vocab.embeddings)
        embeddings_dtype = input_embeddings.dtype
        input_positional_encodings = F.expand_dims(
            self.input_positional_encodings[:input_length, :].astype(
                embeddings_dtype), 0)

        transformed_inputs = F.dropout(
            input_positional_encodings + input_embeddings, self.p_drop)

        encoding = self.encoder(transformed_inputs)

        output_probs = None
        output_embeddings = self.target_vocab.embed(
            [self.target_vocab.start_id])
        output_embeddings = F.expand_dims(output_embeddings, 0)
        output_embeddings = F.tile(output_embeddings, (batch_size, 1, 1))

        end_predicted = F.tile(F.reshape(xp.array([False]), (1, 1)),
                               (batch_size, 1))

        all_done = False
        current_length = 0
        while (length is None
               and not all_done) or (length is not None
                                     and current_length < length):
            output_positional_encodings = F.expand_dims(
                self.output_positional_encodings[:output_embeddings.
                                                 shape[-2], :].astype(
                                                     embeddings_dtype), 0)

            transformed_ouputs = F.dropout(
                output_positional_encodings + output_embeddings, self.p_drop)

            decoding = self.decoder(encoding, transformed_ouputs)

            logits = self.linear(decoding, n_batch_axes=2)
            token_probs = F.softmax(logits, axis=-1)

            next_token_probs = token_probs[:, -1, :]
            next_token_ids = F.argmax(next_token_probs, axis=-1)
            next_token_embeddings = F.embed_id(next_token_ids,
                                               self.target_vocab.embeddings)
            next_token_embeddings = F.expand_dims(next_token_embeddings,
                                                  axis=1)
            output_embeddings = F.concat(
                [output_embeddings, next_token_embeddings], axis=1)

            next_output_probs = F.expand_dims(next_token_probs, axis=1)
            if output_probs is None:
                output_probs = next_output_probs
            else:
                output_probs = F.concat([output_probs, next_output_probs],
                                        axis=1)

            next_token_end = (next_token_ids.array == self.target_vocab.end_id)
            next_end_predicted = F.expand_dims(
                end_predicted[:, -1].array | next_token_end, -1)
            end_predicted = F.concat([end_predicted, next_end_predicted],
                                     axis=-1)
            all_done = xp.all(next_end_predicted.array)
            current_length += 1

        output_dtype = output_embeddings.dtype
        output_masks = F.where(
            end_predicted.array,
            xp.zeros_like(end_predicted.array, dtype=output_dtype),
            xp.ones_like(end_predicted.array, dtype=output_dtype))
        return output_probs, output_masks
