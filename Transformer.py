import torch
from torch import nn as nn
import numpy as np
from time import time

"""
TODO:
    1. Create the module for transformer. -> Done
    2. Create the methods for creating the position -> Done, causal and decoder masks.
    3. Test out the model on some real world data.
"""

class self_attention(nn.Module):
    def __init__(self, d_model_in, d_model_out):
        super(self_attention, self).__init__()
        self.d_model = d_model_out
        self.Wk = nn.Linear(d_model_in, d_model_out)
        self.Wq = nn.Linear(d_model_in, d_model_out)
        self.Wv = nn.Linear(d_model_in, d_model_out)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, query, key, value, mask = None):
        key = self.Wk(key)
        query = self.Wq(query)
        value = self.Wv(value)

        attention_score = torch.matmul(query, torch.transpose(key, 1,2)) / (self.d_model ** 0.5)
        
        if mask is not None:
            attention_score = torch.where(
                mask == 0,
                -np.inf,
                attention_score
            )
        attention_weight = self.softmax(attention_score)
        return torch.matmul(attention_weight, value)

class multihead_self_attention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(multihead_self_attention, self).__init__()
        self.attention_layers = nn.ModuleList([self_attention(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.projection_layer = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, mask = None):
        result = []
        for layer in self.attention_layers:
            result.append(layer(query, key, value, mask))
        result = torch.concatenate(result, dim = -1).reshape(*value.size())
        result = self.projection_layer(result)
        return result

class feedforward_layer(nn.Module):
    def __init__(self, d_model, d_hidden, dropout_p = 0.2):
        super(feedforward_layer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p = dropout_p)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_model, d_hidden, dropout_p):
        super(EncoderBlock, self).__init__()
        self.mha_layer = multihead_self_attention(n_heads, d_model)
        self.ff_layer = feedforward_layer(d_model, d_hidden, dropout_p)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = dropout_p)
        self.dropout2 = nn.Dropout(p = dropout_p)
    def forward(self, x, mask = None):
        x_mha_out = self.mha_layer(x,x,x, mask)
        x_mha_out = self.dropout1(x_mha_out)
        x_mha_out = self.layer_norm1(x_mha_out)

        x_ff_out = self.ff_layer(x_mha_out)
        x_ff_out = self.dropout2(x_ff_out)

        x_output = self.layer_norm2(x_mha_out + x_ff_out)

        return x_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_encoders, n_heads, d_model, d_hidden, dropout_p):
        super(TransformerEncoderLayer, self).__init__()
        self.encoder_sublayers = nn.ModuleList([
            EncoderBlock(n_heads, d_model, d_hidden, dropout_p) for _ in range(n_encoders)
        ])
    def forward(self, x, mask = None):
        for layer in self.encoder_sublayers:
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,n_heads, d_model, d_hidden, dropout_p):
        super(DecoderBlock, self).__init__()
        self.masked_mha_layer = multihead_self_attention(n_heads, d_model)
        self.cross_mha_layer = multihead_self_attention(n_heads, d_model)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.ff_layer = feedforward_layer(d_model, d_hidden, dropout_p)

        self.dropout1 = nn.Dropout(p = dropout_p)
        self.dropout2 = nn.Dropout(p = dropout_p)
        self.dropout3 = nn.Dropout(p = dropout_p)
        self.dropout4 = nn.Dropout(p = dropout_p)
    def forward(self, x, encoder_output, decoder_mask = None, causal_mask = None):
        x = self.dropout1(x)
        x_masked_mha_out = self.masked_mha_layer(x,x,x, decoder_mask)
        x_masked_mha_out = self.dropout2(x_masked_mha_out)
        x = self.layer_norm1(x + x_masked_mha_out)

        x_cross_mha_out = self.cross_mha_layer(encoder_output,x,x,causal_mask)
        x_cross_mha_out = self.dropout3(x_cross_mha_out)
        x = self.layer_norm2(x + x_cross_mha_out)

        x_ff_out = self.ff_layer(x)
        x_ff_out = self.dropout4(x_ff_out)
        x = self.layer_norm3(x + x_ff_out)

        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_decoders, n_heads, d_model, d_hidden, dropout_p):
        super(TransformerDecoderLayer, self).__init__()
        self.decoder_sublayers = nn.ModuleList([
            DecoderBlock(n_heads, d_model, d_hidden, dropout_p) for _ in range(n_decoders)
        ])
    def forward(self, x, encoder_output, decoder_mask = None, causal_mask = None):
        for layer in self.decoder_sublayers:
            x = layer(x, encoder_output, decoder_mask, causal_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, n_encoders, n_decoders, n_heads, d_model, d_hidden, dropout_p, n_class):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoderLayer(n_encoders, n_heads, d_model, d_hidden, dropout_p)
        self.decoder = TransformerDecoderLayer(n_decoders, n_heads, d_model, d_hidden, dropout_p)
        self.ff = nn.Linear(d_model, n_class)
    def forward(self, x_encoder, encoder_mask, x_decoder, decoder_mask, causal_mask):
        encoder_output = self.encoder(x_encoder, encoder_mask)
        decoder_output = self.decoder(x_decoder, encoder_output, decoder_mask, causal_mask)
        return self.ff(decoder_output)

class Model(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, sequence_length, batch_size, d_model, n_encoders, n_decoders, n_heads, d_hidden, dropout_p, n_class):
        super(Model, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(output_vocab_size, d_model)
        self.transformer = Transformer(n_encoders, n_decoders, n_heads, d_model, d_hidden, dropout_p, n_class)
        self.position_encoding_matrix = self.positional_encoding(sequence_length, d_model)
        self.position_encoding_matrix = torch.concatenate([
            torch.tensor(self.position_encoding_matrix.reshape(1, sequence_length, d_model), dtype = torch.float) for _ in range(batch_size)
        ], dim = 0)
    def positional_encoding(self, sequence_length, d_model, n = 10000):
        """
        source : https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        """
        matrix = np.zeros((sequence_length, d_model))
        for k in range(sequence_length):
            for i in range(d_model // 2):
                denominator = np.power(n, 2*i / d_model)
                matrix[k, 2 * i] = np.sin(k / denominator)
                matrix[k, 2 * i + 1] = np.cos(k / denominator)
        return matrix
    def forward(self, x_encoder, encoder_mask, x_decoder, decoder_mask, causal_mask):
        x_encoder = self.encoder_embedding(x_encoder)
        x_decoder = self.decoder_embedding(x_decoder)
        x_encoder = x_encoder + self.position_encoding_matrix
        x_decoder = x_decoder + self.position_encoding_matrix
        output = self.transformer(x_encoder, encoder_mask, x_decoder, decoder_mask, causal_mask)
        return output



def test():
    batch_size = 32
    sequence_length = 128
    d_model = 512
    d_hidden = 128
    dropout_p = 0.2
    n_heads = 8
    n_encoders = 6
    n_decoders = 6
    input_vocab_size = 10000
    output_vocab_size = 10000
    n_class = 10000

    start = time()
    model = Model(input_vocab_size, output_vocab_size, sequence_length, batch_size, d_model, n_encoders, n_decoders, n_heads, d_hidden, dropout_p, n_class)
    x_encoder = torch.randint(0, input_vocab_size - 1, (batch_size, sequence_length))
    x_decoder = torch.randint(0, output_vocab_size - 1, (batch_size, sequence_length))
    encoder_mask = None
    decoder_mask = None
    causal_mask = None
    print(model)
    output = model(x_encoder, encoder_mask, x_decoder, decoder_mask, causal_mask)
    print("Output Shape : ", output.size(), " | time : ", time() - start)

if __name__ == '__main__':
    test()


