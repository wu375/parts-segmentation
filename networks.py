import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

class AttentionLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        d_model=1024,
        n_heads=8,
        input_len=7,
        dropout=0.1,
    ):
        super(AttentionLayer, self).__init__()

        self._n_heads = n_heads
        self._dk = d_model

        self._q_linear = nn.Linear(input_size, d_model)
        self._q_layernorm = nn.LayerNorm([input_len, d_model])
        self._k_linear = nn.Linear(input_size, d_model)
        self._k_layernorm = nn.LayerNorm([input_len, d_model])
        self._v_linear = nn.Linear(input_size, d_model)
        self._v_layernorm = nn.LayerNorm([input_len, d_model])

        self._att_dropout = nn.Dropout(dropout)
        self._ff_dropout1 = nn.Dropout(dropout)
        self._ff_dropout2 = nn.Dropout(dropout)
        self._att_norm = nn.LayerNorm(d_model)

        ff_size = d_model
        # self._attention_mlp = nn.ModuleList([nn.Linear(self._attention_size, self._attention_size) for _ in range(attention_mlp_layers)])
        self._attention_mlp1 = nn.Linear(d_model, ff_size)
        self._attention_mlp2 = nn.Linear(ff_size, output_size)
        self._activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, K, 2*D_slot)
        k = self._k_linear(x) # (batch, K, d_model)
        k = self._k_layernorm(k)
        v = self._v_linear(x) # (batch, K, d_model)
        v = self._v_layernorm(v)
        q = self._q_linear(x)
        q = self._q_layernorm(q)

        K = x.shape[1]

        k = k.view(k.shape[0], K, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, K, k)
        v = v.view(v.shape[0], K, self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, K, v)
        q = q.view(q.shape[0], q.shape[1], self._n_heads, -1).permute(0, 2, 1, 3) # (batch, n_heads, K, q)

        qk = torch.matmul(q, k.permute(0, 1, 3, 2)) # (batch, n_heads, K, q) @ (batch, n_heads, k, K)
        
        q = q * (self._dk ** -0.5)

        weights = F.softmax(qk, dim=-1) # (batch, n_heads, K, K)

        attentions = torch.matmul(weights, v) # (batch, n_heads, K, v)

        attentions = attentions.permute(0, 2, 1, 3).contiguous() # (batch, K, n_heads, v)
        attentions = attentions.view((attentions.shape[0], attentions.shape[1], -1)) # (batch, K, d_model)

        attentions = self._att_dropout(attentions)
        attentions = self._att_norm(attentions)

        attentions = self._ff_dropout1(self._activation(self._attention_mlp1(attentions)))
        attentions = self._ff_dropout2(self._attention_mlp2(attentions)) # (batch, K, out)

        return attentions


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            K=7,
            d_model=1024,
            n_layers=2,
    ):
        super(TransformerEncoder, self).__init__()

        self._layers = nn.ModuleList([AttentionLayer(input_size=input_size, output_size=output_size, d_model=d_model, input_len=K) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)

        return x

# TODO: to pytorch
def build_grid(height, width):
    h = np.linspace(0., 1., num=height)
    w = np.linspace(0., 1., num=width)
    grid = np.meshgrid(h, w, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [height, width, -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    grid = np.concatenate([grid, 1.0 - grid], axis=-1)
    grid = torch.tensor(grid) #.cuda()
    return grid


class SpatialDecoder(nn.Module):
    def __init__(
            self,
            height,
            width,
            D_slot,
            cnn_hidden=512,
    ):
        super(SpatialDecoder, self).__init__()

        self._height = height
        self._width = width

        self._grid_projector = nn.Linear(4, D_slot)

        cnn_layers = []
        in_channels = [D_slot, cnn_hidden, cnn_hidden, cnn_hidden, cnn_hidden, cnn_hidden]
        out_channels = [cnn_hidden, cnn_hidden, cnn_hidden, cnn_hidden, cnn_hidden, 4]

        for in_channel, out_channel in zip(in_channels, out_channels):
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ))
            cnn_layers.append(nn.ReLU())
        cnn_layers = cnn_layers[:-1]
        self._cnns = nn.Sequential(*cnn_layers)

        self.register_buffer('_grid', build_grid(self._height, self._width)) # (1, h, w, 4)


    def forward(self, slots):
        # slots: (batch, K, D_slot)
        batch_size, K = slots.shape[0], slots.shape[1]
        slots = slots.view(-1, slots.shape[-1])[:, None, None, :] # (batch*K, 1, 1, D_slot)
        slots = slots.expand(slots.shape[0], self._height, self._width, slots.shape[-1]) # (batch*K, h, w, D_slot)

        grid = self._grid_projector(self._grid) # (1, h, w, D_slot)

        slots = slots + grid # (batch*K, h, w, D_slot)
        slots = slots.permute(0, 3, 1, 2) # (batch*K, D_slot, h, w)
        cnn_out = self._cnns(slots) # (batch*K, 4, h, w)
        cnn_out = cnn_out.view(batch_size, K, *cnn_out.shape[1:]) # (batch, K, 4, h, w)

        recons = cnn_out[:, :, :3] # (batch, K, 3, h, w)
        masks = cnn_out[:, :, 3:] # (batch, K, 1, h, w)

        masks = F.softmax(masks, dim=1) # (batch, K, 1, h, w)

        return recons, masks


class BottomUpEncoder(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
    ):
        super(BottomUpEncoder, self).__init__()

        in_channels = [input_size, hidden_size//2, hidden_size, hidden_size, output_size]
        out_channels = [hidden_size//2, hidden_size, hidden_size, output_size, output_size]
        cnn_layers = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            cnn_layers.append(nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=5,
                stride=1,
                padding=2,
            ))
            cnn_layers.append(nn.ReLU())
        cnn_layers = cnn_layers[:-1]
        self._cnns = nn.Sequential(*cnn_layers)

    def forward(self, x):
        # x: (batch, c, h, w)
        x = self._cnns(x)
        return x


class SlotAttention(nn.Module):
    def __init__(
            self,
            input_size,
            obs_channels,
            qk_size,
            v_size,
    ):
        super(SlotAttention, self).__init__()

        self._input_norm = nn.LayerNorm(input_size)
        self._obs_norm = nn.LayerNorm(obs_channels)
        self._out_norm = nn.LayerNorm(v_size)

        self._qk_size = qk_size

        self._q_linear = nn.Linear(input_size, qk_size)
        self._k_linear = nn.Linear(obs_channels, qk_size)
        self._v_linear = nn.Linear(obs_channels, v_size)


    def forward(self, slots, obs):
        # slots: (batch, K, input_size)
        # obs: (batch, h, w, c)
        batch_size, h, w, c = obs.shape

        slots = self._input_norm(slots)
        obs = self._obs_norm(obs)

        k = self._k_linear(obs) # (batch, h, w, 2*D_slot)
        v = self._v_linear(obs) # (batch, h, w, 2*D_slot)

        q = self._q_linear(slots) # (batch, K, 2*D_slot)

        q = q * (self._qk_size ** -0.5)

        # (batch, K, 2*D_slot) @ (batch, 2*D_slot, h*w) = (batch, K, h*w)
        qk = torch.matmul(q, k.permute(0, 3, 1, 2).view(k.shape[0], k.shape[3], -1))
        
        attentions = F.softmax(qk, dim=1) # (batch, K, h*w)

        temp = attentions / attentions.sum(-1, keepdim=True)
        temp = torch.matmul(temp, v.view(v.shape[0], v.shape[1]*v.shape[2], v.shape[3]))
        temp = self._out_norm(temp)

        # attentions = torch.matmul(weights, v) 
        attentions = attentions.view(batch_size, h, w, -1) # (batch, h, w, K)
        spatial_logits = attentions.permute(0, 3, 1, 2)

        slots = temp

        return slots, spatial_logits


