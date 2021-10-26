import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from networks import TransformerEncoder, SpatialDecoder, BottomUpEncoder, SlotAttention, build_mlp

# from torchviz import make_dot

class Parts(nn.Module):
    def __init__(
        self,
        K=7, # aka n_slots
        D_slot=64,
        height=64,
        width=64,
        n_refine_steps=1,
        transformer_d_model=1024,
        decoder_var=0.09,
        decoder_hidden=512,
        rnn_hidden=256,
        encoder_channel=256,
        beta=0.5,
        mode='segmentation',
    ):

        super(Parts, self).__init__()

        self._K = K
        self._D_slot = D_slot
        self._n_refine_steps = n_refine_steps
        self._mode = mode
        self._decoder_var = decoder_var
        self._beta = beta
        self._rnn_hidden = rnn_hidden
        # self._rnn_hidden = 2*D_slot
        self._h = height
        self._w = width

        self._prediction_transformer = TransformerEncoder(
            input_size=D_slot*2,
            output_size=D_slot*2,
            d_model=transformer_d_model, 
            n_layers=2,
        )
        self._prediction_lstm = nn.LSTMCell(input_size=D_slot*2, hidden_size=self._rnn_hidden)
        self._prediction_out = nn.Linear(rnn_hidden, D_slot*2)

        self._spatial_decoder = SpatialDecoder(
            height=height, 
            width=width, 
            D_slot=D_slot,
            cnn_hidden=decoder_hidden,
        )

        self._encoder = BottomUpEncoder(input_size=3+1, output_size=encoder_channel)

        self._slot_attention = SlotAttention(
            input_size=4*D_slot,
            obs_channels=encoder_channel,
            qk_size=2*D_slot,
            v_size=2*D_slot,
        )

        self._grad_norm_layer= nn.LayerNorm(2*D_slot)

        self._lamb_update_mlp = build_mlp(
            input_size=6*D_slot,
            layer_sizes=[256, 256],
            output_size=D_slot*2,
            activation=torch.nn.ReLU,
        )

    def _init_hidden(self, batch_size):
        hx = torch.zeros(batch_size * self._K, self._rnn_hidden).cuda()
        cx = torch.zeros(batch_size * self._K, self._rnn_hidden).cuda()
        init_state = (hx, cx)
        return init_state

    def _init_lamb(self, batch_size):
        lamb = torch.zeros(batch_size, self._K, 2 * self._D_slot).cuda() #.requires_grad_()
        return lamb

    def _sample_z(self, mu, sigma):
        # mu: (batch, K, D_slot)
        # sigma: (batch, K, D_slot)
        normal = D.Normal(mu, sigma)
        normal = D.Independent(normal, reinterpreted_batch_ndims=2)
        z = normal.rsample().cuda() # (batch, K, D_slot)
        return z

    def _reconstruction_loss(self, x, C, sigma, m):
        # x: (batch, 3, h, w)
        # C: (batch, K, 3, h, w)
        # m: (batch, K, 1, h, w)

        some_math = m * (1/(sigma*math.sqrt(2*math.pi))) * (-((C - x.unsqueeze(1)) ** 2) / (2 * sigma)).exp() # (batch, K, 3, h, w)
        some_math = some_math.sum(dim=1, keepdim=True).log() #.sum(dim=2, keepdim=True) # (batch, 1, 3, h, w)
        logp_x_z = some_math.squeeze(1).sum(1, keepdim=True) # (batch, 3, h, w)
        some_math = some_math.view(some_math.shape[0], -1) # (batch, -1)
        nll = -some_math.sum(dim=-1).mean() # scaler
        return nll, logp_x_z

    def _kl_loss(self, mu, sigma):
        # mu: (batch, K, D_slot)
        # sigma: (batch, K, D_slot)
        posterior = D.Normal(mu, sigma)
        prior = D.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        kld = D.kl_divergence(posterior, prior).sum()
        return kld

    def _forward_step(self, x, lamb_prev, hidden, detach_hidden, action=None, burn_in=False):
        # x: (batch, c, h, w)
        # lamb_prev: (batch, K, 2*D_slot)
        # action: (??)
        # hidden: tuple of 2 (batch*K, hidden_size)
        batch_size = x.shape[0]

        # prediction model
        lamb = self._prediction_transformer(lamb_prev) # (batch, K, 2*D_slot)
        lamb = lamb.view(-1, self._D_slot*2)
        hidden = self._prediction_lstm(lamb, hidden)
        lamb = hidden[0].view(batch_size, self._K, -1)
        lamb = self._prediction_out(lamb)

        if detach_hidden:
            hidden = (hidden[0].detach(), hidden[1].detach())

        # lamb = lamb.detach()
        if not self.training:
            lamb.requires_grad = True

        for refine_step in range(self._n_refine_steps):
            lamb = lamb.view(lamb.shape[0], lamb.shape[1], 2, -1) # (batch, K, 2, D_slot)
            mu = lamb[:, :, 0] # (batch, K, D_slot)
            sigma = lamb[:, :, 1].exp()

            z = self._sample_z(mu, sigma) # (batch, K, D_slot)
            recons, masks = self._spatial_decoder(z) # recons: (batch, K, 3, h, w), masks: (batch, K, 1, h, w)
            # recons = torch.rand(batch_size, self._K, 3, 64, 64).cuda()
            # masks = torch.rand(batch_size, self._K, 1, 64, 64).cuda()

            loss_recon, logp_x_z = self._reconstruction_loss(x, recons, self._decoder_var, masks) # scaler, (batch, 1, h, w)
            loss_kl = self._kl_loss(mu, sigma)

            loss = loss_recon + loss_kl * self._beta

            if self._mode == 'segmentation' or burn_in:
                lamb_grad = torch.autograd.grad(
                    loss, 
                    lamb, 
                    retain_graph=refine_step==self._n_refine_steps-1,
                )[0] # (batch, K, 2, D_slot)
                # lamb_grad = torch.rand(batch_size, self._K, 2, self._D_slot).cuda()
                # print(lamb_grad.shape)
                # exit()
                lamb_grad = lamb_grad.view(batch_size, self._K, -1)
                lamb_grad = self._grad_norm_layer(lamb_grad)
                lamb_grad = torch.clamp(lamb_grad, min=-10, max=10) # (batch, K, 2*D_slot)

                lamb = lamb.view(batch_size, self._K, -1) # (batch, K, 2*D_slot)

                obs_encoded = self._encoder(torch.cat([x, logp_x_z], dim=1))
                assert obs_encoded.shape[-2] == self._h and obs_encoded.shape[-1] == self._w, 'encoder paddings not same'
                obs_encoded = obs_encoded.permute(0, 2, 3, 1)
                slotted_input = torch.cat([lamb, lamb_grad], dim=-1) # (batch, K, 4*D_slot)

                slotted_output, spatial_att_logits = self._slot_attention(slotted_input, obs_encoded) # (batch, K, 2*D_slot), (batch, K, h, w)
                
                lamb_delta_input = torch.cat((slotted_output, lamb, lamb_grad), dim=2) # (batch, K, 6*D_slot)
                lamb_delta = self._lamb_update_mlp(lamb_delta_input) # (batch, K, 2*D_slot)
                # lamb = lamb.detach() + lamb_delta.detach()
                lamb = lamb + lamb_delta
                # if self.training:
                #     lamb.requires_grad = True
                # lamb.retain_grad()
            else:
                lamb = lamb.view(batch_size, self._K, -1).detach()
                break

            if detach_hidden:
                # if self.training:
                #     lamb.requires_grad = True
                lamb = lamb.detach()

        if self._mode == 'segmentation' or burn_in:
            return {
                'lamb': lamb,
                'hidden': hidden,
                'recons':(recons*masks).sum(dim=1), # (batch, 3, h, w)
                'seg_topdown': masks.argmax(dim=1).squeeze(1), # (batch, h, w)
                'seg_bottomup': spatial_att_logits.argmax(dim=1), # (batch, h, w)
                'loss': loss,
                'loss_recon': loss_recon,
                'loss_kl': loss_kl,
                'lamb_grad': lamb_grad.mean(),
            }
        else:
            return {
                'lamb': lamb,
                'hidden': hidden,
                'recons':(recons*masks).sum(dim=1), # (batch, 3, h, w)
                'loss': loss,
                'loss_recon': loss_recon,
                'loss_kl': loss_kl,
            }

    def forward(self, x, hidden=None, optimizer=None):
        # x: (batch, time, c, h, w)
        batch_size, n_steps = x.shape[0], x.shape[1]

        if hidden is None:
            hidden = self._init_hidden(batch_size)
        lamb = self._init_lamb(batch_size)

        loss = 0.
        loss_recon = 0.
        loss_kl = 0.
        recons = []
        if self._mode == 'segmentation':
            lamb_grad = 0.
            seg_topdown = []
            seg_bottomup = []
        for t in range(n_steps):
            # print(t)
            detach_hidden = (t+1) % 4 == 0
            out = self._forward_step(x[:, t], lamb, hidden, detach_hidden=detach_hidden)
            lamb = out['lamb']
            hidden = out['hidden']
            loss = loss + out['loss']
            loss_recon = loss_recon + out['loss_recon']
            loss_kl = loss_kl + out['loss_kl']
            recons.append(out['recons'])
            if self._mode == 'segmentation':
                lamb_grad = lamb_grad + out.get('lamb_grad', 0)
                seg_topdown.append(out['seg_topdown'])
                seg_bottomup.append(out['seg_bottomup'])
            
            if detach_hidden and optimizer is not None:

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                loss = loss.detach()

        loss = loss / n_steps
        loss_recon = loss_recon / n_steps
        loss_kl = loss_kl / n_steps
        recons = torch.stack(recons, dim=1) # (batch, time, 3, h, w)

        if self._mode == 'segmentation':
            lamb_grad = lamb_grad / n_steps
            seg_topdown = torch.stack(seg_topdown, dim=1) # (batch, time, h, w)
            seg_bottomup = torch.stack(seg_bottomup, dim=1) # (batch, time, h, w)

            return {
                'lamb': lamb,
                'hidden': hidden,
                'recons':recons,
                'seg_topdown':seg_topdown,
                'seg_bottomup':seg_bottomup,
                'loss': loss,
                'loss_recon': loss_recon,
                'loss_kl': loss_kl,
                'lamb_grad': lamb_grad,
            }
        else:
            return {
                'lamb': lamb,
                'hidden': hidden,
                'recons':recons,
                'loss': loss,
                'loss_recon': loss_recon,
                'loss_kl': loss_kl,
            }


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    model = Parts(
        K=7, # aka n_slots
        D_slot=64,
        height=40,
        width=40,
        n_refine_steps=1,
        transformer_d_model=1024,
        decoder_var=0.09,
        rnn_hidden=256,
        encoder_channel=256,
        beta=0.5,
        mode='segmentation',
    )
    model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

    obs = torch.rand(8, 24, 3, 40, 40).cuda()

    data = model.forward(obs, optimizer=optimizer)
    # data['loss'].backward()

    # print(loss)

    exit('success')