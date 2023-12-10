import os
import torch
from model.fusion_module import *
from model.transformer import Transformer
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_item = n_item
        self.device = args.device
        self.emb_size = 512
        self.batch_size = args.train_batch_size
        self.node_dropout = [0.1][0]
        self.mess_dropout = [0.1,0.1,0.1]

        self.norm_adj = norm_adj

        self.layers = eval('[512,512,512]') # 3 layers
        # self.layers = eval('[512,512]') # 2 layers
        # self.layers = eval('[512]') # 1 layer

        self.decay = eval('[1e-5]')[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_gc2_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                       layers[k+1])))})
            weight_dict.update({'b_gc2_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, items, all_vision, all_text, drop_flag=True):
        # all_vision/all_text shape (num_items, fusion_embed_dim)
        # print(self.device, items.device, all_vision.device, all_text.device)

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        
        ego_embeddings = torch.cat([all_vision, all_text], 0)
        oge_embeddings = torch.cat([all_text, all_vision], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):

            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            edis_embeddings = torch.sparse.mm(A_hat, oge_embeddings)
            
            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]
        
            # transformed edis messages of neighbors.
            mus_embeddings = torch.matmul(edis_embeddings, self.weight_dict['W_gc2_%d' % k]) \
                                             + self.weight_dict['b_gc2_%d' % k]

            sum_embeddings = sum_embeddings + mus_embeddings

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bii_embeddings = torch.mul(oge_embeddings, edis_embeddings)
            bi_embeddings = bi_embeddings + bii_embeddings
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.sum(all_embeddings, dim=0)

        v_g_embeddings = all_embeddings[:self.n_item, :]
        t_g_embeddings = all_embeddings[self.n_item:, :]
        # fusion_g_embeddings = torch.sum(torch.stack([t_g_embeddings, v_g_embeddings]), 0)

        v_embeddings = v_g_embeddings[items]
        t_embeddings = t_g_embeddings[items]
        
        return v_embeddings, t_embeddings

class FusionEncoder(nn.Module):
    def __init__(self, args, **kwargs):

        super().__init__()
        self.args = args

        self.input_transform = FusionInputTransform(args, **args)
        self.embedding = FusionEmbedding(args, **args)
        self.encoder_blocks = Transformer(args.fusion_embed_dim,
                                          args.fusion_layers,
                                          args.fusion_heads,
                                          args.fusion_feedforward_dim,
                                          args.fusion_dropout)
        self.output_transform = FusionOutputTransform(args, **args)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(self, save_path):
        state_file = os.path.join(save_path, "fusion_model.pth")
        state = {
            "input": self.input_transform.state_dict(),
            "embedding": self.embedding.state_dict(),
            "blocks": self.encoder_blocks.state_dict(),
            "output": self.output_transform.state_dict()
        }
        torch.save(state, state_file)

    def from_pretrained(self, load_path, map_location):
        state_file = os.path.join(load_path, "fusion_model.pth")
        if not os.path.exists(state_file):
            return
        state = torch.load(state_file, map_location=map_location)
        del state["embedding"]["position_embeddings.weight"]
        self.input_transform.load_state_dict(state["input"], strict=False)
        self.embedding.load_state_dict(state["embedding"], strict=False)
        self.encoder_blocks.load_state_dict(state["blocks"], strict=False)
        self.output_transform.load_state_dict(state["output"], strict=False)

    def forward(self, input_ids, vision, vision_mask, text, text_mask, mode=None):
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)
        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)  # [N, L, D]
        mask = (input_ids >= 3)     # [N, L]

        return hidden, mask


class FusionEncoderForMaskedLM(FusionEncoder):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.path = args.path
        self.n_items = args.train_item_num + 3
        self.ce_mask_ratio = args.ce_mask_ratio
        self.modality_embedding = ModalityEmbedding(args, **args)
        self.R = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        self.adj_max, self.norm_adj_mat, self.mean_adj_mat = self.get_adj_mat()
        self.global_embedding = NGCF(self.n_items, self.norm_adj_mat, args)
    
    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_items + self.n_items, self.n_items + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_items, self.n_items:] = R
        adj_mat[self.n_items:, :self.n_items] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape)

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            if rowsum.max() == 0.:
                rowsum += 1
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix')
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def get_adj_mat(self):
        try:
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def modality_embedding_init(self, dataset, clip_model):
        vision, vision_mask, text, text_mask = dataset.full_item_features()

        vision = vision.unsqueeze(dim=1).to(self.args.device)
        vision_mask = vision_mask.unsqueeze(dim=1).to(self.args.device)
        text = text.unsqueeze(dim=1).to(self.args.device)
        text_mask = text_mask.unsqueeze(dim=1).to(self.args.device)

        with torch.no_grad():
            vision, vision_mask = clip_model(vision, vision_mask, mode="vision", format=self.args.train_vision_format)
            text, text_mask = clip_model(text, text_mask, mode="text", format=self.args.train_text_format)

        vision = vision.squeeze(dim=1)  # [N, D]
        vision_mask = vision_mask.squeeze(dim=1)  # [N]
        text = text.squeeze(dim=1)  # [N, D]
        text_mask = text_mask.squeeze(dim=1)  # [N]

        self.modality_embedding.weight_init(vision, vision_mask, text, text_mask)

    def forward(self,
                input_ids,
                vision=None,
                vision_mask=None,
                text=None,
                text_mask=None,
                mode=None):
        if mode == "mlm":
            return self.mlm(input_ids)
        elif mode == "pred":
            return self.prediction(input_ids)
        else:
            raise NotImplementedError("FusionEncoderForMaskedLM Forward Mode Error!")

    def modality_embedding_hidden(self):
        input_ids = torch.arange(3, self.modality_embedding.item_num + 3, dtype=torch.int, device=self.args.device)
        input_ids = input_ids.unsqueeze(dim=-1)
        # torch.Size([12101, 1])

        vision, vision_mask, text, text_mask = self.modality_embedding(input_ids)
        embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = hidden[:, -1, :]

        return hidden

    def mlm(self, input_ids):
        # input_ids: [N, L] 1024x20
        ce_mask = torch.full_like(input_ids, self.ce_mask_ratio, dtype=torch.float) * (input_ids != 0)
        ce_mask = torch.bernoulli(ce_mask).type(torch.bool)
        ce_mask[:, -1] = 1
        masked_ids = input_ids.masked_fill(ce_mask, 1)
        replace_mask = torch.full_like(input_ids, 0.1, dtype=torch.float) * ce_mask
        replace_mask = torch.bernoulli(replace_mask).type(torch.bool)
        random_ids = torch.randint_like(input_ids, low=3, high=self.args.train_item_num + 2)
        masked_ids = torch.where(replace_mask, random_ids, masked_ids)
        restore_mask = torch.full_like(input_ids, 0.1, dtype=torch.float) * ce_mask
        restore_mask = torch.bernoulli(restore_mask).type(torch.bool)
        masked_ids = torch.where(restore_mask, input_ids, masked_ids)
        masked_ids[:, -1] = 1
        
        # item id starts at 3
        all_ids = torch.arange(0, self.n_items, dtype=torch.int, device=self.args.device)
        all_vision, all_vision_mask, all_text, all_text_mask = self.modality_embedding(all_ids)
        # embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        vision, vision_mask, text, text_mask = self.modality_embedding(masked_ids)
        mask = torch.stack([vision_mask, text_mask], dim=2)  # [N, M]
        # get embedding
        # embed, mask = self.input_transform(vision, vision_mask, text, text_mask)
        # embed: [N, L, M, D] 1024x20x2x512
        # mask: [N, L, M] 1024x20x2

        # vision_embed = embed[:, :, 0, :]  # [N, L, D]
        # text_embed = embed[:, :, 1, :]  # [N, L, D]
        vision_embed = []
        text_embed = []

        # item_i_text <-> item_i_vision
        for i in range(input_ids.size(1)):
            v, t = self.global_embedding(masked_ids[:, i], all_vision, all_text)
            vision_embed.append(v)
            text_embed.append(t)

        vision_embed = torch.stack(vision_embed, dim=1)
        text_embed = torch.stack(text_embed, dim=1)
        embed = torch.stack([vision_embed, text_embed], dim=2) # 1024x20x2x512

        # put position embedding, modality embedding and replacement embedding together
        embed = self.embedding(masked_ids, embed) # 1024x20x2x512
        # input for sequence encoder
        inputs = torch.flatten(embed, 1, 2) # 1024x40x512

        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = torch.masked_select(hidden, ce_mask.unsqueeze(-1)).reshape(-1, hidden.shape[-1])

        embedding_hidden = self.modality_embedding_hidden()

        hidden = nn.functional.normalize(hidden, dim=-1)
        embedding_hidden = nn.functional.normalize(embedding_hidden, dim=-1)

        predict = torch.matmul(hidden, embedding_hidden.T) / self.args.contrastive_temperature  # [L, V]

        labels = torch.masked_select(input_ids - 3, ce_mask).reshape(-1)
        return predict, labels

    def prediction(self, input_ids):
        labels = input_ids[:, -1].detach()  # [N]
        input_ids = input_ids.clone()
        input_ids[:, -1] = 1

        all_ids = torch.arange(0, self.n_items, dtype=torch.int, device=self.args.device)
        all_vision, all_vision_mask, all_text, all_text_mask = self.modality_embedding(all_ids)
        vision, vision_mask, text, text_mask = self.modality_embedding(input_ids)
        mask = torch.stack([vision_mask, text_mask], dim=2)
        # embed, mask = self.input_transform(vision, vision_mask, text, text_mask)

        vision_embed = []
        text_embed = []
        for i in range(input_ids.size(1)):
            v, t = self.global_embedding(input_ids[:, i], all_vision, all_text)
            vision_embed.append(v)
            text_embed.append(t)
        vision_embed = torch.stack(vision_embed, dim=1)
        text_embed = torch.stack(text_embed, dim=1)
        embed = torch.stack([vision_embed, text_embed], dim=2)

        embed = self.embedding(input_ids, embed)
        inputs = torch.flatten(embed, 1, 2)
        hidden = self.encoder_blocks(inputs, ~mask.reshape(inputs.shape[0], -1))
        hidden = hidden.reshape(embed.shape)
        hidden = self.output_transform(hidden, mask)
        hidden = hidden[:, -1, :]

        embedding_hidden = self.modality_embedding_hidden()

        hidden = nn.functional.normalize(hidden, dim=-1)
        embedding_hidden = nn.functional.normalize(embedding_hidden, dim=-1)
        predict = torch.matmul(hidden, embedding_hidden.T)  # [N, V]

        return predict, labels - 3
