import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    #TODO: default values?
    def __init__(self, num_items, lookback, hidden_units, dropout_rate, num_blocks, num_heads, **kwargs):
        super(SASRec, self).__init__()

        self.item_num = num_items

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0) #+1 because padding
        self.pos_emb = torch.nn.Embedding(lookback, hidden_units) # TO IMPROVE #TODO?
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(hidden_units,
                                                            num_heads,
                                                            dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            #TODO: torch function instead of self-made?
            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):#TODO FEDE --> log_seqs: (type, gpu/position)
        #seqs = self.item_emb(torch.LongTensor(log_seqs).to(next(self.parameters()).device))
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = torch.tile(torch.arange(log_seqs.shape[1], device=next(self.parameters()).device), [log_seqs.shape[0], 1]) #)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=next(self.parameters()).device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
    
    def forward(self, input_seqs, poss_item_seqs):
        log_feats = self.log2feats(input_seqs).unsqueeze(2)
        # unsqueeze(2) to make it (B, T, 1, E) for broadcasting with poss_item_embs (B, T, I, E)

        #poss_item_embs = self.item_emb(torch.LongTensor(poss_item_seqs).to(next(self.parameters()).device))
        poss_item_embs = self.item_emb(poss_item_seqs)

        # Use only last timesteps in poss_item_seqs --> cut log_feats to match poss_item_embs
        log_feats = log_feats[:, -poss_item_embs.shape[1]:, :, :] # (B, T, 1, E)

        # Check that all dimensions except the second-to-last one must match for broadcasting
        assert log_feats.shape[0] == poss_item_embs.shape[0]
        assert log_feats.shape[1] == poss_item_embs.shape[1]
        assert log_feats.shape[3] == poss_item_embs.shape[3]
        
        poss_item_logits = (log_feats * poss_item_embs).sum(dim=-1)
        
        return poss_item_logits # pos_pred, neg_pred
    
    # poss_item_logits.shape = (num_users_in_batch, num_items_in_poss_item_seqs)
    
    # positive/negative items = [1,0,0,0] len(...) = num_items_in_poss_item_seqs

    # cross_entropy([0.99,0.01,0.2], [1,0,0])
    # cross_entropy([0.99][1]) + cross_entropy([0.01,0.2],[0,0])

    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(next(self.parameters()).device))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(next(self.parameters()).device))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits # pos_pred, neg_pred

    # def predict(self, user_ids, log_seqs, item_indices): # for inference
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     item_embs = self.item_emb(torch.LongTensor(item_indices).to(next(self.parameters()).device)) # (U, I, C)

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     # preds = self.pos_sigmoid(logits) # rank same item list for different users

    #     return logits # preds # (U, I)
    