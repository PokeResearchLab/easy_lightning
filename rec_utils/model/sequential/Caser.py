

import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils import activation_getter


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


class Caser(nn.Module):     #TODO check fede, why there is never .todevice?
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, L, dims, num_hor, num_ver, drop_rate, ac_conv, ac_fc, num_items, num_users, *args, **kwargs):
        super(Caser, self).__init__()

        # init args
        self.num_hor = num_hor
        self.num_ver = num_ver
        self.drop_ratio = drop_rate
        self.ac_conv = activation_getter[ac_conv]   #activation function for convolution layer (i.e., phi_c in paper)
        self.ac_fc = activation_getter[ac_fc]       #activation function for fully-connected layer (i.e., phi_a in paper)

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users+1, dims)
        self.item_embeddings = nn.Embedding(num_items +1, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.num_ver, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.num_hor, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.num_ver * dims
        self.fc1_dim_h = self.num_hor * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        # self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        # self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        # self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        # self.b2.weight.data.zero_()


    def forward(self, input_seqs, poss_item_seqs):#, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        input_seqs: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        print("input seqs", input_seqs)
        print(aaaa)
        item_embs = self.item_embeddings(input_seqs).unsqueeze(1)  # use unsqueeze() to get 4-D

       # user_emb = self.user_embeddings(user_var).squeeze(1)
        #user_emb WITHOUT the user_var
    


        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.num_ver:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.num_hor:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(poss_item_seqs)
        b2 = self.b2(poss_item_seqs)

        # if for_pred:
        #     w2 = w2.squeeze()
        #     b2 = b2.squeeze()
        #     res = (x * w2).sum(1) + b2
        # else:
        #     res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return x
