import torch

class GRU4Rec(torch.nn.Module):
    def __init__(self, num_items, hidden_size, num_layers=1,
                 dropout_hidden=0, dropout_input=0, emb_size=128, **kwargs):
        super(GRU4Rec, self).__init__()

        self.num_items = num_items
        
        hidden = torch.zeros(num_layers, hidden_size, requires_grad=True)
        self.register_buffer("hidden", hidden) #register buffer is needed to move the tensor to the right device

        self.inp_dropout = torch.nn.Dropout(p=dropout_input)

        self.h2o = torch.nn.Linear(hidden_size, num_items+1)

        self.look_up = torch.nn.Embedding(num_items+1, emb_size)
        
        self.gru = torch.nn.GRU(emb_size, hidden_size, num_layers, dropout=dropout_hidden, batch_first=True)

    def forward(self, input_seqs, poss_item_seqs):
        '''
        TODO: rivedere
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        embedded = self.look_up(input_seqs)

        embedded = self.inp_dropout(embedded)

        output, hidden = self.gru(embedded, torch.tile(self.hidden, (1, input_seqs.shape[0], 1)))

        scores = self.h2o(output)

        scores = scores[:, -poss_item_seqs.shape[1]:, :]

        scores = torch.gather(scores, -1, poss_item_seqs) # Get scores for items in poss_item_seqs

        return scores