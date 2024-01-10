import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl
from .rnn_nn import Embedding, RNN, LSTM


# class RNNClassifier(pl.LightningModule):
class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        # self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        
        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
        if use_lstm:
            self.rnn = LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        else:
            self.rnn = RNN(input_size=embedding_dim, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
        embedded_seq = self.embedding(sequence)

        if lengths is not None:
            embedded_seq = pack_padded_sequence(embedded_seq, lengths.cpu())
            
        rnn_output, (hidden, cell) = self.rnn(embedded_seq)    
        
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)  
 
        last_output = hidden[0, :, :]
        logits = self.fc(last_output) 
        output = self.sigmoid(logits).squeeze()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
    
#     def training_step(self, batch, batch_idx):
#         inputs = batch["data"]
#         targets = batch["label"]
#         lengths = batch["lengths"]
#         outputs = self.forward(inputs, lengths)
#         loss = nn.BCELoss()(outputs, targets)
#         self.log('train_loss', loss, logger=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs = batch["data"]
#         targets = batch["label"]
#         lengths = batch["lengths"]
#         outputs = self.forward(inputs, lengths)
#         loss = nn.BCELoss()(outputs, targets)
#         self.log('val_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.01)
