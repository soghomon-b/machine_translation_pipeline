#used trnasformer from: https://huggingface.co/spaces/alifalhasan/arabic2english/blob/main/src/train/transformer.py


import torch
from torch import nn


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        max_len,
        device,
    ):
        """
        Initializes the Transformer model.
        Args:
            embedding_size: Size of the embeddings.
            src_vocab_size: Size of the source vocabulary.
            trg_vocab_size: Size of the target vocabulary.
            src_pad_idx: Index of the padding token in the source vocabulary.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dropout: Dropout probability.
            max_len: Maximum sequence length.
            device: Device to place tensors on.
        """

        super(Transformer, self).__init__()
        # Embeddings for source and target sequences
        self.src_embeddings = nn.Embedding(src_vocab_size, embedding_size)
        self.src_positional_embeddings = nn.Embedding(max_len, embedding_size)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_positional_embeddings = nn.Embedding(max_len, embedding_size)
        self.device = device
        # Transformer layer
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
        )
        # Final fully connected layer
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        """
        Creates a mask to ignore padding tokens in the source sequence.
        Args:
            src: Source sequence tensor.
        Returns:
            src_mask: Mask tensor.
        """
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        """
        Forward pass of the Transformer model.
        Args:
            src: Source sequence tensor.
            trg: Target sequence tensor.
        Returns:
            out: Output tensor.
        """
        src_seq_length = src.shape[0]
        trg_seq_length = trg.shape[0]
        S = 1
        # Generate position indices for source and target sequences
        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, S)
            .to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, S)
            .to(self.device)
        )
        # Apply embeddings and dropout for source and target sequences
        embed_src = self.dropout(
            (self.src_embeddings(src) + self.src_positional_embeddings(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_embeddings(trg) + self.trg_positional_embeddings(trg_positions))
        )
        # Generate masks for source padding and target sequences
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )
        # Forward pass through Transformer
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        # Apply final fully connected layer
        out = self.fc_out(out)
        return out
