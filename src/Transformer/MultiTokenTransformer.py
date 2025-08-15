# MultiTokenTransformer.py
"""
Defines the MultiTokenTransformer model architecture.

This model extends the concept of the TwoTokenTransformer by dividing the input
keypoint and CNN feature vectors into multiple smaller sub-vectors (tokens).
This allows the transformer's self-attention mechanism to learn finer-grained
relationships between different parts of the pose and different aspects of the
image features.

Key Architectural Features:
- Splits the keypoint vector into 17 tokens, each representing an (x, y, confidence)
  tuple for a single keypoint.
- Splits the CNN vector into one or more tokens.
- Each sub-vector is linearly projected into the model's embedding space (d_model).
- A special learnable [CLS] (classification) token is prepended to the sequence.
- Learnable positional encodings are added to give the model a sense of token order.
- The entire sequence is processed by a Transformer Encoder.
- The final output of the [CLS] token is used for classification, aggregating
  information from the entire sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTokenTransformer(nn.Module):
    """
    A Transformer that processes multiple tokens derived from keypoint and CNN features.
    """
    def __init__(self,
                 kp_dim: int = 51,
                 cnn_dim: int = 512,
                 num_classes: int = 47,
                 d_model: int = 256,
                 nhead: int = 1,
                 n_layers: int = 1,
                 dim_ff: int = 512,  # Feed-forward dimension
                 dropout: float = 0.35):
        """
        Initializes the MultiTokenTransformer model.

        Args:
            kp_dim (int): Dimensionality of the full keypoint vector (e.g., 17 keypoints * 3 values = 51).
            cnn_dim (int): Dimensionality of the full CNN feature vector.
            num_classes (int): The number of output classes.
            d_model (int): The embedding dimension for each token in the transformer.
            nhead (int): The number of multi-head attention heads.
            n_layers (int): The number of layers in the transformer encoder.
            dim_ff (int): The dimension of the feed-forward network within the transformer.
            dropout (float): The dropout rate used in the transformer layers.
        """
        super().__init__()

        # --- Define how to split the input vectors into smaller tokens ---
        # Each tuple element defines the size of a chunk to be treated as a token.
        # Here, we split the 51-dim keypoint vector into 17 tokens of size 3 (x, y, conf).
        self.kp_splits = (3,) * 17
        # Here, we treat the entire 512-dim CNN vector as a single token.
        self.cnn_splits = (512,)
        # Total number of tokens = 1 (for [CLS]) + 17 (for KP) + 1 (for CNN) = 19
        self.n_tokens = 1 + len(self.kp_splits) + len(self.cnn_splits)

        # --- Projection Layers ---
        # Create a list of linear layers to project each token into d_model space.
        self.kp_proj = nn.ModuleList(
            [nn.Linear(d, d_model) for d in self.kp_splits])
        self.cnn_proj = nn.ModuleList(
            [nn.Linear(d, d_model) for d in self.cnn_splits])

        # --- Special Tokens and Encodings ---
        # A learnable [CLS] token, similar to BERT, to aggregate sequence information.
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        # Learnable positional encodings to provide sequence order information.
        # Shape: (1, n_tokens, d_model) to be broadcast across the batch.
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_tokens, d_model))

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True  # Expects input shape [Batch, Sequence, Features]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Classification Head ---
        # A final classifier that operates on the output of the [CLS] token.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, kp: torch.Tensor, cnn: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            kp (torch.Tensor): Keypoint features tensor of shape [B, kp_dim].
            cnn (torch.Tensor): CNN features tensor of shape [B, cnn_dim].

        Returns:
            torch.Tensor: Logits for each class, shape [B, num_classes].
        """
        B = kp.size(0)  # Get the batch size

        # --- 1. Slice and Project Keypoint Tokens ---
        kp_tokens = []
        idx = 0
        for split, proj in zip(self.kp_splits, self.kp_proj):
            # Slice the vector to get the chunk for the current token
            sub_vector = kp[:, idx:idx + split]
            # Project it to d_model and add a sequence dimension
            kp_tok = proj(sub_vector).unsqueeze(1)  # Shape: (B, 1, d_model)
            kp_tokens.append(kp_tok)
            idx += split

        # --- 2. Slice and Project CNN Tokens ---
        cnn_tokens = []
        idx = 0
        for split, proj in zip(self.cnn_splits, self.cnn_proj):
            sub_vector = cnn[:, idx:idx + split]
            cnn_tok = proj(sub_vector).unsqueeze(1) # Shape: (B, 1, d_model)
            cnn_tokens.append(cnn_tok)
            idx += split

        # --- 3. Assemble the Full Token Sequence ---
        # Expand the [CLS] token to match the batch size
        cls_tok = self.cls.expand(B, -1, -1)
        # Concatenate all tokens: [CLS, KP_Token_1, ..., KP_Token_17, CNN_Token_1]
        x = torch.cat([cls_tok, *kp_tokens, *cnn_tokens], dim=1)  # Shape: (B, n_tokens, d_model)

        # --- 4. Add Positional Encoding ---
        # Add the learnable positional encodings to the token embeddings.
        x = x + self.positional_encoding.expand(B, -1, -1)

        # --- 5. (Optional) Token Dropout for Regularization ---
        if self.training:
            # Create a random mask to drop ~10% of tokens by setting them to zero.
            # This forces the model to not rely too heavily on any single token.
            mask = torch.rand(B, self.n_tokens, device=x.device) < 0.1
            x[mask] = 0.0

        # --- 6. Pass through Transformer Encoder ---
        x = self.encoder(x)  # Shape remains (B, n_tokens, d_model)

        # --- 7. Extract [CLS] Token and Classify ---
        # The output corresponding to the [CLS] token at position 0 is used for classification.
        cls_out = x[:, 0]
        return self.head(cls_out)