import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model
        self.pe: torch.Tensor = self._get_positional_encoding(d_model, height, width)

    def _get_positional_encoding(self, d_model, width, height):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        Returns:
            Tensor with positional encodings added, of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        # Ensure the input has the correct number of channels
        assert self.d_model == channels, "Dimension mismatch: d_model and input channels must be the same"
        # Add positional encodings to the input tensor
        x = x + self.pe.unsqueeze(0) #the unsqueeze() might not be necessary, idk
        # cax = plt.matshow(x[0][1600])
        # plt.gcf().colorbar(cax)
        # plt.imshow(self.pe[100], cmap = "gray")
        return x

class InputEmbeddings(nn.Module):
    def __init__(self, in_channels=1664, out_dim=768):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.projection = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        # x shape: [batch_size, 1664, 12, 25]
        batch_size = x.size(0)
        
        # Reshape: [batch_size, 1664, 12, 25] -> [batch_size, 1664, 300]
        x = x.view(batch_size, self.in_channels, -1)
        
        # Transpose: [batch_size, 1664, 300] -> [batch_size, 300, 1664]
        x = x.transpose(1, 2)
        
        # Project: [batch_size, 300, 1664] -> [batch_size, 300, 768]
        x = self.projection(x)
        
        return x  # Shape: [batch_size, sequence_length, embedding_dim]