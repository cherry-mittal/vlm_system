import torch
import torch.nn as nn
from typing import Optional, Tuple

class siglipVisionConfig:

  def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, num_channels=3, image_size=224, patch_size=16,
               layer_norm_eps=1e-6, attention_dropout=0.0, num_image_tokens: int = None, **kwargs):
    super().__init__()

    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.layer_norm_eps = layer_norm_eps
    self.attention_dropout = attention_dropout
    self.num_image_tokens = num_image_tokens


class siglipVisionEmbeddings(nn.Module):

  def __init__(self, config: siglipVisionConfig):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.image_size = config.image_size
    self.patch_size = config.patch_size

    self.patch_embeddings = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding="valid")
    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.num_positions = self.num_patches
    self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
    self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

  def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    _,_,height,width = pixel_values.shape
    patch_embeds = self.patch_embeddings(pixel_values)
    embeddings = patch_embeds.flatten(2)
    embeddings = embeddings.transpose(1,2)
    embeddings = embeddings + self.position_embeddings(self.position_ids)
    return embeddings

class siglipVisionEncoder(nn.Module):

  def __init__(self, config: siglipVisionConfig):
    super().__init__()

    self.config = config
    self.layers = nn.ModuleList([siglipoVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
    hidden_states = input_embeds
    for encoder_layer in self.layers:
      hidden_states = encoder_layer(hidden_states)

    return hidden_states

class siglipVisionEncoderLayer(nn.Module):

  def __init__(self, config: siglipVisionConfig):
    super().__init__()

    self.config = config
    self.embed_dim = config.hidden_size
    self.attention = siglipAttention(config)
    self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    self.mlp = siglipMLP(config)
    self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.norm1(hidden_states)
    hidden_states,_ = self.attention(hidden_states=hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

class siglipAttention(nn.Module):
  def __init__(self, config: siglipVisionConfig):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.embed_dim // self.num_heads
    self.scale = self.head_dim**-0.5
    self.dropout = config.attention_dropout
    self.q_ = nn.Linear(self.embed_dim, self.embed_dim)
    self.k_ = nn.Linear(self.embed_dim, self.embed_dim)
    self.v_ = nn.Linear(self.embed_dim, self.embed_dim)
    self.out_ = nn.Linear(self.embed_dim, self.embed_dim)

  def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch_size, seq_len,_ = hidden_states.size()
    query_state = self.q_(hidden_states)
    key_state = self.k_(hidden_states)
    value_state = self.v_(hidden_states)

    query_state = query_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    key_state = key_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    value_state = value_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    attn_weights = torch.matmul(query_state, key_state.transpose(2,3)) * self.scale
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_state)
    attn_output = attn_output.transpose(1,2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
    attn_output = self.out_(attn_output)

    return attn_output, attn_weights


class siglipMLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
    hidden_states = self.fc2(hidden_states)
    return hidden_states


class siglipVisionTransformer(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.embeddings = siglipVisionEmbeddings(config)
    self.encoder = siglipVisionEncoder(config)
    self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    hidden_state = self.embeddings(pixel_values)
    last_hidden_state = self.encoder(inputs_embeds=hidden_states)
    last_hidden_state = self.norm(last_hidden_state)
    return last_hidden_state

class siglipModel(nn.Module):
  def __init__(self, config: siglipVisionConfig):
    super().__init__()
    self.config = config
    self.trans_model = siglipVisionTransformer(config)

  def forward(self, pixel_values) -> Tuple:
    return self.trans_model(pixel_values=pixel_values)