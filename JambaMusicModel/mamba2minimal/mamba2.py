"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(huggingface_model_id: str, device: Device = None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2LMHeadModel(args, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: InferenceCache | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.
               If h is provided, assumes single-step inference.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        # --- Inference path ---
        if h is not None and u.shape[1] == 1:
            # If hidden state h is provided and input is single step, call step()
            return self.step(u, h)

        # --- Training/Full Sequence Path ---
        batch_size, seqlen, _ = u.shape
        assert seqlen % self.args.chunk_size == 0, f"Sequence length ({seqlen}) must be a multiple of chunk_size ({self.args.chunk_size})"

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Convolution part
        # Need to handle potential initial conv_state if provided (though typically None in fwd)
        # The original code computed conv_state *after* the conv, let's keep it for consistency if needed
        # For forward pass, we apply convolution directly
        xBC_conv = self.conv1d(xBC.transpose(1, 2)) # (batch, d_inner + 2*d_state, seqlen + d_conv - 1)

        # Apply silu and truncate back to seqlen
        xBC_silu = silu(xBC_conv[:, :, self.args.d_conv - 1 : self.args.d_conv - 1 + seqlen]) # (b, d, l)
        xBC_silu = xBC_silu.transpose(1, 2) # (b, l, d)

        x, B, C = torch.split(
            xBC_silu, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)

        # Prepare inputs for ssd
        x_dt = x * dt.unsqueeze(-1)  # (b, l, h, p)
        A_dt = A * dt             # (b, l, h)
        B_re = rearrange(B, "b l n -> b l 1 n") # (b, l, 1, n)
        C_re = rearrange(C, "b l n -> b l 1 n") # (b, l, 1, n)

        # Call ssd function
        # initial_states might be derivable from h if provided, but typically None for forward
        initial_states = h.ssm_state if h is not None else None # Though h is usually None here
        y, final_ssm_state = ssd(
            x_dt,
            A_dt,
            B_re,
            C_re,
            self.args.chunk_size,
            initial_states=initial_states, # Pass initial state if available
            device=self.device,
        )

        # Apply skip connection with D
        y = y + x * self.D.unsqueeze(-1) # D is (h,), x is (b,l,h,p) -> broadcast
        y = rearrange(y, "b l h p -> b l (h p)")

        # Apply norm and output projection
        y = self.norm(y, z) # Gated RMSNorm
        y = self.out_proj(y)

        # Construct cache for return (even if input h was None)
        # Reconstruct conv_state: The state *before* the final token's convolution input is processed.
        # This requires the input `xBC` before convolution.
        # The original code computed conv_state differently, let's adapt to match step logic if possible.
        # step() logic updates conv_state *before* conv. Forward pass conv_state is tricky.
        # Let's use the final `xBC` values to fill the cache, similar to how step would leave it.
        # Pad or truncate xBC seqlen to d_conv - this seems correct based on original step logic prep
        # conv_state_for_cache = F.pad(
        #     rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - seqlen % self.args.d_conv, 0) # Padding needs care
        # )
        # Simplified: Use the last d_conv steps of the input xBC. If seqlen < d_conv, pad.
        conv_state_data = xBC[:, -self.args.d_conv:, :].transpose(1, 2) # (b, d, min(l, d_conv))
        if seqlen < self.args.d_conv:
             conv_state_for_cache = F.pad(conv_state_data, (self.args.d_conv - seqlen, 0))
        else:
             conv_state_for_cache = conv_state_data


        updated_h = InferenceCache(conv_state_for_cache, final_ssm_state)
        return y, updated_h

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # --- Convolution State Update ---
        # 1. Clone the original conv_state to avoid in-place modification of the input cache's tensor
        new_conv_state_val = h.conv_state.clone()

        # 2. Apply updates to the cloned tensor
        new_conv_state_val.copy_(torch.roll(new_conv_state_val, shifts=-1, dims=-1))
        new_conv_state_val[:, :, -1] = xBC # xBC is (batch, d_inner + 2*d_state)
        
        # 3. Perform convolution using the updated (cloned) conv_state
        convolved_xBC = torch.sum(
            new_conv_state_val * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        convolved_xBC += self.conv1d.bias
        processed_xBC = silu(convolved_xBC)

        x_from_conv, B_from_conv, C_from_conv = torch.split(
            processed_xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # --- SSM State Update ---
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        
        x_for_ssm = rearrange(x_from_conv, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B_from_conv, x_for_ssm)
        
        # Calculate the new ssm_state value (based on the original h.ssm_state)
        new_ssm_state_val = h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx
        
        # Create a new InferenceCache object with the new conv_state and new ssm_state.
        updated_h = InferenceCache(conv_state=new_conv_state_val, ssm_state=new_ssm_state_val)

        # --- Output Calculation ---
        y = torch.einsum("bhpn, bn -> bhp", updated_h.ssm_state, C_from_conv)
        y = y + rearrange(self.D, "h -> h 1") * x_for_ssm
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z) # z is from the initial in_proj
        y = self.out_proj(y)

        return y.unsqueeze(1), updated_h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        # If no initial state provided, create a 5D zero tensor for the first chunk's input state
        initial_states_for_cat = torch.zeros_like(states[:, :1]) # (b, 1, h, p, n)
    else:
        # If initial_state (b, h, p, n) is provided, unsqueeze it to add the chunk dimension (dim=1)
        initial_states_for_cat = initial_states.unsqueeze(1) # (b, 1, h, p, n)

    # Concatenate the initial state with the intra-chunk states
    # Original line causing error: states = torch.cat([initial_states, states], dim=1)
    states_with_initial = torch.cat([initial_states_for_cat, states], dim=1) # Now concatenating two 5D tensors

    # Compute decay_chunk based on A_cumsum
    # Note: Padding A_cumsum[:, :, :, -1] with (1, 0) means the first element corresponds to the initial state influence
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)) # (b, h, c+1, c+1)

    # Apply recurrence using the combined states (shape: b, c+1, h, p, n)
    # Einsum: decay_chunk(b,h,z,c') * states_with_initial(b,c,h,p,n) -> new_states(b,z,h,p,n) where c' and c match
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states_with_initial)

    # Extract the states at chunk boundaries (excluding the final state which is stored separately)
    # and the final state after processing all chunks.
    states_after_recurrence = new_states[:, :-1] # Shape (b, z=c, h, p, n)
    final_state = new_states[:, -1] # Shape (b, h, p, n)

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum) # (b, h, c, l)
    # Use the states *after* the recurrence relation has been applied
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states_after_recurrence, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)
