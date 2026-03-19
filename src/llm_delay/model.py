from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# ─────────────────────────────────────────────
# Trajectory → LLM hidden space projection
# ─────────────────────────────────────────────

class CrossModalityAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Regression head
# ─────────────────────────────────────────────

class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# Ana model
# ─────────────────────────────────────────────

class LLMDelayRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        traj_dim: int,
        llm_hidden_dim: int | None = None,
        adapter_dropout: float = 0.1,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.llm.config.use_cache = False

        for p in self.llm.parameters():
            p.requires_grad = False

        hidden_size = self.llm.get_input_embeddings().weight.shape[1]
        if llm_hidden_dim is not None and hidden_size != llm_hidden_dim:
            raise ValueError(
                f"Config llm_hidden_dim={llm_hidden_dim} ama "
                f"model hidden_size={hidden_size} — uyuşmuyor."
            )

        self.input_embeddings = self.llm.get_input_embeddings()
        self.adapter  = CrossModalityAdapter(traj_dim, hidden_size, dropout=adapter_dropout)
        self.reg_head = RegressionHead(hidden_size, dropout=head_dropout)
        self._backbone = self.llm.model

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token id'leri embedding vektörlerine çevirir. [B, L] → [B, L, H]"""
        return self.input_embeddings(input_ids)

    def forward(
        self,
        prompt_ids:    torch.Tensor,          # [B, text_len]
        st1_ids:       torch.Tensor,          # [B, s1]
        st2_ids:       torch.Tensor,          # [B, s2]
        st3_ids:       torch.Tensor,          # [B, s3]
        st4_ids:       torch.Tensor,          # [B, s4]
        focusing_emb:  torch.Tensor,          # [B, traj_dim]
        active_embs:   torch.Tensor,          # [B, max_active, traj_dim]
        prior_embs:    torch.Tensor,          # [B, max_prior,  traj_dim]
        active_mask:   torch.Tensor,          # [B, max_active]  bool
        prior_mask:    torch.Tensor,          # [B, max_prior]   bool
        prompt_attention_mask: torch.Tensor | None = None,

    ) -> torch.Tensor:                        # [B]

        device = prompt_ids.device
        B      = prompt_ids.shape[0]

        # ── Token embedding ───────────────────────────────────────────
        z_prompt = self._embed_ids(prompt_ids)   # [B, text_len, H]
        z_st1    = self._embed_ids(st1_ids)       # [B, s1, H]
        z_st2    = self._embed_ids(st2_ids)       # [B, s2, H]
        z_st3    = self._embed_ids(st3_ids)       # [B, s3, H]
        z_st4    = self._embed_ids(st4_ids)       # [B, s4, H]

        # ── Trajectory projection: traj_dim → hidden_size ────────────
        z_f = self.adapter(focusing_emb).unsqueeze(1)   # [B, 1, H]
        z_a = self.adapter(active_embs)                  # [B, max_active, H]
        z_p = self.adapter(prior_embs)                   # [B, max_prior,  H]


        batch_embeds: list[torch.Tensor] = []

        for i in range(B):
            aa = z_a[i][active_mask[i]]   # [n_active_i, H] — sıfır da olabilir
            pp = z_p[i][prior_mask[i]]    # [n_prior_i,  H] — sıfır da olabilir

            seq = torch.cat(
                [
                    z_prompt[i],  # uçuş + hava durumu metni
                    z_st1[i],     # "...focus trajectory: {"
                    z_f[i],       # focusing embedding
                    z_st2[i],     # "} ...active trajectories: {"
                    aa,           # active embeddings (boşsa 0 token)
                    z_st3[i],     # "} ...prior trajectories: {"
                    pp,           # prior embeddings  (boşsa 0 token)
                    z_st4[i],     # "} Missing types..."
                ],
                dim=0,
            )
            batch_embeds.append(seq)

        # ── Sağdan padding ile batch tensor'u ────────────────────────
        max_len     = max(s.shape[0] for s in batch_embeds)
        hidden_size = batch_embeds[0].shape[1]

        inputs_embeds  = torch.zeros(B, max_len, hidden_size,
                                     dtype=batch_embeds[0].dtype, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for i, seq in enumerate(batch_embeds):
            L = seq.shape[0]
            inputs_embeds[i,  :L] = seq
            attention_mask[i, :L] = 1

        # ── Backbone forward ─────────────────────────────────────────
        outputs = self._backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state                          # [B, max_len, H]
        last_idx    = attention_mask.sum(dim=1) - 1                     # [B]
        h           = last_hidden[torch.arange(B, device=device), last_idx]  # [B, H]

        return self.reg_head(h)   # [B]