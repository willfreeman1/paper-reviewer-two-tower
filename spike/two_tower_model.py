"""
Small mixers that sit on frozen SPECTER2 fingerprints + OpenAlex extras.
SPECTER2 is never updated.

The mixer is residual: output = normalize(SPECTER2 + delta(extras)).
The last layer starts at zeros, so before any training the model is exactly
frozen SPECTER2 cosine. Extra features can help; they cannot throw away
the text fingerprint (the failure mode of a from-scratch 256-d MLP).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMixer(nn.Module):
    def __init__(self, text_dim, extra_dim, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(text_dim + extra_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, text_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, text, extras):
        h = self.drop(F.relu(self.fc1(torch.cat([text, extras], dim=-1))))
        return F.normalize(text + self.fc2(h), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        n_subfields,
        n_topics,
        text_dim=768,
        cat_dim=16,
        topic_dim=32,
        hidden=256,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.subfield = nn.Embedding(n_subfields + 1, cat_dim, padding_idx=0)
        self.topic = nn.Embedding(n_topics + 1, topic_dim, padding_idx=0)
        self.paper_mix = ResidualMixer(text_dim, cat_dim + topic_dim + 1, hidden, dropout)
        self.reviewer_mix = ResidualMixer(text_dim, cat_dim + 4, hidden, dropout)

    def encode_papers(self, text, subfield_id, topic_id, year_norm):
        extras = torch.cat(
            [
                self.subfield(subfield_id),
                self.topic(topic_id),
                year_norm.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.paper_mix(text, extras)

    def encode_reviewers(self, text, subfield_id, numerics):
        extras = torch.cat([self.subfield(subfield_id), numerics], dim=-1)
        return self.reviewer_mix(text, extras)
