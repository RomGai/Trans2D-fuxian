# Trans2D / Attention2D (PyTorch Reproduction)

This project is a clean, modular PyTorch re-implementation of **"Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce"**.

## Task formulation
This is **watchlist candidate ranking**, not full-catalog next-item prediction.

For user snapshot time `t`:
1. Build history `S(t)` from past watchlist clicks + recently viewed items.
2. Let current watchlist snapshot be `WL_t` (dynamic candidate set).
3. For each candidate `c in WL_t`, append `c` to history, run Trans2D, output one click logit.
4. Train with binary labels inside each snapshot group (clicked=1, others=0).
5. Evaluate ranking quality **within each snapshot group** using Precision@k / Hit@k / NDCG@k for `k={1,2,5}`.

## Data format
Input files are JSONL where each row is one snapshot:

```json
{
  "user_id": "u1",
  "snapshot_id": "s12",
  "history": [
    {"item_id_hash": "itemA", "seller_id_hash": "seller7", "interaction_type": 1, "hour": 9, "day": 12, "weekday": 2},
    {"item_id_hash": "itemB", "seller_id_hash": "seller9", "interaction_type": 2, "hour": 10, "day": 12, "weekday": 2}
  ],
  "candidates": [
    {"attributes": {"item_id_hash": "itemX", "seller_id_hash": "seller2", "interaction_type": 1, "hour": 11, "day": 12, "weekday": 2}, "label": 1},
    {"attributes": {"item_id_hash": "itemY", "seller_id_hash": "seller3", "interaction_type": 1, "hour": 11, "day": 12, "weekday": 2}, "label": 0}
  ]
}
```

Notes:
- Raw IDs are optional; hash-compatible fields are deterministically hashed in preprocessing.
- `0` is reserved as padding index for all attributes.
- Missing attributes default to `0`.

## Modeling choices (paper-faithful assumptions)
Because some implementation details are underspecified in the paper, this repo documents the following assumptions:
1. **Candidate-expanded batching** is used (one sample per candidate), which exactly preserves "append candidate then score" semantics.
2. Attention2D builds three logits (`full_attn`, `time_attn`, `feature_attn`) and combines them with learnable scalar weights.
3. Causal masking is applied on sequence position dimension; padding masking on key positions.
4. Softmax is taken over flattened key pair dimension `(N*C)`.
5. Candidate representation is taken at candidate position, then mean-pooled over attributes before final linear logit.

## Project layout
See `trans2d_repro/` for source files:
- `data/`: preprocessing, dataset, collate
- `models/`: embeddings, Linear2D, Attention2D, Trans2D block/model
- `utils/`: metrics, seed, checkpointing
- `train.py`, `evaluate.py`, `infer.py`
- `tests/`: shape, masking, grouped metrics checks

## Train
```bash
python train.py --config config.yaml --save-path checkpoints/best.pt
```

## Evaluate
```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pt --split test
```

## Infer / rank candidates
```bash
python infer.py --config config.yaml --checkpoint checkpoints/best.pt --input data/test.jsonl
```

## Tiny forward-pass example
```python
import torch
from models.trans2d import Trans2DModel

vocab_sizes = [100, 100, 50, 20]  # C=4
model = Trans2DModel(vocab_sizes=vocab_sizes, d_model=16, num_heads=4, num_layers=1, dropout=0.1)

# [B=2, N=5, C=4]
attributes = torch.randint(1, 10, (2, 5, 4), dtype=torch.long)
attention_mask = torch.tensor([[1,1,1,1,1],[1,1,1,1,0]], dtype=torch.bool)
candidate_positions = torch.tensor([4, 3], dtype=torch.long)

logits = model(attributes, attention_mask, candidate_positions)
print(logits.shape)  # torch.Size([2])
```
