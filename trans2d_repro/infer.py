import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_watchlist_batch
from data.dataset import WatchlistRankingDataset
from models.trans2d import Trans2DModel
from utils.checkpointing import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = WatchlistRankingDataset(
        path=args.input,
        attribute_names=cfg["data"]["attribute_names"],
        hash_mod=cfg["data"]["hash_mod"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_watchlist_batch)

    vocab_sizes = [cfg["data"]["vocab_sizes"][n] for n in cfg["data"]["attribute_names"]]
    model = Trans2DModel(vocab_sizes, cfg["model"]["d_model"], cfg["model"]["num_heads"], cfg["model"]["num_layers"], cfg["model"]["dropout"])
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["attributes"].to(device),
                batch["attention_mask"].to(device),
                batch["candidate_positions"].to(device),
            )
            probs = torch.sigmoid(logits).cpu()
            for gid, p, y in zip(batch["group_ids"].tolist(), probs.tolist(), batch["labels"].tolist()):
                outputs.append((gid, p, y))

    outputs.sort(key=lambda x: (x[0], -x[1]))
    print("group_id\tprob\tlabel")
    for gid, prob, label in outputs:
        print(f"{gid}\t{prob:.6f}\t{int(label)}")


if __name__ == "__main__":
    main()
