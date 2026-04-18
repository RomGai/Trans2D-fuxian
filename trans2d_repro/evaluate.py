import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_watchlist_batch
from data.dataset import WatchlistRankingDataset
from models.trans2d import Trans2DModel
from utils.checkpointing import load_checkpoint
from utils.metrics import ranking_metrics_grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data"]["val_path"] if args.split == "val" else cfg["data"]["test_path"]
    ds = WatchlistRankingDataset(
        path=data_path,
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

    all_logits, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["attributes"].to(device),
                batch["attention_mask"].to(device),
                batch["candidate_positions"].to(device),
            )
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"])
            all_groups.append(batch["group_ids"])

    metrics = ranking_metrics_grouped(torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_groups), cfg["metrics"]["ks"])
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
