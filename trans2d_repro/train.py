import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.collate import collate_watchlist_batch
from data.dataset import WatchlistRankingDataset
from models.trans2d import Trans2DModel
from utils.checkpointing import save_checkpoint
from utils.logging import log
from utils.metrics import ranking_metrics_grouped
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(path: str, cfg, shuffle: bool) -> DataLoader:
    ds = WatchlistRankingDataset(
        path=path,
        attribute_names=cfg["data"]["attribute_names"],
        hash_mod=cfg["data"]["hash_mod"],
        max_seq_len=cfg["data"]["max_seq_len"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_watchlist_batch,
    )


def evaluate_model(model, loader, device, ks):
    model.eval()
    all_logits, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for batch in loader:
            attrs = batch["attributes"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_pos = batch["candidate_positions"].to(device)
            logits = model(attrs, mask, cand_pos)

            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
            all_groups.append(batch["group_ids"].cpu())

    if not all_logits:
        return {f"NDCG@{k}": 0.0 for k in ks}

    return ranking_metrics_grouped(
        logits=torch.cat(all_logits),
        labels=torch.cat(all_labels),
        group_ids=torch.cat(all_groups),
        ks=ks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save-path", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    train_loader = make_loader(cfg["data"]["train_path"], cfg, shuffle=True)
    val_loader = make_loader(cfg["data"]["val_path"], cfg, shuffle=False)

    vocab_sizes = [cfg["data"]["vocab_sizes"][n] for n in cfg["data"]["attribute_names"]]
    model = Trans2DModel(
        vocab_sizes=vocab_sizes,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_metric = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if cfg["train"].get("lr_decay_by_10_from_epoch2", False) and epoch >= 2:
            for g in optimizer.param_groups:
                g["lr"] = cfg["train"]["lr"] * (0.1 ** (epoch - 1))

        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            attrs = batch["attributes"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_pos = batch["candidate_positions"].to(device)
            labels = batch["labels"].to(device)

            logits = model(attrs, mask, cand_pos)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        val_metrics = evaluate_model(model, val_loader, device, ks=cfg["metrics"]["ks"])
        val_ndcg5 = val_metrics.get("NDCG@5", 0.0)
        avg_loss = total_loss / max(steps, 1)
        log(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_NDCG@5={val_ndcg5:.4f}")

        if val_ndcg5 > best_metric:
            best_metric = val_ndcg5
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["train"]["early_stopping_patience"]:
                log("Early stopping triggered.")
                break

    if best_state is not None:
        save_checkpoint(args.save_path, {"model_state_dict": best_state, "config": cfg})
        log(f"Saved best checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
