"""Legacy entrypoint. Use train_helmet.py / train_vest.py for explicit task separation."""

import argparse

from ppe.training import train_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["helmet", "vest"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--class-weights", nargs="*", type=float, default=[1.0, 4.0, 1.0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    train_task(
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=args.class_weights,
        dataset_root=args.dataset_root,
        save_path=args.save_path,
    )
