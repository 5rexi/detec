from ppe.training import train_task

if __name__ == "__main__":
    train_task(
        task_name="vest",
        epochs=40,
        batch_size=32,
        lr=1e-4,
        weight_decay=5e-4,
        class_weights=[1.0, 4.0, 1.0],
        false_violation_penalty=3.0,
        dataset_root="dataset/vest",
        save_path="weights/resnet_vest.pth",
    )
