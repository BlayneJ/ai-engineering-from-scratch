import os
import time

from datasets import load_dataset


def inspect_glue_mrpc() -> None:
    print("\n=== GLUE MRPC: first 5 examples ===")
    dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

    for i in range(5):
        row = dataset[i]
        print(f"\nExample {i + 1}")
        print("sentence1:", row["sentence1"])
        print("sentence2:", row["sentence2"])
        print("label:", row["label"])


def compare_csv_parquet() -> None:
    print("\n=== CSV vs Parquet size ===")
    dataset = load_dataset("nyu-mll/glue", "mrpc", split="train[:500]")

    os.makedirs("data", exist_ok=True)

    csv_path = "data/mrpc_500.csv"
    parquet_path = "data/mrpc_500.parquet"

    dataset.to_csv(csv_path)
    dataset.to_parquet(parquet_path)

    csv_size = os.path.getsize(csv_path)
    parquet_size = os.path.getsize(parquet_path)

    print(f"CSV:     {csv_size:,} bytes")
    print(f"Parquet: {parquet_size:,} bytes")
    print(f"Parquet is {csv_size / parquet_size:.2f}x smaller")


def create_splits() -> None:
    print("\n=== 70/15/15 split ===")
    dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

    split = dataset.train_test_split(test_size=0.30, seed=42)
    val_test = split["test"].train_test_split(test_size=0.50, seed=42)

    train_ds = split["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    total = len(dataset)
    print(f"Total: {total}")
    print(f"Train: {len(train_ds)} ({len(train_ds) / total:.1%})")
    print(f"Val:   {len(val_ds)} ({len(val_ds) / total:.1%})")
    print(f"Test:  {len(test_ds)} ({len(test_ds) / total:.1%})")


def stream_c4_for_10_seconds() -> None:
    print("\n=== Streaming C4 for 10 seconds ===")
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )

    start = time.time()
    count = 0

    for _ in dataset:
        count += 1
        if time.time() - start >= 10:
            break

    print(f"Processed {count} examples in 10 seconds")


if __name__ == "__main__":
    inspect_glue_mrpc()
    compare_csv_parquet()
    create_splits()
    stream_c4_for_10_seconds()
