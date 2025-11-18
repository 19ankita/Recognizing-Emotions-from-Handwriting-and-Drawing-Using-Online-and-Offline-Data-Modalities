import os
import json
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG — where your experiments are stored
# ----------------------------------------------------------------------
NO_AUG_ROOT = "results_no_aug"
AUG_ROOT = "results_aug"

TASKS = ["pentagon", "house", "cdt", "cursive_writing", "words"]


def load_test_accuracy(result_root, task):
    """
    Read final test accuracy from trace.jsonl inside last timestamp folder.
    """
    task_dir = os.path.join(result_root, task)

    if not os.path.isdir(task_dir):
        return None

    # Pick the latest timestamp directory
    timestamps = sorted(os.listdir(task_dir))
    if len(timestamps) == 0:
        return None

    last_run = os.path.join(task_dir, timestamps[-1])
    trace_file = os.path.join(last_run, "trace.jsonl")

    if not os.path.isfile(trace_file):
        return None

    # Load JSON lines
    with open(trace_file, "r") as f:
        lines = f.readlines()

    # Find last epoch entry
    last = json.loads(lines[-1])

    # Must have 'val_acc' or 'test_acc'
    if "test_acc" in last:
        return last["test_acc"]

    if "val_acc" in last:
        return last["val_acc"]

    return None


def main():

    rows = []

    for task in TASKS:
        no_aug_acc = load_test_accuracy(NO_AUG_ROOT, task)
        aug_acc = load_test_accuracy(AUG_ROOT, task)

        rows.append({
            "Task": task,
            "No Aug Acc": no_aug_acc,
            "With Aug Acc": aug_acc,
            "Improvement": (aug_acc - no_aug_acc) if (no_aug_acc and aug_acc) else None
        })

    df = pd.DataFrame(rows)
    print("\n===== Comparison Table =====\n")
    print(df.to_string(index=False))

    df.to_csv("augmentation_comparison.csv", index=False)
    print("\nSaved CSV: augmentation_comparison.csv\n")


if __name__ == "__main__":
    main()
