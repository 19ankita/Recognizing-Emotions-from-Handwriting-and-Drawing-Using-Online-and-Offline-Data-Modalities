import os
import csv
import matplotlib.pyplot as plt

TASKS = ["pentagon", "house", "tree", "spiral", "words"]
RESULTS_DIR = "results"

plt.figure(figsize=(10, 6))

for task in TASKS:
    csv_path = os.path.join(RESULTS_DIR, f"protonet_{task}", "metrics.csv")
    if not os.path.isfile(csv_path):
        print(f"Skipping {task}: no metrics found.")
        continue

    epochs = []
    train_acc = []
    val_acc = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            epoch, loss, acc, split = row
            epoch = int(epoch)

            if split == "train":
                if epoch not in epochs:
                    epochs.append(epoch)
                train_acc.append(float(acc))

            if split == "val":
                val_acc.append(float(acc))

    plt.plot(epochs, val_acc, label=f"{task} (val acc)")
    
plt.title("Validation Accuracy Across EMOTHAW Tasks")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("all_tasks_accuracy_curves.png")
plt.show()
