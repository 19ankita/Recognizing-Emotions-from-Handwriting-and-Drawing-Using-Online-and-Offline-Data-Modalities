import os
import csv
import numpy as np

TASKS = ["pentagon", "house", "tree", "spiral", "words"]
RESULTS_DIR = "results"

summary = []

for task in TASKS:
    csv_path = os.path.join(RESULTS_DIR, f"protonet_{task}", "metrics.csv")

    if not os.path.isfile(csv_path):
        print(f"Missing metrics for {task}")
        continue

    losses = []
    accs = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            epoch, loss, acc, split = row
            if split == "val":
                accs.append(float(acc))

    avg_acc = np.mean(accs)
    std_acc = np.std(accs)

    summary.append((task, avg_acc*100, std_acc*100))

print("\n=== SUMMARY TABLE ===")
print("Task       | Accuracy (%) | STD (%)")
print("-------------------------------------")
for t, acc, std in summary:
    print(f"{t:10s} | {acc:12.2f} | {std:7.2f}")

# Save to CSV
with open("summary_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Task", "Accuracy (%)", "STD (%)"])
    for row in summary:
        writer.writerow(row)
