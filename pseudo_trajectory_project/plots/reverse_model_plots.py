import matplotlib.pyplot as plt

# Training loss values
epochs = list(range(1, 21))
losses = [
    0.3198, 0.2855, 0.2556, 0.2275, 0.1916,
    0.1543, 0.1185, 0.0880, 0.0636, 0.0456,
    0.0332, 0.0240, 0.0182, 0.0167, 0.0145,
    0.0134, 0.0140, 0.0128, 0.0130, 0.0122
]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Reverse Model Training Loss")
plt.grid(True)

plt.tight_layout()

# Save files
plt.savefig("reverse_model_training_loss.pdf", bbox_inches="tight")
plt.savefig("reverse_model_training_loss.png", dpi=300, bbox_inches="tight")

plt.show()
