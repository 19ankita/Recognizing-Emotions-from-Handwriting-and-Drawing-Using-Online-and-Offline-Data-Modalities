import matplotlib.pyplot as plt

models = ["Linear Regression"]
rmse_values = [17.04]

plt.bar(models, rmse_values)
plt.ylabel("RMSE")
plt.title("RMSE for TOTAL DASS Prediction (Words Task)")

plt.savefig("rmse_words_task.png", dpi=300, bbox_inches="tight")
plt.show()