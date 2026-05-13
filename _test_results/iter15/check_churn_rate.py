import pandas as pd

f = pd.read_csv("results/features.csv", usecols=["customer_id", "churn_label", "persona"])
rate = f["churn_label"].mean()
print(f"Total customers: {len(f):,}")
print(f"ACTUAL churn rate (label): {rate:.4f} = {rate*100:.2f}%")
print(f"Required range: 15.00% - 25.00%")
print(f"Within range? {0.15 <= rate <= 0.25}")
print()
print("By persona:")
print(f.groupby("persona")["churn_label"].agg(["mean", "count"]))

print()
p = pd.read_csv("results/churn_predictions.csv")
print(f"PREDICTED churn_probability — mean: {p['churn_probability'].mean():.4f} = {p['churn_probability'].mean()*100:.2f}%")
print(f"PREDICTED churn_probability — median: {p['churn_probability'].median():.4f} = {p['churn_probability'].median()*100:.2f}%")
print(f"PREDICTED churn_probability — at threshold 0.5: {(p['churn_probability'] >= 0.5).mean():.4f} = {(p['churn_probability'] >= 0.5).mean()*100:.2f}%")
