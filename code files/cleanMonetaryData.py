monetary_df.columns = [
    "Effective Date", "Bank Rate", "Repo", "Reverse Repo",
    "SDF Rate", "MSF Rate", "CRR", "SLR"
]

monetary_df["Effective Date"] = pd.to_datetime(
    monetary_df["Effective Date"], errors="coerce"
)

for col in ["Repo", "Reverse Repo", "CRR", "SLR"]:
    monetary_df[col] = pd.to_numeric(
        monetary_df[col].replace("-", np.nan),
        errors="coerce"
    )

monetary_df = monetary_df.sort_values("Effective Date").ffill()
latest_macro = monetary_df.iloc[-1]

repo = latest_macro["Repo"]
crr = latest_macro["CRR"]
slr = latest_macro["SLR"]
