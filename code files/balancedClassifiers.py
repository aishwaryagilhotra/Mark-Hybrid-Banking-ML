clf_models = {
    "Random Forest": RandomForestClassifier(
        class_weight='balanced', random_state=42
    ),
    "LightGBM": LGBMClassifier(
        class_weight='balanced', random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
}
