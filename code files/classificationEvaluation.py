print("\n===== CLASSIFICATION RESULTS =====")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in clf_models.items():

    cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print(f"\n{name} CV ROC-AUC:", cv_auc.mean())

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds_temp = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    final_preds = (probs >= best_threshold).astype(int)

    print("Best Threshold:", best_threshold)
    print("Accuracy:", accuracy_score(y_test, final_preds))
    print("Precision:", precision_score(y_test, final_preds))
    print("Recall:", recall_score(y_test, final_preds))
    print("F1:", f1_score(y_test, final_preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("PR-AUC:", average_precision_score(y_test, probs))

    # Confusion Matrix
    cm = confusion_matrix(y_test, final_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{name} ROC Curve")
    plt.show()

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.title(f"{name} Precision-Recall Curve")
    plt.show()
