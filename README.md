# 🎮 Gaming Channel Subscriber Retention

A machine learning web application built with XGBoost to predict if a gaming channel viewer is about to unsubscribe based on their engagement metrics.

🔗 **Live App Demo:** [Subscriber Retention App](https://ml-project08-gaming-subscriber-retention.streamlit.app/)

## 🧠 Concepts Covered
* **XGBoost:** Implementing the industry-standard gradient boosting framework for tabular data.
* **Recall vs. Accuracy:** Allowing the user to toggle between a standard balanced model and a model specifically weighted (`scale_pos_weight`) to maximize Recall (catching at-risk subscribers).
* **Cross-Validation:** Using K-Fold cross-validation to prove the statistical trade-off between Accuracy and Recall.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas
* **Deployment & UI:** Streamlit