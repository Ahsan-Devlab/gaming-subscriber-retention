import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, accuracy_score

st.set_page_config(page_title='Subscriber Retention', page_icon='🎮')

st.title("🎮 Gaming Channel Subscriber Retention")

@st.cache_data
def load_and_train():
    df = pd.read_csv('gaming_data.csv')
    X = df[['watch_time_minutes', 'comment_frequency']]
    y = df['unsubscribed']

    model_standard = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    model_standard.fit(X, y)

    model_recall = xgb.XGBClassifier(scale_pos_weight=5, eval_metric='logloss')
    model_recall.fit(X, y)

    return model_standard, model_recall, X, y


model_standard, model_recall, X, y = load_and_train()

st.sidebar.header("Viewer Stats")
user_watch_time = st.sidebar.number_input("Monthly Watch Time (Minutes)", min_value=0, max_value=2000, value=200)
user_comments = st.sidebar.slider("Monthly Comments", min_value=0, max_value=50, value=30)

st.subheader("Select AI Strategy")
strategy = st.selectbox(
    "Choose the model optimization target:",
    ["Standard Model (Balanced Accuracy)", "Aggressive Model (High Recall - Catch All Risks)"]
)
if st.button("Predict Viewer Status"):
    features = [[user_watch_time, user_comments]]
    
    if "Standard" in strategy:
        prediction = model_standard.predict(features)[0]
    else:
        prediction = model_recall.predict(features)[0]
        
    if prediction == 1:
        st.error("🚨 **AT RISK:** This viewer is predicted to UNSUBSCRIBE. Send them a loyalty reward or shoutout!")
    else:
        st.success("✅ **SAFE:** This viewer is a loyal subscriber.")


st.write("---")
st.subheader("Model Performance (Cross-Validation)")
st.write("Notice how the Aggressive model sacrifices a bit of general accuracy to achieve a massive boost in Recall (catching the people leaving).")

cv_acc_std = cross_val_score(model_standard, X, y, cv=5, scoring='accuracy').mean()
cv_rec_std = cross_val_score(model_standard, X, y, cv=5, scoring='recall').mean()

cv_acc_rec = cross_val_score(model_recall, X, y, cv=5, scoring='accuracy').mean()
cv_rec_rec = cross_val_score(model_recall, X, y, cv=5, scoring='recall').mean()

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Standard Model**\n\nAccuracy: {cv_acc_std*100:.1f}%\n\nRecall: {cv_rec_std*100:.1f}%")
with col2:
    st.warning(f"**Aggressive (Recall) Model**\n\nAccuracy: {cv_acc_rec*100:.1f}%\n\nRecall: {cv_rec_rec*100:.1f}%")