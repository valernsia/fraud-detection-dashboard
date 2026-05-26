import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Fintech Fraud Detection Dashboard")

st.markdown("""
This dashboard simulates credit card fraud detection using a machine learning model trained on anonymized transaction data.

Users can adjust transaction characteristics using the slider on the side to simulate different transaction behaviours and observe how the fraud risk score changes in real time.

""")
st.divider()



col1, col2, col3, col4 = st.columns([1,1,1,1.5])



df = pd.read_csv("data/creditcard.csv").sample(5000, random_state = 42)

st.markdown("""
### Transaction Dataset Preview

The table below shows a sample of anonymized credit card transaction data used to train the fraud detection model.

- `Class = 0` represents normal transactions
- `Class = 1` represents fraudulent transactions
""")

st.write(df.head())

st.markdown("""
### Class Imbalance in Fraud Detection

Fraudulent transactions are much rarer than normal transactions. A Weighted Logistic Regression model was used to give fraud cases greater importance during training.
""")

fig, ax = plt.subplots()

df["Class"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_title("Fraud vs Normal Transactions")
ax.set_xlabel("Class")
ax.set_ylabel("Count")

st.pyplot(fig)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


@st.cache_resource
def train_model():

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(
    class_weight="balanced",
    max_iter=10000
    )

    model.fit(X_train, y_train)

    return model, X


model, X = train_model()

st.sidebar.header("Transaction Input")

amount = st.sidebar.slider(
    "Transaction Amount",
    min_value=0.0,
    max_value=5000.0,
    value=100.0
)

v25 = st.sidebar.slider(
    "V25",
    min_value=-20.0,
    max_value=20.0,
    value=0.0
)

v14 = st.sidebar.slider(
    "V14",
    min_value=-20.0,
    max_value=20.0,
    value=0.0
)

v3 = st.sidebar.slider(
    "V3",
    min_value=-20.0,
    max_value=20.0,
    value=0.0
)

v10 = st.sidebar.slider(
    "V10",
    min_value=-20.0,
    max_value=20.0,
    value=0.0
)

sample_transaction = X.mean().copy()
sample_transaction["Amount"] = amount
sample_transaction["V25"] = v25
sample_transaction["V14"] = v14
sample_transaction["V3"] = v3
sample_transaction["V10"] = v10

coefficients = model.coef_[0]

feature_contributions = pd.DataFrame({
    "Feature": X.columns,
    "Value": sample_transaction.values,
    "Coefficient": coefficients
})

feature_contributions["Contribution"] = (
    feature_contributions["Value"]
    * feature_contributions["Coefficient"]
) #contribution of a variable = value x coeff

top_features = feature_contributions.sort_values(
    by="Contribution",
    ascending=False
).head(5)

st.markdown("""
### Top Fraud Signals

These features contributed most strongly to the model's fraud prediction for the current transaction.
""")

st.dataframe(
    top_features[
        ["Feature", "Value", "Contribution"]
    ]
)

sample_df = pd.DataFrame([sample_transaction])

prediction = model.predict(sample_df)[0]

prediction_probability = model.predict_proba(sample_df)[0][1]

col1.metric(
    "Fraud Probability",
    f"{prediction_probability:.2%}"
)

col2.metric(
    "Transaction Amount",
    f"${amount:.2f}"
)

risk_level = (
    "High"
    if prediction_probability > 0.7
    else "Medium"
    if prediction_probability > 0.3
    else "Low"
)

col3.metric(
    "Risk Level",
    risk_level
)

col4.metric(
    "Model",
    "Weighted LR"
)

st.divider()

st.markdown("""
### Fraud Risk Score

This score represents the model's estimated probability that the transaction is fraudulent based on the selected transaction characteristics.
""")

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction_probability * 100,

    title = {'text': "Fraud Risk Score"},

    gauge = {
        'axis': {'range': [0, 100]},

        'bar': {'color': "darkred"},

        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "gold"},
            {'range': [70, 100], 'color': "salmon"}
        ]
    }
))

st.plotly_chart(fig)


if prediction_probability > 0.7:
    st.error(f"High Fraud Risk: {prediction_probability:.2%}")

elif prediction_probability > 0.3:
    st.warning(f"Medium Fraud Risk: {prediction_probability:.2%}")

else:
    st.success(f"Low Fraud Risk: {prediction_probability:.2%}")

st.subheader("Fraud Prediction Result")


if prediction == 1:
    st.warning(
        f"This transaction is classified as potentially fraudulent "
       
    )

else:
    st.success(
        f"This transaction is not currently classified as fraudulent. "
 
    )

st.caption(
    "The model classifies transactions with predicted fraud probability above 50% as potentially fraudulent."
)