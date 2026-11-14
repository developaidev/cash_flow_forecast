import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# App Title & Description
st.title("📈 Financial Forecasting Dashboard")
st.write(
    "Upload your historical financial Excel file to predict future Revenue, Expenses, and Profit."
)

# 1. File Uploader for Excel
uploaded_file = st.file_uploader(
    "Upload Excel file (Must contain columns: Month, Revenue, Expense)",
    type=["xlsx"],
)

if uploaded_file is not None:
  # Read the Excel file
  df = pd.read_excel(uploaded_file)

  st.subheader("📊 Historical Data Preview")
  with st.expander("Preview Excel Dataset"):
      st.dataframe(
      df, #Display all rows
      hide_index = True, #To hide index column
      use_container_width = True #Use all available space to display
      )

  # Check if required columns exist
  required_cols = ["Month", "Revenue", "Expense"]
  if all(col in df.columns for col in required_cols): #all is a built in-function to ensure all cols are matched before proceeding to next step.
    # Train Models
    # Storing Month 1, 2, 3...etc into X variable
    X = df[["Month"]]
    
    # .fit(X,Y) being X as Input/Feaure, while Y as Target/Label, at its learning/training phase.
    model_revenue = LinearRegression().fit(X, df["Revenue"])
    model_expense = LinearRegression().fit(X, df["Expense"])

    # User Input: How many months to forecast?
    st.subheader("🔮 Forecast Settings")
    months_ahead = st.slider(
        "Select number of months to forecast into the future:", 1, 12, 3  #1 as minimum, 12 as maximum, while 3 as starting default
    )

    last_month = int(df["Month"].max())
    
    # np.arange means numpy array range, creating horizontal list eg: [13,14,15]
    # months_ahead is set equal to slicer's starting default
    future_months_list = np.arange(
        last_month + 1, last_month + 1 + months_ahead
    )
    
    # reshape convert horizontal list into vertical list (column) before scikit-learn (machine learning model) can handle as an input
    # -1 > automatically finding out how many rows are needed.
    # 1 > means only 1 column to be created
    future_X = future_months_list.reshape(-1, 1)

    # Make Predictions
    pred_rev = model_revenue.predict(future_X)
    pred_exp = model_expense.predict(future_X)
    pred_profit = pred_rev - pred_exp

    # Create a DataFrame for results
    forecast_df = pd.DataFrame({
        "Month": future_months_list,
        "Projected Revenue": pred_rev,
        "Projected Expense": pred_exp,
        "Projected Profit": pred_profit,
    }
        hide_index = True, #To hide index column
    )

    st.subheader("🚀 Forecast Results")
    st.dataframe(
        forecast_df.style.format({
            "Projected Revenue": "${:.2f}k",
            "Projected Expense": "${:.2f}k",
            "Projected Profit": "${:.2f}k",
        })
    )

    # Plotting interactive charts using Streamlit
    st.subheader("📈 Financial Trends Chart")
    chart_data = pd.DataFrame({
        "Revenue": np.concatenate([df["Revenue"], pred_rev]),
        "Expense": np.concatenate([df["Expense"], pred_exp]),
    }, index=np.concatenate([df["Month"], future_months_list]))

    st.line_chart(chart_data)

  else:
    st.error(
        "Your Excel file is missing one or more required columns: Month,"
        " Revenue, Expense"
    )
else:
  st.info(
      "👈 Please upload an Excel file to get started. Sample format: columns"
      " named `Month`, `Revenue`, and `Expense`."
  )
