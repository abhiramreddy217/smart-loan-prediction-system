from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================================
# CREATE FLASK APP
# =========================================

app = Flask(__name__)

# =========================================
# LOAD DATASET
# =========================================

data = pd.read_csv("dataset.csv")

# =========================================
# REMOVE MISSING VALUES
# =========================================

data = data.dropna()

# =========================================
# ENCODE TARGET COLUMN
# =========================================

label_encoder = LabelEncoder()

data['Loan_Status'] = label_encoder.fit_transform(
    data['Loan_Status']
)

# =========================================
# INPUT FEATURES
# =========================================

X = data[
    [
        'ApplicantIncome',
        'LoanAmount',
        'CreditScore',
        'EMI'
    ]
]

# =========================================
# OUTPUT TARGET
# =========================================

y = data['Loan_Status']

# =========================================
# SPLIT DATASET
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# CREATE MODEL
# =========================================

model = LogisticRegression(max_iter=1000)

# =========================================
# TRAIN MODEL
# =========================================

model.fit(X_train, y_train)

# =========================================
# HOME PAGE
# =========================================

@app.route('/')

def home():

    return render_template('index.html')

# =========================================
# PREDICTION PAGE
# =========================================

@app.route('/predict', methods=['POST'])

def predict():

    try:

        # =========================================
        # GET USER INPUTS
        # =========================================

        income = float(request.form['income'])

        loan = float(request.form['loan'])

        credit = float(request.form['credit'])

        emi = float(request.form['emi'])

        # =========================================
        # SMART VALIDATION RULES
        # =========================================

        if credit < 650:

            result = "❌ Loan Rejected - Low Credit Score"

        elif emi > (income * 0.5):

            result = "❌ Loan Rejected - EMI Too High"

        elif loan > (income * 5):

            result = "❌ Loan Rejected - Loan Amount Too High"

        else:

            # =========================================
            # CREATE SAMPLE DATAFRAME
            # =========================================

            sample_data = pd.DataFrame(
                [[income, loan, credit, emi]],
                columns=[
                    'ApplicantIncome',
                    'LoanAmount',
                    'CreditScore',
                    'EMI'
                ]
            )

            # =========================================
            # MACHINE LEARNING PREDICTION
            # =========================================

            prediction = model.predict(sample_data)

            # =========================================
            # FINAL RESULT
            # =========================================

            if prediction[0] == 1:

                result = "✅ Loan Approved"

            else:

                if income >= 5000 and credit >= 700:

                    result = "✅ Loan Approved"

                else:

                    result = "❌ Loan Rejected"

        # =========================================
        # RETURN RESULT
        # =========================================

        return render_template(
            'index.html',
            prediction=result
        )

    except ValueError:

        return render_template(
            'index.html',
            prediction="❌ Please enter valid numeric values"
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}"
        )

# =========================================
# RUN FLASK APP
# =========================================

if __name__ == "__main__":

    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True
    )