import pickle
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open("reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("col.pkl", "rb") as f:
    model_columns = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    for col in ['no_of_trainings', 'age','length_of_service', 'avg_training_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_dummies=pd.get_dummies(df)
    df_dummies=df_dummies.reindex(columns=model_columns, fill_value=0)

    pred = reg_model.predict(df_dummies)[0]
    proba = reg_model.predict_proba(df_dummies)[0]

    result = "Promoted" if pred==1 else "Not Promoted"
    return render_template("index.html", prediction=result, prob_no=proba[0], prob_yes=proba[1])

if __name__ == "__main__":
    app.run(debug=True)