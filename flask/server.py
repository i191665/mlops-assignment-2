import pandas as pd
from tensorflow import keras
import os

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms.fields import StringField
from wtforms.validators import DataRequired

SECRET_KEY = os.urandom(32)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

model_path = os.path.join(os.path.dirname(__file__), "model/stock_predictor.h5")
model = keras.models.load_model(model_path)

api_key = "JTU5ZLRZQK4D1X0P"
ticker = "AAPL"
interval = "DAILY"


@app.route("/")
def home():
    df = pd.read_csv(
        f"https://www.alphavantage.co/query?function=TIME_SERIES_{interval}_ADJUSTED&symbol={ticker}&datatype=csv&apikey={api_key}"
    )
    prices = df["close"].to_list()[:7]
    prices = prices[::-1]

    dates = df["timestamp"].to_list()[:7]
    dates = dates[::-1]

    prediction = model.predict([prices]).item()

    return render_template(
        "home.html", prices=prices, dates=dates, prediction=prediction
    )


class PricesForm(FlaskForm):
    prices = StringField("Prices", validators=[DataRequired()])


@app.route("/service", methods=["GET", "POST"])
def service():
    form = PricesForm()

    if form.validate_on_submit():
        prices = form.prices.data

        try:
            prices = [float(price) for price in prices.split(",")]
        except:
            error = "Could not parse input. Make sure inputs contain only numbers separated by commas."
            return render_template("service.html", form=form, error=error)

        if len(prices) < 8:
            error = "There must be atleast 8 price points for the model to predict"
            return render_template("service.html", form=form, error=error)

        if len(prices) > 8:
            error = "There must be a maximum of 8 price points"
            return render_template("service.html", form=form, error=error)

        x = prices[:-1]
        y = prices[-1]

        prediction = model.predict([x]).item()
        mae = abs(y - prediction)

        return render_template(
            "service.html", form=form, target=y, prediction=prediction, mae=mae
        )

    return render_template("service.html", form=form)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
