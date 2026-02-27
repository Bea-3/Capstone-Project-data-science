import os
from flask import Flask, request, jsonify
import pandas as pd

from energy_pipeline import run_energy_pipeline
from forecast import predict_30_days

app = Flask(__name__)


# -----------------------------------------
# HEALTH CHECK
# -----------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -----------------------------------------
# RUN DETERMINISTIC ENERGY PIPELINE
# -----------------------------------------
@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    try:
        data = request.get_json()

        appliances = pd.DataFrame(data.get("appliances", []))
        consumption_log = pd.DataFrame(data.get("consumption_log", []))
        energy_balance = pd.DataFrame(data.get("energy_balance", []))
        energy_purchase = pd.DataFrame(data.get("energy_purchase", []))
        energy_accounts = pd.DataFrame(data.get("energy_accounts", []))

        updated_log, updated_balance = run_energy_pipeline(
            appliances,
            consumption_log,
            energy_balance,
            energy_purchase,
            energy_accounts
        )

        return jsonify({
            "consumption_log": updated_log.to_dict(orient="records"),
            "energy_balance": updated_balance.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------
# ML FORECAST ENDPOINT
# -----------------------------------------
@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        input_df = pd.DataFrame(data.get("forecast_input", []))

        result = predict_30_days(input_df)

        return jsonify({
            "forecast": result.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------
# START SERVER (Render uses gunicorn)
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)