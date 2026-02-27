import pandas as pd
import numpy as np
from datetime import timedelta


# -------------------------------------------------
# CLEAN ACCOUNT IDS
# -------------------------------------------------
def _clean_account_ids(appliances, consumption_log, energy_balance, energy_purchase, energy_accounts):

    if "energy_account_id" in appliances.columns:
        appliances = appliances.rename(columns={"energy_account_id": "account_id"})

    for df in [appliances, consumption_log, energy_balance, energy_purchase, energy_accounts]:
        if "account_id" in df.columns:
            df["account_id"] = df["account_id"].astype(str).str.strip()

    return appliances, consumption_log, energy_balance, energy_purchase, energy_accounts


# -------------------------------------------------
# CALCULATE DAILY KWH FROM APPLIANCES
# -------------------------------------------------
def _prepare_daily_kwh(appliances: pd.DataFrame,
                       energy_accounts: pd.DataFrame) -> pd.DataFrame:

    appliances = appliances.copy()

    appliances["kwh_per_day"] = (
        appliances["wattage"] *
        appliances["quantity"] *
        appliances["hours_per_day"] *
        appliances["duty_cycle"]
    ) / 1000

    daily_kwh = (
        appliances.groupby("account_id")["kwh_per_day"]
        .sum()
        .reset_index()
    )

    daily_kwh.columns = ["account_id", "daily_kwh"]

    daily_kwh = daily_kwh.merge(
        energy_accounts[["account_id", "base_tariff"]],
        on="account_id",
        how="left"
    )

    daily_kwh["units_per_day"] = daily_kwh["daily_kwh"] / daily_kwh["base_tariff"]

    return daily_kwh


# -------------------------------------------------
# PREPARE STARTING BALANCE
# -------------------------------------------------
def _prepare_starting_balance(energy_balance: pd.DataFrame,
                              energy_purchase: pd.DataFrame) -> pd.DataFrame:

    energy_balance = energy_balance.copy()

    purchase_totals = (
        energy_purchase.groupby("account_id")["units_purchased"]
        .sum()
        .reset_index()
    )

    purchase_totals.columns = ["account_id", "lifetime_units_purchased"]

    energy_balance = energy_balance.merge(
        purchase_totals,
        on="account_id",
        how="left"
    )

    energy_balance["lifetime_units_purchased"] = energy_balance["lifetime_units_purchased"].fillna(0)
    energy_balance["lifetime_units_consumed"] = 0
    energy_balance["current_units"] = energy_balance["lifetime_units_purchased"]

    return energy_balance


# -------------------------------------------------
# GENERATE DAILY CONSUMPTION LOG
# -------------------------------------------------
def _generate_consumption_log(consumption_log: pd.DataFrame,
                              daily_kwh: pd.DataFrame,
                              energy_balance: pd.DataFrame) -> pd.DataFrame:

    consumption_log = consumption_log.copy()

    consumption_log = consumption_log.merge(
        daily_kwh,
        on="account_id",
        how="left"
    )

    variation = np.random.uniform(0.9, 1.1, size=len(consumption_log))

    consumption_log["estimated_kwh_used"] = consumption_log["daily_kwh"] * variation
    consumption_log["estimated_units_used"] = consumption_log["units_per_day"] * variation

    consumption_log = consumption_log.sort_values(["account_id", "date"])

    updated_logs = []

    for acc in consumption_log["account_id"].unique():

        acc_log = consumption_log[consumption_log["account_id"] == acc].copy()

        start_balance = float(
            energy_balance.loc[
                energy_balance["account_id"] == acc,
                "current_units"
            ].iloc[0]
        )

        acc_log["remaining_units"] = start_balance - acc_log["estimated_units_used"].cumsum()

        updated_logs.append(acc_log)

    consumption_log = pd.concat(updated_logs)

    return consumption_log


# -------------------------------------------------
# UPDATE ENERGY BALANCE AFTER CONSUMPTION
# -------------------------------------------------
def _update_energy_balance_with_consumption(energy_balance: pd.DataFrame,
                                            consumption_log: pd.DataFrame) -> pd.DataFrame:

    energy_balance = energy_balance.copy()

    consumed_totals = (
        consumption_log.groupby("account_id")["estimated_units_used"]
        .sum()
        .reset_index()
    )

    consumed_totals.columns = ["account_id", "lifetime_units_consumed"]

    energy_balance = energy_balance.merge(
        consumed_totals,
        on="account_id",
        how="left"
    )

    energy_balance["lifetime_units_consumed"] = energy_balance["lifetime_units_consumed"].fillna(0)

    energy_balance["current_units"] = (
        energy_balance["lifetime_units_purchased"] -
        energy_balance["lifetime_units_consumed"]
    )

    last_dates = (
        consumption_log.groupby("account_id")["date"]
        .max()
        .reset_index()
    )

    last_dates.columns = ["account_id", "last_updated"]

    energy_balance = energy_balance.merge(
        last_dates,
        on="account_id",
        how="left"
    )

    return energy_balance


# -------------------------------------------------
# FORECAST DEPLETION DATE
# -------------------------------------------------
def _forecast_depletion(energy_balance: pd.DataFrame,
                        daily_kwh: pd.DataFrame) -> pd.DataFrame:

    energy_balance = energy_balance.copy()

    energy_balance = energy_balance.merge(
        daily_kwh[["account_id", "units_per_day"]],
        on="account_id",
        how="left"
    )

    energy_balance["days_remaining"] = (
        energy_balance["current_units"] /
        energy_balance["units_per_day"]
    )

    energy_balance["days_remaining"] = energy_balance["days_remaining"].replace(
        [np.inf, -np.inf], np.nan
    )

    energy_balance.loc[
        energy_balance["current_units"] <= 0,
        "days_remaining"
    ] = 0

    energy_balance["forecast_depletion_date"] = energy_balance.apply(
        lambda row: (
            row["last_updated"] + timedelta(days=row["days_remaining"])
            if pd.notna(row["last_updated"]) and pd.notna(row["days_remaining"])
            else pd.NaT
        ),
        axis=1
    )

    return energy_balance


# -------------------------------------------------
# MAIN PIPELINE WRAPPER
# -------------------------------------------------
def run_energy_pipeline(
    appliances: pd.DataFrame,
    consumption_log: pd.DataFrame,
    energy_balance: pd.DataFrame,
    energy_purchase: pd.DataFrame,
    energy_accounts: pd.DataFrame,
):

    appliances, consumption_log, energy_balance, energy_purchase, energy_accounts = _clean_account_ids(
        appliances, consumption_log, energy_balance, energy_purchase, energy_accounts
    )

    daily_kwh = _prepare_daily_kwh(appliances, energy_accounts)

    energy_balance = _prepare_starting_balance(energy_balance, energy_purchase)

    consumption_log = _generate_consumption_log(consumption_log, daily_kwh, energy_balance)

    energy_balance = _update_energy_balance_with_consumption(energy_balance, consumption_log)

    energy_balance = _forecast_depletion(energy_balance, daily_kwh)

    return consumption_log, energy_balance