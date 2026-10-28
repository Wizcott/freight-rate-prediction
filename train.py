from pathlib import Path

import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parent

def features(df):
    x = df.drop(
        columns=[
            "posted_rate",
            "predicted_rate",
            "load_id"
        ],
        errors="ignore"
    ).copy()

    dt = pd.to_datetime(x.pop("date"))

    x["day_number"] = (
        dt - pd.Timestamp("2025-01-01")
    ).dt.days

    x["day_of_year"] = dt.dt.dayofyear

    x["weekday"] = dt.dt.weekday

    x["month"] = dt.dt.month

    x["day_of_month"] = dt.dt.day

    x["lane"] = (
        x["pickup"].astype(str)
        + " -> "
        + x["delivery"].astype(str)
    )

    x["distance_sq"] = x["distance"] ** 2

    x["weight_per_mile"] = (
        x["weight"] / (x["distance"] + 1)
    )

    if "quote_signal" in x:

        x["quote_times_distance"] = (
            x["quote_signal"] * x["distance"]
        )

    else:

        x["quote_times_distance"] = np.nan

    if "market_index" in x:

        x["market_times_distance"] = (
            x["market_index"] * x["distance"]
        )

    else:

        x["market_times_distance"] = np.nan

    numerical_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
        "weight"
    ]

    for column in numerical_columns:

        if column not in x:

            x[column] = np.nan

    categorical_columns = [
        "pickup",
        "delivery",
        "equipment",
        "lane"
    ]

    for column in categorical_columns:

        x[column] = (
            x[column]
            .fillna("Unknown")
            .astype(str)
        )

    return x

def model(iterations=550):

    return CatBoostRegressor(

        iterations=iterations,

        depth=8,

        learning_rate=0.055,

        loss_function="RMSE",

        l2_leaf_reg=5,

        verbose=False,

        thread_count=4,

        random_seed=42

    )

def main():

    train = pd.read_csv(
        ROOT / "data/train-test.csv"
    )

    val = pd.read_csv(
        ROOT / "data/validation.csv"
    )

    dec = pd.read_csv(
        ROOT / "data/december-chart-inputs.csv"
    )

    print("Training dataset:", train.shape)

    print("Validation dataset:", val.shape)

    print("December dataset:", dec.shape)

    assert train["load_id"].is_unique

    assert val["load_id"].is_unique

    x = features(train)

    y = train["posted_rate"]

    categorical_columns = [
        "pickup",
        "delivery",
        "equipment",
        "lane"
    ]

    cats = [
        x.columns.get_loc(column)
        for column in categorical_columns
    ]

    mask = (
        pd.to_datetime(train["date"])
        < pd.Timestamp("2025-10-01")
    )

    x_train = x.loc[mask]

    y_train = y.loc[mask]

    x_test = x.loc[~mask]

    y_test = y.loc[~mask]

    print("\nTraining rows:", len(x_train))

    print("Testing rows:", len(x_test))

    print("\nTraining CatBoost model...")

    m = model()

    m.fit(
        x_train,
        y_train,
        cat_features=cats
    )

    predictions = m.predict(x_test)

    truth = y_test

    mae = mean_absolute_error(
        truth,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            truth,
            predictions
        )
    )

    r2 = r2_score(
        truth,
        predictions
    )

    wape = (
        np.abs(truth - predictions).sum()
        / np.abs(truth).sum()
    )

    metrics = {

        "holdout_month": "2025-10",

        "train_rows": int(mask.sum()),

        "holdout_rows": int((~mask).sum()),

        "MAE": float(mae),

        "RMSE": float(rmse),

        "R2": float(r2),

        "WAPE": float(wape)

    }

    print("\nMODEL EVALUATION RESULTS")

    print(json.dumps(metrics, indent=2))

    with open(
        ROOT / "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    print("\nTraining final model...")

    final = model()

    final.fit(
        x,
        y,
        cat_features=cats
    )

    validation_features = features(val)

    validation_features = validation_features[
        x.columns
    ]

    vp = final.predict(
        validation_features
    )

    vp = np.maximum(vp, 0.01)

    template = pd.read_csv(
        ROOT / "data/validation-predictions-template.csv"
    )

    assert len(template) == len(val)

    assert (
        set(template["load_id"])
        == set(val["load_id"])
    )

    prediction_frame = pd.DataFrame({

        "load_id": val["load_id"],

        "predicted_rate": vp

    })

    out = template[["load_id"]].merge(

        prediction_frame,

        on="load_id",

        validate="one_to_one",

        how="left"

    )

    assert out["predicted_rate"].notna().all()

    out.to_csv(

        ROOT / "validation_predictions.csv",

        index=False,

        float_format="%.4f"

    )

    print("\nSaved validation_predictions.csv")

    december_features = features(dec)

    december_features = december_features[
        x.columns
    ]

    dp = final.predict(
        december_features
    )

    dp = np.maximum(dp, 0.01)

    dec["predicted_rate"] = np.round(
        dp,
        4
    )

    dec.to_csv(

        ROOT / "data/december_chart_inputs.csv",

        index=False

    )

    print("Saved December predictions")

    final.save_model(

        str(ROOT / "freight_model.cbm")

    )

    print("\nSaved freight_model.cbm")

    print("\nTraining completed successfully")

if __name__ == "__main__":

    main()