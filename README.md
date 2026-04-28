# F1 Race Prediction Project

This project uses historical Formula 1 data to predict how many positions a driver will gain or lose in a race.
The main implementation is in `f1.ipynb`.

## Installation

Yes — the user should install the Python libraries used by the notebook before running it.

### Recommended setup

It is best to use a virtual environment so the project dependencies stay isolated.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer installing manually, these are the main packages required by the notebook:

- `pandas`
- `numpy`
- `xgboost`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Optional, but useful for running notebooks:

- `jupyter`

> Note: the notebook reads CSV files from the local `data/` folder, so that folder must stay in the project directory.

## What the notebook does

The notebook builds a race-level machine learning pipeline:

1. **Loads historical F1 CSV files** from `data/`.
2. **Merges them into one training table** with race, driver, constructor, standings, pit stop, and circuit information.
3. **Creates engineered features** such as driver age, experience, recent form, pit stop efficiency, and reliability.
4. **Defines the target** as the number of positions gained or lost in a race.
5. **Trains three regression models** on races that happened before a chosen target race.
6. **Predicts the finishing order** for specific races such as Monaco 2024, Dutch GP 2023, and São Paulo GP 2023.
7. **Plots predictions vs. actual results**.
8. **Compares model accuracy** using Mean Absolute Error (MAE).
9. **Compares feature importances** across the three models.
10. **Summarizes the average MAE** across all evaluated races.

## Goal of the prediction

The notebook does **not** predict lap-by-lap behavior.
It predicts the race movement of each driver using this target:

`positions_changed = grid - positionOrder`

- If the value is **positive**, the driver gained positions.
- If the value is **negative**, the driver lost positions.
- If the value is **0**, the driver finished where they started.

The models learn this value from historical data, and the predicted delta is then used to estimate the finishing rank.

## Models used

The notebook trains and compares three regression models:

### 1. `RandomForestRegressor`
- An ensemble of many decision trees.
- Each tree is trained on a bootstrap sample of the data, and the final prediction is the average of all trees.
- It is robust and often a good baseline for tabular data.
- In this project it predicts the expected position change using the engineered race features.


### 2. `GradientBoostingRegressor`
- A boosting model from scikit-learn.
- Like XGBoost, it builds trees one after another, each one focusing on the errors of the previous ensemble.
- It is used here as another tabular regression approach to compare with the other models.


### 3. `XGBRegressor`
- A gradient-boosted tree model from **XGBoost**.
- It builds trees sequentially, where each new tree tries to correct the mistakes of the previous ones.
- In this project it is used to learn the relationship between race features and position change.
- Usually strong when there are complex non-linear patterns in the data.

### Why three models?

The notebook compares different tree-based regression strategies to see which one best estimates race position changes.
The final output includes:

- race-by-race MAE comparison,
- feature importance comparison,
- overall model ranking by average MAE.




## Columns used from each CSV file

Below is the list of columns actually used by the notebook.

### `results.csv`
Used columns:
- `raceId`
- `driverId`
- `constructorId`
- `grid`
- `positionOrder`
- `statusId`



### `races.csv`
Used columns:
- `raceId`
- `year`
- `round`
- `circuitId`
- `name`
- `date`


### `drivers.csv`
Used columns:
- `driverId`
- `driverRef`
- `dob`
- `nationality`

### `constructors.csv`
Used columns:
- `constructorId`
- `constructorRef`
- `nationality`

### `driver_standings.csv`
Used columns:
- `raceId`
- `driverId`
- `points`
- `wins`

### `pit_stops.csv`
Used columns:
- `raceId`
- `driverId`
- `duration`

### `circuits.csv`
Used columns:
- `circuitId`
- `country`

## How the dataset is created

The dataset is created by starting from `results.csv` and then enriching it with metadata from the other CSV files.

### Step 1: Base race results
The initial table comes from `results.csv`, which contains one row per driver per race.

### Step 2: Add race metadata
`results.csv` is merged with selected columns from `races.csv` so that every result row also knows:

- race year,
- round number,
- circuit ID,
- race name,
- race date.

### Step 3: Add driver metadata
The merged table is then joined with selected columns from `drivers.csv`:

- driver reference name,
- date of birth,
- nationality.

### Step 4: Add constructor metadata
Then the table is joined with selected columns from `constructors.csv`:

- constructor reference name,
- constructor nationality.

### Step 5: Add circuit country
The dataset is merged with selected columns from `circuits.csv`:

- circuit country.

### Step 6: Add standings information
`driver_standings.csv` is aligned with `races.csv` so standings can be shifted to the race **before** the current one.
This gives pre-race features like:

- points before the race,
- wins before the race.

### Step 7: Add recent form features
Using the already merged historical table, the notebook computes rolling averages for:

- driver recent form,
- team recent form,
- driver track history.

### Step 8: Add pit stop efficiency
`pit_stops.csv` is combined with `results.csv` to connect pit stop duration to each driver and constructor.
The notebook then computes team pit-stop efficiency using rolling averages.

### Step 9: Add reliability
The notebook creates a mechanical DNF indicator from `statusId` in `results.csv`, then computes a team reliability feature over the last 10 races.

## Features used for training

These are the final model features used in the notebook:

- `grid`
- `driver_age`
- `driver_experience`
- `points_before_race`
- `wins_before_race`
- `driver_recent_form`
- `team_recent_form`
- `driver_track_history`
- `constructor_pit_efficiency`
- `team_reliability`
- `is_home_race_driver`
- `is_home_race_team`



## Missing values

The notebook reads the CSV files with:

```python
na_values=['\\N']
```

That means the dataset converts the Formula 1 database missing-value marker `\N` into proper `NaN` values.

## Races evaluated in the notebook

The final notebook repeats the same pipeline for multiple race examples:

- **Monaco 2024**
- **Dutch GP 2023**
- **São Paulo GP 2023**

For each target race:

1. the code trains on all races that happened earlier,
2. it predicts position change for the selected race,
3. it plots the result,
4. it computes MAE,
5. it stores the result for the final comparison table.

## Outputs produced

The notebook produces:

- race prediction scatter plots,
- MAE comparison for each model,
- feature-importance bar plots,
- a final average MAE leaderboard.




