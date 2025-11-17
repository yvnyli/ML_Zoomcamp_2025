# Predict ride share price

Data was downloaded from Kaggle (NYC FHV (Uber/Lyft) Trip Data Expanded (2019-2022), https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data/data), which is a subset of TLC Trip Records from NYC Open Data.
**The entire dataset is too large. For this project, only one month of data (one file) is used and it is uploaded to this repo.**

This table contains 20,159,102 rows and 24 columns about ride share rides that happened in Febrary 2019.

**Let's predict the cost of a ride: base_passenger_fare, which is the base passenger fare before tolls, tips, taxes, and fees.**



## Notes and command line workflows
Start the project by setting up dependency management with uv.
```
(cd into working directory)
uv init
uv venv
.venv\Scripts\activate
uv add pandas scikit-learn xgboost seaborn fastapi uvicorn
uv add --dev psutil pympler tqdm
```

Explore data with a notebook.
```
(cd into working directory)
(.venv\Scripts\activate)
uv run --with jupyter jupyter lab
```



