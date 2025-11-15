Notes and command line workflows

Data is downloaded from Kaggle (NYC FHV (Uber/Lyft) Trip Data Expanded (2019-2022), https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data/data), which is a subset of TLC Trip Records from NYC Open Data.

Start the project by setting up dependency management with uv.
(cd into working directory)
uv init
uv venv
.venv\Scripts\activate
uv add pandas scikit-learn xgboost seaborn fastapi uvicorn

Explore data with a notebook.
(cd into working directory)
(.venv\Scripts\activate)
uv run --with jupyter jupyter lab


