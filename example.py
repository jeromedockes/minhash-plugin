import polars as pl

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import datasets

import minhash

employees = datasets.fetch_employee_salaries()
df = pl.from_pandas(employees.X)
cols = [
    "department",
    "department_name",
    "division",
    "assignment_category",
    "employee_position_title",
]
X = df.select(pl.col(cols).mh.minhash(30))

score = cross_val_score(HistGradientBoostingRegressor(), X, employees.y)
print(f"score:\n{score}")
