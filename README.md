toy project to try out polars expression plugins.

min-hash all byte n-grams with n in [1, 8].

```
>>> import polars as pl
>>> df = pl.DataFrame(dict(A="one two".split(), B="two three".split()))
>>> df
shape: (2, 2)
┌─────┬───────┐
│ A   ┆ B     │
│ --- ┆ ---   │
│ str ┆ str   │
╞═════╪═══════╡
│ one ┆ two   │
│ two ┆ three │
└─────┴───────┘
>>> import minhash
>>> df.select(pl.all().mh.minhash(n_seeds=3))
shape: (2, 6)
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ A_mh_0     ┆ B_mh_0     ┆ A_mh_1     ┆ B_mh_1     ┆ A_mh_2     ┆ B_mh_2     │
│ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
│ u32        ┆ u32        ┆ u32        ┆ u32        ┆ u32        ┆ u32        │
╞════════════╪════════════╪════════════╪════════════╪════════════╪════════════╡
│ 4005290117 ┆ 1109059461 ┆ 2343285992 ┆ 603952496  ┆ 3434996698 ┆ 356999327  │
│ 1109059461 ┆ 4005290117 ┆ 603952496  ┆ 3971276239 ┆ 356999327  ┆ 3660921960 │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

The timing comparison below doesn't really make sense because

- the MinHashEncoder performs some checks etc that the `select` call doesn't
- the hashing functions are different (`minhash` is based on fasthash, the
  default MinHashEncoder just does one multiplication,
  MinHashEncoder(hashing='murmur') is based on murmurhash)

Still it shows that even a relatively unoptimized implementation as a polars
expression can be interesting in terms of computation time.

```
>>> import polars as pl
>>> from skrub import datasets, MinHashEncoder
>>> import minhash
>>> employees = datasets.fetch_employee_salaries()
>>> cols = [
...     "department",
...     "department_name",
...     "division",
...     "assignment_category",
...     "employee_position_title",
... ]
>>> df = employees.X[cols]
>>> pol_df = pl.from_pandas(df)
>>> %timeit pol_df.select(pl.all().mh.minhash(30))
31.4 ms ± 290 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
>>> mh = MinHashEncoder()
>>> %timeit mh.fit_transform(df)
174 ms ± 239 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
>>> mh = MinHashEncoder(ngram_range=(1, 8), hashing='murmur')
>>> %timeit mh.fit_transform(df)
1.47 s ± 4.57 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
check we got informative features:

```
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.ensemble import HistGradientBoostingRegressor
>>> X = pol_df.select(pl.all().mh.minhash(30))
>>> cross_val_score(HistGradientBoostingRegressor(), X, employees.y)
array([0.83240414, 0.81623089, 0.86734452, 0.86810541, 0.85545101])
```
