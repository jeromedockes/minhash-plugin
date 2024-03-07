from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.utils.udfs import _get_shared_lib_location

from minhash.utils import parse_into_expr

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

lib = _get_shared_lib_location(__file__)

def minhash(expr: IntoExpr, *, seed: int) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="minhash",
        is_elementwise=True,
        kwargs={"seed": seed}
    )

