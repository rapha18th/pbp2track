"""Library tables on disk: parquet when pyarrow is installed, pandas pickle otherwise."""
import os

import pandas as pd


def has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, root: str, name: str, fmt: str = "auto") -> str:
    """Write <root>/<name>.parquet (or .pkl). Returns the path written."""
    if fmt == "auto":
        fmt = "parquet" if has_parquet() else "pickle"
    path = os.path.join(root, f"{name}.parquet" if fmt == "parquet" else f"{name}.pkl")
    if fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_pickle(path)
    return path


def read_table(root: str, name: str) -> pd.DataFrame:
    """Read <root>/<name>.parquet or <root>/<name>.pkl, whichever exists."""
    pq = os.path.join(root, f"{name}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    pk = os.path.join(root, f"{name}.pkl")
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    raise FileNotFoundError(f"{name}.parquet or {name}.pkl in {root}")
