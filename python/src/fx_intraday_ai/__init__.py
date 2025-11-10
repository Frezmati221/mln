"\"\"\"Intraday FX AI package.\"\"\""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - fallback for editable installs
    __version__ = version("fx_intraday_ai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
