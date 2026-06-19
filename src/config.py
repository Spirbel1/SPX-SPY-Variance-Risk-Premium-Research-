from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Global runtime settings for data, modeling, and backtesting."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"

    start_date: str = os.getenv("START_DATE", "2014-01-01")
    end_date: str | None = os.getenv("END_DATE", None)

    fred_api_key: str | None = os.getenv("FRED_API_KEY", None)

    initial_train_years: int = int(os.getenv("INITIAL_TRAIN_YEARS", "5"))
    test_window_years: int = int(os.getenv("TEST_WINDOW_YEARS", "1"))
    step_years: int = int(os.getenv("STEP_YEARS", "1"))

    weekly_signal_day: str = os.getenv("WEEKLY_SIGNAL_DAY", "FRI")
    transaction_cost_bps: float = float(os.getenv("TRANSACTION_COST_BPS", "2.0"))

    dataset_filename: str = "vrp_dataset.parquet"

    @property
    def dataset_path(self) -> Path:
        return self.data_processed_dir / self.dataset_filename


SETTINGS = Settings()
