from dirty_signal_lab.ingest import generate_dirty_ticks
from dirty_signal_lab.clean import clean_ticks
from dirty_signal_lab.features import compute_features
from dirty_signal_lab.backtest import backtest


def test_pipeline_smoke():
    df = generate_dirty_ticks("TEST", n=2000, seed=1)
    clean_df, _ = clean_ticks(df)
    feat = compute_features(clean_df).dropna()
    perf = backtest(feat)
    assert perf["n_obs"] > 0
