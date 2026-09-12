import numpy as np
from src.data import get_market_returns
from src.regime import (
    detect_regimes,
    crisis_flag_evaluation,
    causal_crisis_flag_evaluation,
)

returns, dates, _ = get_market_returns("^GSPC", "2018-01-01", "2024-12-31")

# Retrospective (same-window) -- the tautological one
result = detect_regimes(returns, dates, window_size=64, step=10)
window_starts = list(range(0, len(returns) - 64, 10))
retro = crisis_flag_evaluation(
    returns, result["verdicts"], window_starts, window_size=64)
print(f"RETROSPECTIVE: tp={retro['tp']} fp={retro['fp']} "
      f"fn={retro['fn']} tn={retro['tn']}  "
      f"prec={retro['precision']:.3f} rec={retro['recall']:.3f}")

# Causal (future-block) -- the honest one
causal = causal_crisis_flag_evaluation(
    returns, window_size=64, step=10,
    crisis_percentile=95, flag_percentile=95)
print(f"CAUSAL:        tp={causal['tp']} fp={causal['fp']} "
      f"fn={causal['fn']} tn={causal['tn']}  "
      f"prec={causal['precision']:.3f} rec={causal['recall']:.3f}")