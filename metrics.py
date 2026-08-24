# metrics.py
from typing import List, Dict, Any, Optional
import math


def _safe_pnl(trade: Dict[str, Any]) -> float:
    """
    Extract a trade's PnL value from common keys in a robust way.
    Prefers 'pnl' then 'pnl_net' then 'pnl_gross' then 0.0.
    """
    for k in ("pnl", "pnl_net", "pnl_gross"):
        if k in trade and trade[k] is not None:
            try:
                return float(trade[k])
            except Exception:
                continue
    return 0.0


def compute_metrics(trades: List[Dict[str, Any]],
                    starting_equity: float = 0.0,
                    returns_precision: int = 8) -> Dict[str, Any]:
    """
    Compute a standard set of backtest metrics from a list of trade dicts.

    Each trade dict is expected to contain at least:
      - 'pnl' or 'pnl_net' or 'pnl_gross' (numeric)
      - 'side' (optional, 'long' or 'short')

    Parameters
    ----------
    trades : List[Dict[str, Any]]
        List of trade dictionaries.
    starting_equity : float
        Equity at the start of the sequence (default 0.0).
    returns_precision : int
        Number of decimal places for floating outputs (for readability).

    Returns
    -------
    Dict[str, Any]
        Dictionary with metrics:
          - total_trades, win_rate (pct), net_profit, gross_profit, gross_loss,
            avg_trade, avg_win, avg_loss, profit_factor, expectancy,
            largest_win, largest_loss, max_drawdown, max_drawdown_pct,
            consecutive_wins, consecutive_losses, longest_win_streak,
            longest_loss_streak, long_trades, short_trades, equity_curve (list)
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "long_trades": 0,
            "short_trades": 0,
            "equity_curve": [starting_equity],
        }

    pnls: List[float] = [_safe_pnl(t) for t in trades]
    total_trades = len(pnls)

    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]

    gross_profit = sum(wins)
    gross_loss = sum(losses)  # negative or zero
    net_profit = sum(pnls)
    avg_trade = net_profit / total_trades

    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0  # negative value

    # Profit factor: gross_profit / abs(gross_loss). If gross_loss == 0, set to None (infinite)
    profit_factor: Optional[float]
    if abs(gross_loss) < 1e-12:
        profit_factor = None
    else:
        profit_factor = gross_profit / abs(gross_loss)

    win_rate = (len(wins) / total_trades) * 100.0

    # Expectancy per trade: (win_rate * avg_win + loss_rate * avg_loss)
    win_prob = len(wins) / total_trades
    loss_prob = len(losses) / total_trades
    expectancy = win_prob * avg_win + loss_prob * avg_loss

    # Equity curve and drawdown (absolute and percent)
    equity = starting_equity
    equity_curve: List[float] = [equity]
    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0

    for p in pnls:
        equity += p
        equity_curve.append(equity)
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = (dd / peak) * 100.0 if peak > 0 else (dd if peak == 0 else dd / peak * 100.0)
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Largest win / loss
    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0

    # Consecutive wins/losses and longest streaks
    consecutive_wins = 0
    consecutive_losses = 0
    longest_win_streak = 0
    longest_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for p in pnls:
        if p > 0:
            current_win_streak += 1
            current_loss_streak = 0
        elif p < 0:
            current_loss_streak += 1
            current_win_streak = 0
        else:
            # zero PnL breaks streaks
            current_win_streak = 0
            current_loss_streak = 0

        if current_win_streak > longest_win_streak:
            longest_win_streak = current_win_streak
        if current_loss_streak > longest_loss_streak:
            longest_loss_streak = current_loss_streak

    consecutive_wins = longest_win_streak
    consecutive_losses = longest_loss_streak

    long_trades = sum(1 for t in trades if str(t.get("side", "")).lower() == "long")
    short_trades = sum(1 for t in trades if str(t.get("side", "")).lower() == "short")

    # Round numeric outputs for readability but keep raw equity_curve values
    def _r(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, float):
            return round(x, returns_precision)
        return x

    metrics = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "net_profit": _r(net_profit),
        "gross_profit": _r(gross_profit),
        "gross_loss": _r(gross_loss),
        "avg_trade": _r(avg_trade),
        "avg_win": _r(avg_win),
        "avg_loss": _r(avg_loss),
        "profit_factor": _r(profit_factor),
        "expectancy": _r(expectancy),
        "largest_win": _r(largest_win),
        "largest_loss": _r(largest_loss),
        "max_drawdown": _r(max_dd),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "consecutive_wins": consecutive_wins,
        "consecutive_losses": consecutive_losses,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "equity_curve": equity_curve,
    }

    return metrics
