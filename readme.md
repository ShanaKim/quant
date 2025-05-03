# Cont-Kukanov Backtest Project
This project implements a backtesting framework to evaluate the Cont-Kukanov order allocation strategy against standard baselines using synthetic multi-venue market data.

## 1. Problems with Pseudocode and Understanding the Cont-Kukanov Model

The original pseudocode had issues that undermined its effectiveness:
- `max_v ← min(order_size-used, venues[v].ask_size)` ensured no overfill by capping allocations per venue, but combined with `if sum(alloc) ≠ order_size: continue`, it prevented any underfill or overfill, rendering parameter tuning (e.g., `lambda_over`, `lambda_under`) useless as the optimizer couldn’t adjust for risk or penalties.
- I understood the Cont-Kukanov model as an optimization strategy that splits orders into market (`M`) and limit (`L_k`) components across venues, minimizing cost by leveraging price differences, rebates, and penalties for overfill/underfill.

## 2. Issues with l1_day.csv and Multi-Venue Data Creation

The provided `l1_day.csv` had only one publisher ID (2), which was insufficient for testing the Cont-Kukanov model’s efficiency against baselines (Best Ask, TWAP, VWAP), as it requires multiple venues to optimize order execution. To address this, I created a synthetic `multi_venue_data.csv` with 5 venues. 

## 3. Function Structures and Logic

- **allocate(venues, order_size, lambda_over, lambda_under, theta_queue, step_size):**
  - Iterates over market order sizes (`M`) in `step_size` increments, allocates remaining shares as limit orders (`L_k`) per venue, and returns the best split and cost using `compute_cost`.
  - Logic: Sorts venues by `ask + fee`, fills market orders first, then limit orders, skipping windows with insufficient liquidity (`< order_size * 0.1`).

- **compute_cost(split, venues, order_size, lambda_over, lambda_under, theta_queue):**
  - Calculates total cost as cash spent (market + limit fills, adjusted for fees/rebates) plus penalties for overfill (`lambda_over`), underfill (`lambda_under`), and risk (`theta_queue * total_order`).
  - Logic: Executes market orders at cheapest venues, applies limit orders, and adds penalties based on execution deviation.

- **simulate_cont_kukanov(csv_path, initial_order_size, window, lambda_over, lambda_under, theta_queue):**
  - Aggregates snapshots into 1-minute windows, merges duplicate venues by summing `ask_size`, and optimizes order execution across windows.
  - Logic: Executes `initial_order_size` shares, skipping low-liquidity windows, and uses the last venue’s price for remaining shares.

## 4. Grid Search Range and Reasonableness

I chose the grid search range:
- `"lambda_over": [0.01, 0.05, 0.1]`
- `"lambda_under": [0.01, 0.05, 0.1]`
- `"theta_queue": [0.0001, 0.0005, 0.001]`
This range is reasonable as it covers typical penalty values (0.01–0.1 for over/underfill, 0.0001–0.001 for risk), reflecting realistic market constraints and allowing the optimizer to find a balance between cost and execution risk.

## 5. Output on New multi_venue_data.csv

The evaluation on my new `multi_venue_data.csv` showed:
- Cont-Kukanov: Total Cash Spent: 1,100,011.0, Avg Fill Price: 220.0022  
- Best Ask: Total Cash Spent: 1,128,356.5, Avg Fill Price: 225.6713  
- TWAP: Total Cash Spent: 1,123,339.83, Avg Fill Price: 224.6680  
- VWAP: Total Cash Spent: 1,125,940.96, Avg Fill Price: 225.1882  
- Savings: +251.2 bps (vs. Best Ask), +207.7 bps (vs. TWAP), +230.3 bps (vs. VWAP)  
This indicates Cont-Kukanov outperformed all baselines, achieving a lower average fill price (~220) versus others (~224–225), likely by capturing more favorable prices with its static allocation logic.


## 6. Idea for Improving Fill Realism

To enhance fill realism, I could incorporate slippage by adjusting `ask` prices based on executed volume (e.g., increasing `ask` by 0.01 per 1000 shares traded) and model queue position effects by reducing `ask_size` as orders fill, reflecting real-world market dynamics.