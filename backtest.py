import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import product
import json

Venue = namedtuple("Venue", ["venue_id", "ask", "ask_size", "fee", "rebate"])

def load_clean_data(csv_path):
    '''
    Load and preprocess market data from CSV.

    Args:
        csv_path (str): Path to CSV file with columns: ts_event, publisher_id, ask_px_00, ask_sz_00

    Returns:
        snapshot_dict: dict mapping each ts_event -> list of Venue objects
    '''
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Check for required columns
    required_cols = ['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename columns and select necessary ones
    df = df.rename(columns={
        'ask_px_00': 'ask',
        'ask_sz_00': 'ask_size'
    })
    df = df[['ts_event', 'publisher_id', 'ask', 'ask_size']]

    # Clean data
    df = df.dropna()  # Drop rows with missing values
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])  # Ensure unique venue per timestamp
    df = df[df['ask_size'] > 0]  # Filter out invalid ask sizes

    # Convert timestamps to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'])

    # Simulate fee and rebate per venue (consistent across snapshots)
    venue_ids = df['publisher_id'].unique()
    np.random.seed(42)  # For reproducibility
    venue_fees = {v: np.round(np.random.uniform(0.001, 0.003), 4) for v in venue_ids}
    venue_rebates = {v: np.round(np.random.uniform(0.0005, 0.0015), 4) for v in venue_ids}

    # Create snapshots: dict mapping ts_event -> list of Venue objects
    snapshot_dict = {}
    for ts, group in df.groupby('ts_event'):
        venues = [
            Venue(
                venue_id=row['publisher_id'],
                ask=row['ask'],
                ask_size=row['ask_size'],
                fee=venue_fees[row['publisher_id']],
                rebate=venue_rebates[row['publisher_id']]
            )
            for _, row in group.iterrows()
        ]
        snapshot_dict[ts] = venues

    return snapshot_dict

def compute_cost(split, venues, order_size, lambda_over=0.05, lambda_under=0.05, theta_queue=0.0005):
    """
    Compute total cost with market (M) and limit (L_k) orders.
    split = [M, L_1, L_2, ..., L_N]
    """
    M = split[0]  # Market order
    L_values = split[1:]  # Limit orders per venue
    N = len(venues)
    executed = 0
    cash_spent = 0
    remaining_sizes = [v.ask_size for v in venues]

    # Market order execution
    m_to_allocate = M
    sorted_venues = sorted(enumerate(venues), key=lambda x: x[1].ask + x[1].fee)
    for idx, venue in sorted_venues:
        if m_to_allocate <= 0:
            break
        fill = min(m_to_allocate, remaining_sizes[idx])
        executed += fill
        cash_spent += fill * (venue.ask + venue.fee)
        remaining_sizes[idx] -= fill
        m_to_allocate -= fill

    # Limit order execution
    for k, l in enumerate(L_values):
        filled = min(l, remaining_sizes[k])
        executed += filled
        cash_spent += filled * (venues[k].ask + venues[k].fee)
        maker_rebate = max(l - filled, 0) * venues[k].rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (M + sum(L_values) + underfill)  # Risk on total order + underfill
    cost_pen = lambda_under * underfill + lambda_over * overfill
    return cash_spent + risk_pen + cost_pen

def allocate(venues, order_size, lambda_over=0.05, lambda_under=0.05, theta_queue=0.0005, step_size=500):
    N = len(venues)
    total_ask_size = sum(v.ask_size for v in venues)
    if total_ask_size < order_size * 0.1:
        return [0] * (N + 1), float('inf')
    
    best_cost = float('inf')
    best_split = [0] * (N + 1)

    max_m = min(order_size, total_ask_size)
    for m in range(0, int(max_m) + 1, step_size):
        remaining = max(order_size - m, 0)
        l_values = [0] * N
        remaining_sizes = [v.ask_size for v in venues]
        
        # Allocate market orders
        m_to_allocate = m
        sorted_venues = sorted(enumerate(venues), key=lambda x: x[1].ask + x[1].fee)
        for idx, venue in sorted_venues:
            if m_to_allocate <= 0:
                break
            fill = min(m_to_allocate, remaining_sizes[idx])
            m_to_allocate -= fill
            remaining_sizes[idx] -= fill

        # Allocate limit orders using original indices
        remaining = max(order_size - m, 0)
        for idx, venue in enumerate(venues):  # Use original order for limit orders
            if remaining <= 0:
                break
            l = min(remaining, remaining_sizes[idx])
            l_values[idx] = l
            remaining -= l

        split = [m] + l_values
        cost = compute_cost(split, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = split

    return best_split, best_cost

def simulate_cont_kukanov(csv_path, initial_order_size=5000, window="1min", lambda_over=0.05, lambda_under=0.05, theta_queue=0.0005):
    """
    Simulate Cont-Kukanov strategy over windows with merged deduplication of venues.
    """
    snapshots = load_clean_data(csv_path)

    windowed_snapshots = {}
    for ts, venues in snapshots.items():
        window_ts = pd.Timestamp(ts).floor(window)
        if window_ts not in windowed_snapshots:
            windowed_snapshots[window_ts] = {}
        for venue in venues:
            if venue.venue_id in windowed_snapshots[window_ts]:
                # Merge: Sum ask_size, keep latest ask, fee, rebate
                existing = windowed_snapshots[window_ts][venue.venue_id]
                windowed_snapshots[window_ts][venue.venue_id] = Venue(
                    venue_id=venue.venue_id,
                    ask=venue.ask,  # Latest ask price
                    ask_size=existing.ask_size + venue.ask_size,  # Sum ask_size
                    fee=venue.fee,  # Latest fee
                    rebate=venue.rebate  # Latest rebate
                )
            else:
                windowed_snapshots[window_ts][venue.venue_id] = venue
    windowed_snapshots = {ts: list(venues.values()) for ts, venues in windowed_snapshots.items()}

    order_size = initial_order_size
    total_cost = 0
    total_executed = 0

    for ts, venues in windowed_snapshots.items():
        if order_size <= 0:
            break
        if sum(v.ask_size for v in venues) < order_size * 0.1:
            continue
        best_split, snapshot_cost = allocate(venues, order_size, lambda_over, lambda_under, theta_queue)
        executed = sum(min(split_val, venues[i].ask_size) for i, split_val in enumerate(best_split[1:])) + min(best_split[0], sum(v.ask_size for v in venues))
        total_cost += snapshot_cost
        total_executed += executed
        order_size -= executed

    if order_size > 0 and venues:
        last_venue = min(venues, key=lambda v: v.ask + v.fee)
        final_cost = order_size * (last_venue.ask + last_venue.fee)
        total_cost += final_cost
        total_executed += order_size

    avg_fill_price = total_cost / total_executed if total_executed > 0 else 0
    return total_cost, total_executed, avg_fill_price

# New baseline functions
def sim_bestask(csv_path, initial_order_size=5000, window="1min"):
    snapshots = load_clean_data(csv_path)

    windowed_snapshots = {}
    for ts, venues in snapshots.items():
        window_ts = pd.Timestamp(ts).floor(window)
        if window_ts not in windowed_snapshots:
            windowed_snapshots[window_ts] = []
        windowed_snapshots[window_ts].extend(venues)

    order_size = initial_order_size
    total_cost = 0
    total_executed = 0

    for ts, venues in windowed_snapshots.items():
        if order_size <= 0:
            break
        if venues:
            best_venue = min(venues, key=lambda v: v.ask + v.fee)
            executed = min(order_size, best_venue.ask_size)
            cost = executed * (best_venue.ask + best_venue.fee)
            total_cost += cost
            total_executed += executed
            order_size -= executed

    if order_size > 0 and venues:
        last_venue = min(venues, key=lambda v: v.ask + v.fee)
        final_cost = order_size * (last_venue.ask + last_venue.fee)
        total_cost += final_cost
        total_executed += order_size

    avg_fill_price = total_cost / total_executed if total_executed > 0 else 0
    return total_cost, total_executed, avg_fill_price

def twap(csv_path, initial_order_size=5000, window="1min"):
    snapshots = load_clean_data(csv_path)

    windowed_snapshots = {}
    for ts, venues in snapshots.items():
        window_ts = pd.Timestamp(ts).floor(window)
        if window_ts not in windowed_snapshots:
            windowed_snapshots[window_ts] = []
        windowed_snapshots[window_ts].extend(venues)

    num_windows = len(windowed_snapshots)
    shares_per_window = initial_order_size / num_windows if num_windows > 0 else 0
    order_size = initial_order_size
    total_cost = 0
    total_executed = 0
    executed_per_window = {}

    for ts, venues in windowed_snapshots.items():
        if order_size <= 0:
            break
        if venues:
            target = shares_per_window  # Strict TWAP target
            remaining_target = target
            executed = 0
            # Allocate across venues in this window
            for venue in sorted(venues, key=lambda v: v.ask + v.fee):
                if remaining_target <= 0:
                    break
                to_execute = min(remaining_target, venue.ask_size)
                cost = to_execute * (venue.ask + venue.fee)
                total_cost += cost
                executed += to_execute
                remaining_target -= to_execute

            total_executed += executed
            executed_per_window[ts] = executed
            order_size -= executed

    if order_size > 0 and venues:
        last_venue = min(venues, key=lambda v: v.ask + v.fee)
        final_cost = order_size * (last_venue.ask + last_venue.fee)
        total_cost += final_cost
        total_executed += order_size

    avg_fill_price = total_cost / total_executed if total_executed > 0 else 0
    return total_cost, total_executed, avg_fill_price

def vwap(csv_path, initial_order_size=5000, window="1min"):
    snapshots = load_clean_data(csv_path)

    windowed_snapshots = {}
    for ts, venues in snapshots.items():
        window_ts = pd.Timestamp(ts).floor(window)
        if window_ts not in windowed_snapshots:
            windowed_snapshots[window_ts] = []
        windowed_snapshots[window_ts].extend(venues)

    # Calculate total volume (ask_size) across all windows and venues
    total_volume = 0
    volume_per_window = {}
    for ts, venues in windowed_snapshots.items():
        window_volume = sum(v.ask_size for v in venues)
        volume_per_window[ts] = window_volume
        total_volume += window_volume

    order_size = initial_order_size
    total_cost = 0
    total_executed = 0

    for ts, venues in windowed_snapshots.items():
        if order_size <= 0:
            break
        if venues and total_volume > 0:
            # Allocate shares proportional to this window's volume
            window_volume = volume_per_window[ts]
            target_shares = (window_volume / total_volume) * initial_order_size
            target_shares = min(target_shares, order_size)  # Don't exceed remaining shares
            remaining_target = target_shares
            executed = 0

            # Within the window, allocate across venues based on their ask_size
            total_window_ask_size = sum(v.ask_size for v in venues)
            for venue in sorted(venues, key=lambda v: v.ask + v.fee):
                if remaining_target <= 0:
                    break
                weight = venue.ask_size / total_window_ask_size if total_window_ask_size > 0 else 0
                venue_target = weight * target_shares
                to_execute = min(venue_target, venue.ask_size, remaining_target)
                cost = to_execute * (venue.ask + venue.fee)
                total_cost += cost
                executed += to_execute
                remaining_target -= to_execute

            total_executed += executed
            order_size -= executed

    if order_size > 0 and venues:
        last_venue = min(venues, key=lambda v: v.ask + v.fee)
        final_cost = order_size * (last_venue.ask + last_venue.fee)
        total_cost += final_cost
        total_executed += order_size

    avg_fill_price = total_cost / total_executed if total_executed > 0 else 0
    return total_cost, total_executed, avg_fill_price

def evaluate(csv_path, initial_order_size=5000, window="1min"):
    # Cont-Kukanov with default params
    ck_cost, ck_executed, ck_avg_price = simulate_cont_kukanov(csv_path, initial_order_size, window)
    
    # Baselines
    ba_cost, ba_executed, ba_avg_price = sim_bestask(csv_path, initial_order_size, window)
    twap_cost, twap_executed, twap_avg_price = twap(csv_path, initial_order_size, window)
    vwap_cost, vwap_executed, vwap_avg_price = vwap(csv_path, initial_order_size, window)

    # Savings in basis points (1 bp = 0.01%)
    def bps_savings(cost1, cost2):
        if cost2 == 0:
            return 0
        return ((cost2 - cost1) / cost2) * 10000  # Convert to basis points

    ba_savings = bps_savings(ck_cost, ba_cost)
    twap_savings = bps_savings(ck_cost, twap_cost)
    vwap_savings = bps_savings(ck_cost, vwap_cost)

    return {
        "cont_kukanov": {
            "params": {"lambda_over": 0.05, "lambda_under": 0.05, "theta_queue": 0.0005},
            "total_cash_spent": ck_cost,
            "avg_fill_price": ck_avg_price
        },
        "best_ask": {
            "total_cash_spent": ba_cost,
            "avg_fill_price": ba_avg_price
        },
        "twap": {
            "total_cash_spent": twap_cost,
            "avg_fill_price": twap_avg_price
        },
        "vwap": {
            "total_cash_spent": vwap_cost,
            "avg_fill_price": vwap_avg_price
        },
        "savings_bps": {
            "vs_best_ask": ba_savings,
            "vs_twap": twap_savings,
            "vs_vwap": vwap_savings
        }
    }

def gen_output_json(csv_path, initial_order_size=5000, window="1min"):
    result = evaluate(csv_path, initial_order_size, window)
    print(json.dumps(result, indent=2))

def grid_search_params(csv_path, initial_order_size=5000, window="1min"):
    param_grid = {
        "lambda_over": [0.01, 0.05, 0.1],
        "lambda_under": [0.01, 0.05, 0.1],
        "theta_queue": [0.0001, 0.0005, 0.001]
    }
    best_cost = float('inf')
    best_params = {}

    for lo in param_grid["lambda_over"]:
        for lu in param_grid["lambda_under"]:
            for tq in param_grid["theta_queue"]:
                cost, _, _ = simulate_cont_kukanov(csv_path, initial_order_size, window, lo, lu, tq)
                if cost < best_cost:
                    best_cost = cost
                    best_params = {"lambda_over": lo, "lambda_under": lu, "theta_queue": tq}

    return best_params

# Update simulate_cont_kukanov to use best params
def simulate_cont_kukanov_optimized(csv_path, initial_order_size=5000, window="1min"):
    best_params = grid_search_params(csv_path, initial_order_size, window)
    return simulate_cont_kukanov(csv_path, initial_order_size, window, 
                                best_params["lambda_over"], 
                                best_params["lambda_under"], 
                                best_params["theta_queue"])



## Made mock data with multiple venues rather than csv file with single venue
def generate_multi_venue_data(
    csv_path="multi_venue_data.csv",
    seed=42,
    start_time="2025-05-02T10:00:00.000",
    duration_minutes=9,
    interval_ms=54,
    venue_ids=[1, 2, 3, 4, 5],
    base_price=223.0,
    price_std=0.5,
    price_bounds=(220, 226),
    normal_size_range=(50, 500),
    spike_prob=0.05,
    spike_size_range=(1000, 2000)
):
    """
    Generate synthetic multi-venue market data and save as CSV.
    """
    np.random.seed(seed)
    start_time = pd.Timestamp(start_time)
    num_timestamps = int(duration_minutes * 60 * 1000 / interval_ms)
    timestamps = [start_time + pd.Timedelta(milliseconds=interval_ms * i) for i in range(num_timestamps)]

    data = []
    for ts in timestamps:
        for venue in venue_ids:
            price_shift = np.random.normal(0, price_std)
            base_price += price_shift
            ask_price = max(price_bounds[0], min(price_bounds[1], base_price))
            ask_size = np.random.randint(*normal_size_range)
            if np.random.random() < spike_prob:
                ask_size = np.random.randint(*spike_size_range)
            data.append([ts, venue, round(ask_price, 2), ask_size])

    df = pd.DataFrame(data, columns=["ts_event", "publisher_id", "ask_px_00", "ask_sz_00"])
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated {csv_path} with {len(df)} rows and {len(venue_ids)} venues.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python backtest.py <csv_path>")
    else:
        gen_output_json(sys.argv[1])

# Example: 
# generate_multi_venue_data("multi_venue_data1.csv")
# gen_output_json("multi_venue_data1.csv")





