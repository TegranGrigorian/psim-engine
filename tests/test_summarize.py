import pandas as pd

from psim_engine.summarize import option_metrics, query_outcomes

T0 = pd.Timestamp("2026-11-10 08:00:00")


def seg(qid, option, segment, mode, start_min, minutes):
    dep = pd.NaT if start_min is None else T0 + pd.Timedelta(minutes=start_min)
    return {
        "from_id": qid, "option": option, "segment": segment, "transport_mode": mode,
        "departure_time": dep, "travel_time": pd.Timedelta(minutes=minutes),
    }


def deps(*ids):
    return pd.Series({i: T0 for i in ids})


def targets(**minutes):
    return pd.DataFrame({
        "query_id": list(minutes), "departure": [T0] * len(minutes),
        "target_minutes": list(minutes.values()),
    })


def test_total_is_measured_from_the_requested_departure_so_it_includes_the_origin_wait():
    # asked to leave 08:00; walk starts 08:04 (4 min wait at home), walk 5, bus 10, walk 3
    df = pd.DataFrame([
        seg("q", 1, 0, "TransportMode.WALK", 4, 5),
        seg("q", 1, 1, "TransportMode.BUS", 12, 10),
        seg("q", 1, 2, "TransportMode.WALK", 22, 3),
    ])
    m = option_metrics(df, deps("q")).iloc[0]
    assert m.total_minutes == 25            # 08:25 - 08:00
    assert m.walk_minutes == 8 and m.in_vehicle_minutes == 10
    assert m.wait_minutes == 7              # 4 at home + 3 at the stop
    assert m.transfers == 0


def test_components_add_up_to_the_total():
    df = pd.DataFrame([
        seg("q", 1, 0, "WALK", 1, 2), seg("q", 1, 1, "BUS", 5, 10),
        seg("q", 1, 2, "BUS", 20, 10), seg("q", 1, 3, "WALK", 30, 2),
    ])
    m = option_metrics(df, deps("q")).iloc[0]
    assert m.transfers == 1
    assert m.walk_minutes + m.in_vehicle_minutes + m.wait_minutes == m.total_minutes


def test_direct_walk_option_uses_walking_time():
    m = option_metrics(pd.DataFrame([seg("q", 0, 0, "WALK", None, 9.5)]), deps("q")).iloc[0]
    assert m.total_minutes == 9.5 and m.in_vehicle_minutes == 0 and m.wait_minutes == 0


def _transit(qid, option, bus_at):
    return [seg(qid, option, 0, "WALK", 0, 2), seg(qid, option, 1, "BUS", bus_at, 11),
            seg(qid, option, 2, "WALK", bus_at + 11, 2)]  # total = bus_at + 13


def test_picks_the_option_closest_to_the_matrix_total():
    # option totals 23, 33, 93 (a bad path r5py also lists); the matrix said 31
    rows = _transit("q", 1, 10) + _transit("q", 2, 20) + _transit("q", 3, 80)
    out = query_outcomes(pd.DataFrame(rows), targets(q=31.0)).iloc[0]
    assert out.completed and out.total_minutes == 33


def test_a_walk_is_chosen_when_it_matches_the_matrix_total():
    rows = [seg("q", 0, 0, "WALK", None, 9.5)] + _transit("q", 1, 20)
    out = query_outcomes(pd.DataFrame(rows), targets(q=9.0)).iloc[0]
    assert out.total_minutes == 9.5 and out.in_vehicle_minutes == 0


def test_ties_go_to_the_lowest_option_number():
    rows = _transit("q", 2, 10) + _transit("q", 1, 10)
    out = query_outcomes(pd.DataFrame(rows), targets(q=23.0)).iloc[0]
    assert out.completed and out.total_minutes == 23


def test_queries_without_options_are_not_completed():
    rows = _transit("a", 1, 10)
    out = query_outcomes(pd.DataFrame(rows), targets(a=23.0, missing=20.0))
    assert list(out.completed) == [True, False]
    assert pd.isna(out.loc[1, "total_minutes"])


def test_empty_results_are_all_not_completed():
    empty = pd.DataFrame([seg("x", 0, 0, "WALK", 0, 1)]).iloc[0:0]
    out = query_outcomes(empty, targets(a=10.0, b=12.0))
    assert list(out.completed) == [False, False]
