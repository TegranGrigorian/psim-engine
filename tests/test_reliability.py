import math
import zipfile
from datetime import date

import numpy as np
import pytest

from psim_engine.reliability import (Leg, Reliability, Schedule, simulate_delays,
                                     summarize_delays)

REL = Reliability(on_time_share=0.82, on_time_window_minutes=5, arrive_margin_minutes=5,
                  target_reliability=0.95, draws=40_000, source="test")


def leg(**kw):
    base = dict(feed="f", route_id="R", stop_id="S", reach_stop=100.0, board=110.0, alight=130.0)
    return Leg(**{**base, **kw})


def schedule(times):
    return {"f": Schedule({("R", "S"): np.array(times, dtype=float)})}


def test_the_rate_reproduces_the_published_on_time_share():
    # P(lateness > 5 min) must equal 1 - 0.82
    assert math.exp(-REL.rate * 5) == pytest.approx(0.18)
    rng = np.random.default_rng(1)
    late = rng.exponential(1 / REL.rate, 200_000)
    assert (late <= 5).mean() == pytest.approx(0.82, abs=0.005)


def test_one_leg_with_no_slack_is_late_by_the_buses_lateness():
    d = simulate_delays([leg(reach_stop=110.0)], schedule([110, 130]), REL, np.random.default_rng(2))
    assert d.min() >= 0
    assert d.mean() == pytest.approx(1 / REL.rate, rel=0.03)  # first leg is never missed


def test_a_planned_wait_absorbs_lateness_only_at_a_transfer():
    # second leg: arrive at the stop 20 min before the bus, so a late first bus does not cost the connection
    legs = [leg(reach_stop=110.0, board=110.0, alight=125.0),
            leg(reach_stop=130.0, board=150.0, alight=170.0)]
    d = simulate_delays(legs, schedule([110, 150, 180]), REL, np.random.default_rng(3))
    assert d.mean() == pytest.approx(1 / REL.rate, rel=0.05), "no missed connections, delay is the last bus's"


def test_a_tight_transfer_is_missed_and_costs_the_headway():
    legs = [leg(reach_stop=100.0, board=100.0, alight=115.0),
            leg(reach_stop=115.0, board=116.0, alight=140.0)]  # 1 minute to make the connection
    d = simulate_delays(legs, schedule([116, 146]), REL, np.random.default_rng(4))
    p_miss = (d >= 30).mean()
    # missed when the first bus is more than the 1 minute cushion later than the second one;
    # for two exponentials with the same rate that is 0.5 * exp(-rate * cushion)
    expected_p_miss = 0.5 * math.exp(-REL.rate * 1.0)
    assert p_miss == pytest.approx(expected_p_miss, abs=0.02)
    assert d.mean() > 1 / REL.rate * 3, "missed transfers dominate the average delay"


def test_no_later_bus_means_a_long_delay():
    legs = [leg(reach_stop=100.0, board=100.0, alight=115.0),
            leg(reach_stop=115.0, board=116.0, alight=140.0)]
    d = simulate_delays(legs, schedule([116]), REL, np.random.default_rng(5))
    # whenever the connection is missed there is no later bus (same 0.5 * exp(-rate) chance)
    assert (d == 180.0).mean() == pytest.approx(0.5 * math.exp(-REL.rate * 1.0), abs=0.02)


def test_summary_is_the_share_within_slack_and_the_target_quantile():
    delays = np.arange(100, dtype=float)  # 0..99 minutes
    p, q = summarize_delays(delays, slack_minutes=9.5, rel=REL)
    assert p == 0.10
    assert q == pytest.approx(94.05)


def test_more_slack_always_means_a_higher_chance_of_being_on_time():
    d = simulate_delays([leg(reach_stop=110.0)], schedule([110]), REL, np.random.default_rng(6))
    ps = [summarize_delays(d, s, REL)[0] for s in (0, 2, 5, 10, 20)]
    assert ps == sorted(ps) and ps[0] < 0.01 and ps[-1] > 0.95


def make_feed(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("calendar.txt", "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nWK,1,1,1,1,1,0,0,20260901,20261231\nSA,0,0,0,0,0,1,0,20260901,20261231\n")
        z.writestr("calendar_dates.txt", "service_id,date,exception_type\nWK,20261111,2\n")
        z.writestr("trips.txt", "route_id,service_id,trip_id\nR1,WK,a\nR1,WK,b\nR1,SA,c\nR2,WK,d\n")
        z.writestr("stop_times.txt",
                   "trip_id,stop_id,departure_time,stop_sequence\n"
                   "a,S1,08:00:00,1\na,S2,08:10:00,2\nb,S1,08:30:30,1\nb,S2,25:10:00,2\nc,S1,09:00:00,1\nd,S1,08:15:00,1\n")


def test_schedule_reads_only_trips_running_on_the_date(tmp_path):
    p = tmp_path / "f.zip"
    make_feed(p)
    s = Schedule.from_gtfs(p, date(2026, 11, 10))  # a Tuesday
    assert s.next_departure("R1", "S1", 0) == 480.0
    assert s.next_departure("R1", "S1", 480.0) == pytest.approx(510.5), "strictly after, with seconds"
    assert s.next_departure("R1", "S1", 511) is None, "the Saturday trip does not run"
    assert s.next_departure("R2", "S1", 0) == 495.0
    assert s.next_departure("R1", "S2", 600) == 1510.0, "times past midnight keep counting"
    assert s.next_departure("RX", "S1", 0) is None


def test_service_exceptions_remove_a_day(tmp_path):
    p = tmp_path / "f.zip"
    make_feed(p)
    s = Schedule.from_gtfs(p, date(2026, 11, 11))  # weekday service removed that day
    assert s.next_departure("R1", "S1", 0) is None
    sat = Schedule.from_gtfs(p, date(2026, 11, 14))
    assert sat.next_departure("R1", "S1", 0) == 540.0
