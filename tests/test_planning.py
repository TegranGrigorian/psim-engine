import pytest

from psim_engine.planning import latest_departure


def test_picks_the_latest_departure_that_still_arrives_by_the_deadline():
    # a bus leaves every 15 minutes (at :00, :15, :30, :45) and takes 20 minutes; a rider leaving
    # at minute d catches the next bus: arrival = next_bus(d) + 20, so T(d) = arrival - d
    def travel(d):
        nxt = -(-d // 15) * 15
        return nxt + 20 - d

    # start 09:00 = 540, margin 5: arrive by 535, so catch the 514 bus? buses at ..., 495, 510, 525.
    # 510 arrives at 530 (ok); 525 arrives at 545 (too late)
    d, t = latest_departure(travel, due_minute=540, margin=5)
    arrival = d + t
    assert arrival <= 535
    assert d == 510 and t == 20, "leave just in time for the 510 bus, wait 0"
    # and nothing later than d works
    assert all(d2 + travel(d2) > 535 for d2 in range(d + 1, 536))


def test_a_constant_travel_time_leaves_exactly_travel_time_before_the_deadline():
    d, t = latest_departure(lambda d: 30.0, due_minute=600, margin=5)
    assert (d, t) == (565, 30.0)  # 565 + 30 = 595 = 600 - 5


def test_no_itinerary_early_enough_means_no_plan():
    assert latest_departure(lambda d: 400.0, due_minute=600, margin=5) is None
    assert latest_departure(lambda d: None, due_minute=600, margin=5) is None


def test_service_that_only_exists_late_enough_is_found():
    # no service before minute 500; afterwards 25 minutes
    def travel(d):
        return None if d < 500 else 25.0

    d, t = latest_departure(travel, due_minute=560, margin=5)
    assert (d, t) == (530, 25.0)


def test_service_that_ends_before_the_deadline_gives_an_earlier_departure():
    # last bus leaves at 520; after that no itinerary
    def travel(d):
        return None if d > 520 else 520 + 15 - d

    d, t = latest_departure(travel, due_minute=600, margin=5)
    assert d == 520 and d + t == 535


def test_the_search_is_logarithmic_not_a_scan():
    calls = []

    def travel(d):
        calls.append(d)
        return 30.0

    latest_departure(travel, due_minute=900, margin=5)
    assert len(calls) <= 12, f"{len(calls)} evaluations"


def test_margin_is_respected_exactly():
    for margin in (0, 5, 10):
        d, t = latest_departure(lambda d: 30.0, due_minute=600, margin=margin)
        assert d + t == 600 - margin
