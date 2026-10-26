"""Planning a trip the way a rider with a trip planner would.

A rider with a class or shift at `due` leaves as late as the timetable allows and still
arrives `margin` minutes before it ("arrive-by" planning). r5py can only route forward from a
departure time, so we search: the arrival time d + T(d) never decreases as the departure d
grows, which makes the latest feasible departure findable by a scan and a bisection.
"""

from typing import Callable

SEARCH_BACK_MINUTES = 150  # how far before the deadline we are willing to look


SCAN_STEP_MINUTES = 10


def latest_departure(
    travel_minutes: Callable[[int], float | None],
    due_minute: float,
    margin: float,
    search_back: int = SEARCH_BACK_MINUTES,
) -> tuple[int, float] | None:
    """The latest integer departure minute `d` with `d + T(d) <= due - margin`.

    `travel_minutes(d)` is the door-to-door time when leaving at minute `d` after midnight
    (None if there is no itinerary). Returns `(d, T(d))`, or None if nothing within
    `search_back` minutes of the deadline arrives in time.

    The feasible departures form an interval: too early and the router finds nothing (no
    service yet, or the trip would exceed its time cap), too late and the arrival is after the
    deadline. We scan down from the deadline for the first feasible departure, then bisect up
    to the last one.
    """
    target = due_minute - margin
    top = int(target)  # leaving later than this can never arrive by `target`
    floor = top - search_back

    def feasible(d: int) -> float | None:
        t = travel_minutes(d)
        return t if t is not None and d + t <= target + 1e-9 else None

    d = top
    while d >= floor:
        t = feasible(d)
        if t is not None:
            break
        d -= SCAN_STEP_MINUTES
    else:
        return None
    lo, best_t = d, t
    hi = min(top, d + SCAN_STEP_MINUTES - 1)  # d + STEP was already seen to be infeasible
    while lo < hi:
        mid = (lo + hi + 1) // 2
        t = feasible(mid)
        if t is not None:
            lo, best_t = mid, t
        else:
            hi = mid - 1
    return lo, best_t
