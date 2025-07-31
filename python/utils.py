from time import perf_counter_ns


def reportTimeUntilNow(t_start: int, identifier: str = "") -> None:
    elapsed_ns = perf_counter_ns() - t_start

    print(f"{identifier}: Took {elapsed_ns/1e3:.2f} us")
