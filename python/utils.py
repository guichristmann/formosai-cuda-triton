from time import perf_counter_ns


def reportTimeUntilNow(t_start: int, identifier: str = "", unit: str = "ms") -> None:
    elapsed_ns = perf_counter_ns() - t_start

    if unit == "us":
        print(f"{identifier}: Took {elapsed_ns/1e3:.2f} us")
    elif unit == "ms":
        print(f"{identifier}: Took {elapsed_ns/1e6:.2f} ms")
