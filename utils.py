from time import time_ns


def measure_time(function) -> any:
    def timed(*args, **kw):
        start = time_ns()
        result = function(*args, **kw)
        time_spent = (time_ns() - start) / 1_000_000_000
        print(f'in ' + '{:.2f}'.format(time_spent) + ' sec\n')
        return result
    return timed

