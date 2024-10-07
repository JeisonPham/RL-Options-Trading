import numpy as np


def extract_time_span(time_interval):
    if time_interval == "1Day":
        multiplier, timespan = 1, "day"
        every = "1d"
    elif time_interval == "1Hour":
        multiplier, timespan = 1, "hour"
        every = "1h"
    elif time_interval == "30minute":
        multiplier, timespan = 30, 'minute'
        every = "30m"
    elif time_interval == "15minute":
        multiplier, timespan = 15, 'minute'
        every = "15m"
    elif time_interval == "1minute":
        multiplier, timespan = 1, "minute"
        every = "1m"
    else:
        raise ValueError("time_interval can only be `1Day` or `1Hour` or `30minute` or `15minute` or `1minute`")

    return multiplier, timespan, every
