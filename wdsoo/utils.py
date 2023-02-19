import numpy as np
from typing import Union


def lengthen_to_n(input_arr: Union[list, np.array], n):
    if isinstance(input_arr, list):
        input_arr = np.array(input_arr)

    # Determine the number of times to repeat the array
    num_repeats = n // input_arr.shape[0] + 1

    # Repeat the array the necessary number of times
    repeated_arr = np.tile(input_arr, num_repeats)

    # Slice the repeated array to the desired length
    output_arr = repeated_arr[:n]

    return output_arr


def get_hours_between_timestamps(t1, t2):
    if t1 > t2:
        raise Exception('t2 must be greater than t1')
    delta = t2 - t1
    days, sec = delta.days, delta.seconds
    return days * 24 + sec // 3600
