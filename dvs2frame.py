import numpy as np
import math


def _dvs2frame(height: int,
               width: int,
               x_array: np.array | list[int] | tuple[int],
               y_array: np.array | list[int] | tuple[int],
               p_array: np.array,
               sign: bool = False):

    frame = np.zeros((height, width))
    np.add.at(frame, (x_array, y_array), p_array)
    if sign:
        frame = np.sign(frame)
    return frame


def sbt(event_stream: np.array,
        height: int,
        width: int,
        delta_T: float,
        time_axis: int = 0,
        x_axis: int = 1,
        y_axis: int = 2,
        p_axis: int = 3,
        sign: bool = False):
    assert len(np.shape(event_stream)) == 2 and np.shape(event_stream)[
        1] == 4, 'event_stream must be a 2D array with shape (N, 4)'
    assert time_axis in [0, 1, 2, 3] and x_axis in [0, 1, 2, 3] and y_axis in [0, 1, 2, 3] and p_axis in [
        0, 1, 2, 3], 'time_axis, x_axis, y_axis and p_axis must be in [0, 1, 2, 3]'
    assert len(set([time_axis, x_axis, y_axis, p_axis])
               ) == 4, 'time_axis, x_axis, y_axis and p_axis must be different'
    assert delta_T > 0, 'delta_T must be positive'
    assert height > 0 and width > 0, 'height and width must be positive'
    event_stream = event_stream[np.argsort(event_stream[:, time_axis])]
    timestamps = event_stream[:, time_axis]
    start = timestamps[0]
    end = timestamps[-1]
    time_split = np.arange(start, end, delta_T)
    indices_spilt = np.searchsorted(timestamps, time_split)
    event_split = np.split(event_stream.astype(int), indices_spilt)[1:]
    frames = [_dvs2frame(height, width, event[:, x_axis], event[:, y_axis],
                         event[:, p_axis], sign=sign) for event in event_split]
    return np.array(frames)


def sbn(event_stream: np.array,
        height: int,
        width: int,
        num_per_frame: int,
        time_axis: int = 0,
        x_axis: int = 1,
        y_axis: int = 2,
        p_axis: int = 3,
        sign: bool = False):
    assert len(np.shape(event_stream)) == 2 and np.shape(event_stream)[
        1] == 4, 'event_stream must be a 2D array with shape (N, 4)'
    assert time_axis in [0, 1, 2, 3] and x_axis in [0, 1, 2, 3] and y_axis in [0, 1, 2, 3] and p_axis in [
        0, 1, 2, 3], 'time_axis, x_axis, y_axis and p_axis must be in [0, 1, 2, 3]'
    assert len(set([time_axis, x_axis, y_axis, p_axis])
               ) == 4, 'time_axis, x_axis, y_axis and p_axis must be different'
    assert num_per_frame > 0, 'num_per_frame must be positive'
    assert height > 0 and width > 0, 'height and width must be positive'
    event_stream = event_stream[np.argsort(event_stream[:, time_axis])]
    num = math.ceil(event_stream.shape[0]/num_per_frame + 1)
    event_split = np.array_split(event_stream.astype(int), num)
    frames = [_dvs2frame(height, width, event[:, x_axis], event[:, y_axis],
                         event[:, p_axis], sign=sign) for event in event_split]
    return np.array(frames)
