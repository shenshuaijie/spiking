from __future__ import annotations
from typing import Union
import numpy as np
import math


Event = Union[tuple[int], list[int], np.array]
EventStream = Union[list[Event], np.array]
Attribute = Union[tuple[int], list[int], np.array]


__all__ = [
    'sbt',
    'sbn',
    'stack_events',
]


def _dvs2frame(height: int,
               width: int,
               x_array: Attribute,
               y_array: Attribute,
               polarity_array: Attribute,
               sign: bool = False,
               depolarize: bool = False):
    """
    Convert a DVS event stream to a frame.
    Args:
        height (int): Height of the frame.
        width (int): Width of the frame.
        x_array (Attribute): x coordinates of the events.
        y_array (Attribute): y coordinates of the events.
        polarity_array (Attribute): Polarity of the events.
        sign (bool, optional): Whether to convert the frame to a sign frame. Defaults to False.
        depolarize (bool, optional): Whether to depolarize the frame. Defaults to False.
    Returns:
        np.array: The frame with shape `(1, height, width)` for non-polarized and shape `(2, height, width)` for polarized.
    """
    if not depolarize:
        frame = np.zeros((height, width))
        np.add.at(frame, (x_array, y_array), polarity_array)
        frame = np.expand_dims(frame, 0)
    else:
        positive_frame = np.zeros((height, width))
        negative_frame = np.zeros((height, width))
        positive_mask = polarity_array > 0
        np.add.at(positive_frame, (x_array[positive_mask], y_array[positive_mask]),
                  polarity_array[positive_mask])
        np.add.at(negative_frame, (x_array[~positive_mask], y_array[~positive_mask]),
                  -polarity_array[~positive_mask])
        frame = np.stack([positive_frame, negative_frame], axis=0)
    if sign:
        frame = np.sign(frame)
    return frame


def sbt(event_stream: EventStream,
        height: int,
        width: int,
        time_axis: int = 0,
        x_axis: int = 1,
        y_axis: int = 2,
        p_axis: int = 3,
        sign: bool = False,
        depolarize: bool = False,
        delta_T: float = 0.00050,
        ):
    """
    Convert a DVS event stream to a sequence of frames with a fixed time interval.
    Args:
        event_stream (EventStream): The event stream with shape `(N, 4)`.
        height (int): Height of the frame.
        width (int): Width of the frame.
        time_axis (int, optional): The axis of the event stream that represents the time. Defaults to 0.
        x_axis (int, optional): The axis of the event stream that represents the x coordinate. Defaults to 1.
        y_axis (int, optional): The axis of the event stream that represents the y coordinate. Defaults to 2.
        p_axis (int, optional): The axis of the event stream that represents the polarity. Defaults to 3.
        sign (bool, optional): Whether to convert the frame to a sign frame. Defaults to False.
        depolarize (bool, optional): Whether to depolarize the frame. Defaults to False.
        delta_T (float, optional): The time interval between two frames. Defaults to 0.00050.
    Returns:
        np.array: The sequence of frames with shape `(N, 1, height, width)` for non-polarized and shape `(N, 2, height, width)` for polarized.
    """

    assert delta_T > 0, 'delta_T must be positive'
    event_stream = event_stream[np.argsort(event_stream[:, time_axis])]
    timestamps = event_stream[:, time_axis]
    start = timestamps[0]
    end = timestamps[-1]
    time_split = np.arange(start, end, delta_T)
    indices_spilt = np.searchsorted(timestamps, time_split)
    event_split = np.split(event_stream.astype(int), indices_spilt)[1:]
    frames = [_dvs2frame(height, width, event[:, x_axis], event[:, y_axis],
                         event[:, p_axis], sign=sign, depolarize=depolarize) for event in event_split]
    return np.array(frames)


def sbn(event_stream: EventStream,
        height: int,
        width: int,
        time_axis: int = 0,
        x_axis: int = 1,
        y_axis: int = 2,
        p_axis: int = 3,
        sign: bool = False,
        depolarize: bool = False,
        num_per_frame: int = 100,
        ):
    """
    Convert a DVS event stream to a sequence of frames with a fixed number of events per frame.
    Args:
        event_stream (EventStream): The event stream with shape `(N, 4)`.
        height (int): Height of the frame.
        width (int): Width of the frame.
        time_axis (int, optional): The axis of the event stream that represents the time. Defaults to 0.
        x_axis (int, optional): The axis of the event stream that represents the x coordinate. Defaults to 1.
        y_axis (int, optional): The axis of the event stream that represents the y coordinate. Defaults to 2.
        p_axis (int, optional): The axis of the event stream that represents the polarity. Defaults to 3.
        sign (bool, optional): Whether to convert the frame to a sign frame. Defaults to False.
        depolarize (bool, optional): Whether to depolarize the frame. Defaults to False.
        num_per_frame (int, optional): The number of events per frame. Defaults to 100.
    Returns:
        np.array: The sequence of frames with shape `(N, 1, height, width)` for non-polarized and shape `(N, 2, height, width)` for polarized.
    """
    assert num_per_frame > 0, 'num_per_frame must be positive'
    event_stream = event_stream[np.argsort(event_stream[:, time_axis])]
    num = math.ceil(event_stream.shape[0]/num_per_frame)
    event_split = np.array_split(event_stream.astype(int), num)
    frames = [_dvs2frame(height, width, event[:, x_axis], event[:, y_axis],
                         event[:, p_axis], sign=sign, depolarize=depolarize) for event in event_split]
    return np.array(frames)


def stack_events(
    event_stream: EventStream,
    height: int,
    width: int,
    based_on: str = 'time',
    time_axis: int = 0,
    x_axis: int = 1,
    y_axis: int = 2,
    p_axis: int = 3,
    sign: bool = False,
    depolarize: bool = False,
    **args
):
    """
    Convert a DVS event stream to a sequence of frames.
    Args:
        event_stream (EventStream): The event stream with shape `(N, 4)`.
        height (int): Height of the frame.
        width (int): Width of the frame.
        based_on (str, optional): The method to stack events. Defaults to 'time'.
        time_axis (int, optional): The axis of the event stream that represents the time. Defaults to 0.
        x_axis (int, optional): The axis of the event stream that represents the x coordinate. Defaults to 1.
        y_axis (int, optional): The axis of the event stream that represents the y coordinate. Defaults to 2.
        p_axis (int, optional): The axis of the event stream that represents the polarity. Defaults to 3.
        sign (bool, optional): Whether to convert the frame to a sign frame. Defaults to False.
        depolarize (bool, optional): Whether to depolarize the frame. Defaults to False.
        **args: Additional arguments for the specific method.
    Returns:
        np.array: The sequence of frames with shape `(N, 1, height, width)` for non-polarized and shape `(N, 2, height, width)` for polarized.
    """
    if isinstance(event_stream, Union[list, tuple]):
        event_stream = np.array(event_stream)
    assert len(np.shape(event_stream)) == 2 and np.shape(event_stream)[
        1] == 4, 'event_stream must be a 2D array with shape (N, 4)'
    assert time_axis in [0, 1, 2, 3] and x_axis in [0, 1, 2, 3] and y_axis in [0, 1, 2, 3] and p_axis in [
        0, 1, 2, 3], 'time_axis, x_axis, y_axis and p_axis must be in [0, 1, 2, 3]'
    assert len(set([time_axis, x_axis, y_axis, p_axis])
               ) == 4, 'time_axis, x_axis, y_axis and p_axis must be different'
    assert height > 0 and width > 0, 'height and width must be positive'
    assert min(event_stream[:, x_axis]) >= 0 and max(
        event_stream[:, x_axis]) < height, 'x_axis must be in [0, height)'
    assert min(event_stream[:, y_axis]) >= 0 and max(
        event_stream[:, y_axis]) < width, 'y_axis must be in [0, width)'

    if based_on == 'time':
        return sbt(event_stream, height, width, time_axis=time_axis, x_axis=x_axis, y_axis=y_axis, p_axis=p_axis, sign=sign, depolarize=depolarize, **args)
    elif based_on == 'num':
        return sbn(event_stream, height, width, time_axis=time_axis, x_axis=x_axis, y_axis=y_axis, p_axis=p_axis, sign=sign, depolarize=depolarize, **args)
    else:
        raise ValueError(f'Not supported based on {based_on}')


if __name__ == "__main__":
    num_events = 1000000
    h, w = 128, 64
    event_stream = np.array([
        np.random.randint(0, 1000000, size=num_events),
        np.random.randint(0, h, size=num_events),
        np.random.randint(0, w, size=num_events),
        np.random.randint(0, 2, size=num_events) * 2 - 1
    ]).T
    event_stream = [list(event) for event in event_stream]
    frames = stack_events(event_stream, h, w,
                          based_on='time', delta_T=50)
    print('sbt, sign: False, depolarity: False, shape: ', frames.shape)
    frames = stack_events(event_stream, h, w,
                          based_on='time', sign=True, delta_T=50)
    print('sbt, sign: True, depolarity: False, shape: ', frames.shape)
    frames = stack_events(event_stream, h, w,
                          based_on='time', depolarize=True, delta_T=50)
    print('sbt, sign: False, depolarity: True, shape: ', frames.shape)
    frames = stack_events(event_stream, h, w,
                          based_on='num', num_per_frame=50)
    print('sbn, sign: False, depolarity: False, shape: ', frames.shape)
    frames = stack_events(event_stream, h, w,
                          based_on='num', sign=True, num_per_frame=50)
    print('sbn, sign: True, depolarity: False, shape: ', frames.shape)
    frames = stack_events(event_stream, h, w,
                          based_on='num', depolarize=True, num_per_frame=50)
    print('sbn, sign: False, depolarity: True, shape: ', frames.shape)
