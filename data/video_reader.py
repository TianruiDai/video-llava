import av
import numpy as np


def sample_video_uniform(
    video_path: str,
    num_frames: int = 8,
) -> np.ndarray:

    """
    Returns
    -------
    frames:
        np.ndarray
        shape = [T, H, W, 3]
        dtype = uint8
    """

    container = av.open(video_path)

    total_frames = container.streams.video[0].frames

    if total_frames <= 0:
        raise ValueError(
            f"Cannot determine number of frames: {video_path}"
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []
    index_set = set(indices.tolist())

    for i, frame in enumerate(container.decode(video=0)):
        if i in index_set:
            frames.append(
                frame.to_ndarray(format="rgb24")
            )

        if i > indices[-1]:
            break

    container.close()

    if len(frames) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames, "
            f"got {len(frames)}."
        )

    return np.stack(frames)

