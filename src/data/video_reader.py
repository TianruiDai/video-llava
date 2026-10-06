import av
import numpy as np

#该脚本为了读取data中的视频数据，然后切为num_frames = 8 frames， 输出一个数组
#数组shape 为 （T, H, W, 3)

def sample_video_uniform(
    video_path: str,
    num_frames: int = 8,
) -> np.ndarray:

    container = av.open(video_path)

    stream = container.stream.video[0]

    total_frames = stream.frames

    # check valid frames

    if total_frames <= 0:
        raise ValueError(
            f' Can not determine total frames for {video_path}'
        )

    indices = np.linespace(0, total_frames-1, num_frames).astype(int)

    index_set = set(indices.tolist())

    frames = []

    for frame_idx, frame in enumerate(
        container.decode(video = 0)
    ):
        if frame_idx in index_set:
            frames.append(
                frame.to_ndarray(format = 'rgb24')
            )
        if frame_idx > indices[-1]:
            break

    container.close()

    #check number of frames

    if len(frames) != num_frames:
        raise RuntimeError(
            f'Expected {num_frames} frames, '
            f'but got {len(frames)} frames'
        )

    return np.stack(frames)

    


