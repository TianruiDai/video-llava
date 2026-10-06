import torch

# video_tower 对应 视频编码器 E_{V}
# 该脚本为了拿到视频经过视频编码器后的特征 Z_{V} = E_{V}(X_{V})

@torch.no_grad()

def extract_video_tower_features(
    model,
    pixel_values_videos,
    layer=-2,
):

    batch_size, num_frames = (
        pixel_values_videos.shape[:2]
    )

    pixel_values = (
        pixel_values_videos.flatten(0, 1)
    )

    outputs = model.model.video_tower(
        pixel_values,
        output_hidden_states=True,
    )

    features = outputs.hidden_states[layer]

    features = features.reshape(
        batch_size,
        num_frames,
        features.shape[-2],
        features.shape[-1],
    )

    return features