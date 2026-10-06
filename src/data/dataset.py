import json

from torch.utils.data import Dataset

from .video_reader import (
    sample_video_uniform,
)

# Dataset 只负责读取数据,而不是直接返回 Video-LLaVA-specific tensor

class VideoQADataset(Dataset):

    def __init__(
        self,
        annotation_path: str,
        num_frames: int = 8,
    ):

        self.num_frames = num_frames

        self.samples = []

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                self.samples.append(
                    json.loads(line)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        video = sample_video_uniform(
            sample["video"],
            num_frames=self.num_frames,
        )

        return {
            "video": video,
            "question": sample["question"],
            "answer": sample["answer"],
        }