import argparse
import torch

from models import (
    build_video_llava,
)

from data import (
    sample_video_uniform,
)



parser = argparse.ArgumentParser()

parser.add_argument(
    "--video",
    required=True,
    type=str,
)

parser.add_argument(
    "--question",
    default="Describe the video.",
    type=str,
)

args = parser.parse_args()

model, processor = build_video_llava()


model.eval()

video = sample_video_uniform(
    args.video,
    num_frames=8,
)

prompt = (
    "USER: <video>\n"
    f"{args.question} "
    "ASSISTANT:"
)

inputs = processor(
    text=prompt,
    videos=video,
    return_tensors="pt",
)

device = next(
    model.parameters()
).device

inputs = {
    key: (
        value.to(
            device=device,
            dtype=torch.bfloat16,
        )
        if torch.is_floating_point(value)
        else value.to(device)
    )
    for key, value in inputs.items()
}

with torch.inference_mode():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

prompt_length = (
    inputs["input_ids"].shape[1]
)

generated_ids = (
    output_ids[:, prompt_length:]
)

answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print(answer)
