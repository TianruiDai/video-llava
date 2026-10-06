from models import build_video_llava


model, processor = build_video_llava()

print(model.config)

for name, module in model.model.named_children():
    print(name, type(module))