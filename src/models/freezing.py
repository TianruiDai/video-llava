#该脚本为了管理视频编码器E_{V}, 图片编码器E_{I}, 语言模型E_{L}和投影层P的参数冻结

def freeze_video_tower(model):

    for param in (
        model.model.video_tower.parameters()
    ):
        param.requires_grad = False


def freeze_image_tower(model):

    for param in (
        model.model.image_tower.parameters()
    ):
        param.requires_grad = False


def freeze_language_model(model):

    for param in (
        model.model.language_model.parameters()
    ):
        param.requires_grad = False


def freeze_all(model):

    for param in model.parameters():
        param.requires_grad = False


def unfreeze_projector(model):

    for param in (
        model.model
        .multi_modal_projector
        .parameters()
    ):
        param.requires_grad = True