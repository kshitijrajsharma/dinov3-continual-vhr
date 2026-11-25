import timm
import torch
from dinov3_vhr.ssl.explora import LoRALinear, apply_explora, merged_state_dict
from timm.optim import param_groups_layer_decay

MODEL = "vit_small_patch16_dinov3.lvd1689m"


def _model():
    return timm.create_model(MODEL, pretrained=False, num_classes=0)


def test_apply_explora_trainable_subset():
    model = _model()
    trainable = apply_explora(model, rank=4, alpha=4.0, unfreeze_blocks=[11])
    grad = {n for n, p in model.named_parameters() if p.requires_grad}
    assert grad == trainable
    # frozen base of an adapted block, trainable adapters and last block
    assert not model.blocks[0].attn.qkv.weight.requires_grad
    assert model.blocks[0].attn.qkv.lora_A.requires_grad
    assert model.blocks[11].attn.qkv.weight.requires_grad
    # norms, prefix tokens, and stem stay trainable
    assert model.blocks[0].norm1.weight.requires_grad
    assert model.cls_token.requires_grad and model.reg_token.requires_grad
    assert model.patch_embed.proj.weight.requires_grad


def test_zero_init_leaves_forward_unchanged():
    torch.manual_seed(0)
    model = _model()
    reference = _model()
    reference.load_state_dict(model.state_dict())
    apply_explora(model, rank=4, alpha=8.0, unfreeze_blocks=[11])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_features(x), reference.forward_features(x))


def test_merged_state_dict_matches_plain_model():
    model = _model()
    apply_explora(model, rank=4, alpha=8.0, unfreeze_blocks=[11])
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.1)
    merged = merged_state_dict(model)
    plain = _model()
    assert set(merged) == set(plain.state_dict())
    plain.load_state_dict(merged)  # strict
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_features(x), plain.forward_features(x))


def test_layer_decay_groups_include_adapters():
    model = _model()
    apply_explora(model, rank=4, alpha=4.0, unfreeze_blocks=[11])
    groups = param_groups_layer_decay(model, weight_decay=0.05, layer_decay=0.9)
    grouped = sum(len(g["params"]) for g in groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == trainable
