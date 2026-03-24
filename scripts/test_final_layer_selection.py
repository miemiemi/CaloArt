import sys

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)

from src.models.calodit_3drope import CaloDit2FinalLayer, CaloLightningDiT, FinalLayer
from src.models.layers_3drope import ClassicDiTFinalLayer, PixArtFinalLayer


def make_model(pe_mode: str, final_layer_type: str, share_mod: bool = True) -> CaloLightningDiT:
    model = CaloLightningDiT(
        input_size=(3, 4, 6),
        patch_size=(1, 2, 3),
        conditions_size=(1, 2, 1),
        condition_embed_dims=(8, 4, 4),
        in_channels=1,
        model_channels=16,
        out_channels=1,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2.0,
        pe_mode=pe_mode,
        dtype="float32",
        use_checkpoint=False,
        share_mod=share_mod,
        initialization="vanilla",
        qk_rms_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rmsnorm=False,
        use_conv=False,
        final_layer_type=final_layer_type,
    )
    model.eval()
    return model


def assert_forward_ok(model: CaloLightningDiT) -> None:
    x, c, t = model.example_input
    with torch.no_grad():
        y = model(x, c, t)
    assert y.shape == x.shape, f"Expected output shape {tuple(x.shape)}, got {tuple(y.shape)}."


def main() -> int:
    torch.manual_seed(0)

    rope_auto = make_model(pe_mode="rope", final_layer_type="auto")
    assert rope_auto.final_layer_type == "pixart"
    assert rope_auto.final_layer_uses_pos_emb is False
    assert isinstance(rope_auto.final_layer, PixArtFinalLayer)
    assert_forward_ok(rope_auto)
    print("PASS rope+auto -> PixArtFinalLayer")

    rope_auto_no_share = make_model(pe_mode="rope", final_layer_type="auto", share_mod=False)
    assert rope_auto_no_share.final_layer_type == "gated"
    assert rope_auto_no_share.final_layer_uses_pos_emb is False
    assert isinstance(rope_auto_no_share.final_layer, FinalLayer)
    assert_forward_ok(rope_auto_no_share)
    print("PASS rope+auto+share_mod=false -> FinalLayer")

    ape_rope_auto = make_model(pe_mode="ape+rope", final_layer_type="auto")
    assert ape_rope_auto.final_layer_type == "calodit2"
    assert ape_rope_auto.final_layer_uses_pos_emb is True
    assert isinstance(ape_rope_auto.final_layer, CaloDit2FinalLayer)
    assert_forward_ok(ape_rope_auto)
    print("PASS ape+rope+auto -> CaloDit2FinalLayer")

    ape_rope_pixart = make_model(pe_mode="ape+rope", final_layer_type="pixart")
    assert ape_rope_pixart.final_layer_type == "pixart"
    assert ape_rope_pixart.final_layer_uses_pos_emb is False
    assert isinstance(ape_rope_pixart.final_layer, PixArtFinalLayer)
    assert_forward_ok(ape_rope_pixart)
    print("PASS ape+rope+pixart -> PixArtFinalLayer")

    ape_rope_calodit2 = make_model(pe_mode="ape+rope", final_layer_type="calodit2")
    assert ape_rope_calodit2.final_layer_type == "calodit2"
    assert ape_rope_calodit2.final_layer_uses_pos_emb is True
    assert isinstance(ape_rope_calodit2.final_layer, CaloDit2FinalLayer)
    assert_forward_ok(ape_rope_calodit2)
    print("PASS ape+rope+calodit2 -> CaloDit2FinalLayer")

    rope_classicdit = make_model(pe_mode="rope", final_layer_type="classicdit", share_mod=False)
    assert rope_classicdit.final_layer_type == "classicdit"
    assert rope_classicdit.final_layer_uses_pos_emb is False
    assert isinstance(rope_classicdit.final_layer, ClassicDiTFinalLayer)
    assert_forward_ok(rope_classicdit)
    print("PASS rope+classicdit -> ClassicDiTFinalLayer")

    rope_final = make_model(pe_mode="rope", final_layer_type="final", share_mod=False)
    assert rope_final.final_layer_type == "gated"
    assert rope_final.final_layer_uses_pos_emb is False
    assert isinstance(rope_final.final_layer, FinalLayer)
    assert_forward_ok(rope_final)
    print("PASS rope+final -> FinalLayer")

    try:
        make_model(pe_mode="rope", final_layer_type="calodit2")
    except ValueError as exc:
        msg = str(exc)
        assert "requires pe_mode to include 'ape'" in msg, msg
    else:
        raise AssertionError("Expected rope+calodit2 construction to fail.")
    print("PASS rope+calodit2 raises ValueError")

    try:
        make_model(pe_mode="rope", final_layer_type="pixart", share_mod=False)
    except ValueError as exc:
        msg = str(exc)
        assert "requires share_mod=True" in msg, msg
    else:
        raise AssertionError("Expected pixart+share_mod=false construction to fail.")
    print("PASS pixart+share_mod=false raises ValueError")

    print("All final layer selection checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
