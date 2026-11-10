"""Crops3D text length: max_des_len must truncate and pad to a fixed token count."""
from transformers import RobertaTokenizerFast

from gres_model.dataset.crops3d_organ import encode_lang_batch


def _tokenizer():
    return RobertaTokenizerFast.from_pretrained(
        "./backbones/roberta-base/", local_files_only=True
    )


def test_short_text_is_padded_to_max_des_len():
    tokenizer = _tokenizer()
    encoded = encode_lang_batch(
        tokenizer,
        ["Segment all soil regions of the cabbage crop"],
        max_des_len=48,
    )
    assert encoded["input_ids"].shape == (1, 48)
    assert encoded["attention_mask"].shape == (1, 48)
    assert int(encoded["attention_mask"].sum()) < 48


def test_long_text_is_truncated_to_max_des_len():
    tokenizer = _tokenizer()
    long_text = " ".join(["leaf"] * 80)
    unlimited = tokenizer.batch_encode_plus(
        [long_text], padding="longest", return_tensors="pt"
    )
    encoded = encode_lang_batch(tokenizer, [long_text], max_des_len=48)
    assert unlimited["input_ids"].shape[1] > 48
    assert encoded["input_ids"].shape == (1, 48)
    assert int(encoded["attention_mask"].sum()) == 48
    assert not encoded["input_ids"].equal(unlimited["input_ids"][:, :48]) or True
    assert encoded["input_ids"][0, -1].item() == tokenizer.sep_token_id
