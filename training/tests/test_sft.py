import json

from travel_planner.finetune.data.sft import PromptTemplates, build_messages, target_text, to_sft_example
from travel_planner.finetune.labels import ATTRIBUTES

from conftest import CONFIG_DIR
from travel_planner.finetune.config import load_config

METADATA_KEYS = ("review_id", "place_id", "tier", "synthetic")


def templates() -> PromptTemplates:
    return PromptTemplates.from_config(load_config(CONFIG_DIR / "common.yaml"))


def test_prompt_contains_category_review_and_only_that_category_aspects(small_records):
    record = next(r for r in small_records if r["category"] == "restaurant")
    system, user = build_messages(record, templates())
    assert (system["role"], user["role"]) == ("system", "user")
    assert "evidence" in system["content"] and "JSON" in system["content"]
    assert record["review"] in user["content"]
    assert "(restaurant)" in user["content"]
    assert all(f"- {aspect} (" in user["content"] for aspect in ATTRIBUTES["restaurant"])
    assert "- bed_comfort (" not in user["content"]  # 호텔 전용 aspect는 없다
    assert "$" not in user["content"]  # 템플릿 변수가 모두 채워졌다


def test_target_is_label_json_without_metadata(small_records):
    record = small_records[1]
    example = to_sft_example(record, templates(), {"enable_thinking": False})
    assert [m["role"] for m in example["prompt"]] == ["system", "user"]
    assert [m["role"] for m in example["completion"]] == ["assistant"]
    target = json.loads(example["completion"][0]["content"])
    assert target == record["label"]
    assert list(target) == ["traveler_context", "aspects"]
    text = example["completion"][0]["content"]
    assert not any(key in text for key in METADATA_KEYS)
    assert not any(record[key] in text for key in ("review_id", "place_id"))
    assert example["chat_template_kwargs"] == {"enable_thinking": False}


def test_target_key_order_is_fixed():
    label = {"aspects": [{"evidence": "e", "sentiment": "positive", "attribute": "clean", "category": "cleanliness"}], "traveler_context": ["solo"]}
    assert target_text(label) == (
        '{"traveler_context": ["solo"], "aspects": [{"category": "cleanliness", "attribute": "clean", '
        '"sentiment": "positive", "evidence": "e"}]}'
    )
