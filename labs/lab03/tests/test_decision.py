import json
import pytest
from labs.lab03.src.decision import load_decision_policy, apply_policy, checkpoint_identity


def test_policy_rejects_test_selection_and_applies_validation_rule(tmp_path):
    model = tmp_path / "best_model"; model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture-weights-only-for-hash")
    assert load_decision_policy(model) is None
    path = model / "decision_threshold.json"
    policy={"split":"test","selected_threshold":.3,"checkpoint_identity":checkpoint_identity(model)}
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match="validation"):
        load_decision_policy(model)
    policy["split"]="validation";path.write_text(json.dumps(policy))
    policy = load_decision_policy(model)
    rows = apply_policy([{"label":"NEGATIVE","score":.6,"probabilities":[.6,.4]}],policy,{0:"NEGATIVE",1:"POSITIVE"})
    assert rows[0]["label"] == "POSITIVE"
    assert rows[0]["argmax_label"] == "NEGATIVE"
    assert rows[0]["score"] == .4
    (model / "model.safetensors").write_bytes(b"different-weights")
    with pytest.raises(ValueError,match="SHA256"):
        load_decision_policy(model)
