from pathlib import Path

WF = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")


def test_deploy_skips_gracefully_when_vps_is_unreachable():
    """VPS chỉ bật khi cần: nếu cổng 22 không phản hồi thì bỏ qua deploy (cảnh báo), không làm workflow đỏ."""
    assert "/dev/tcp/" in WF and "::warning::" in WF
    assert "steps.up.outputs.up == 'true'" in WF


def test_deploy_still_pins_host_key_and_has_timeout():
    assert "fingerprint: ${{ vars.VPS_FINGERPRINT }}" in WF
    assert "command_timeout" in WF and "script_stop: true" in WF


def test_deploy_is_gated_by_repo_variable():
    assert "vars.DEPLOY_ENABLED == 'true'" in WF
