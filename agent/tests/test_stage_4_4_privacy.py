from app.personal_data.privacy import (
    redact_sensitive_text,
    sanitize_personal_value,
)


def test_sensitive_text_is_redacted() -> None:
    text = "身份证 11010519491231002X，手机号 13800138000"
    clean = redact_sensitive_text(text)
    assert "11010519491231002X" not in clean
    assert "13800138000" not in clean
    assert "redacted" in clean


def test_secret_key_is_rejected() -> None:
    try:
        sanitize_personal_value({"api_key": "fake-secret-value"})
    except ValueError as exc:
        assert "禁止保存" in str(exc)
    else:
        raise AssertionError("敏感字段没有被拒绝。")
