from app.memory.long_term_memory import LongTermMemoryService
from app.personal_data.privacy import sanitize_personal_value


def test_long_memory_whitelist_accepts_finance_fact() -> None:
    service = LongTermMemoryService(
        postgres_dsn="postgresql://unused", strict_whitelist=True
    )
    service.validate_fact_key(
        fact_type="family_finance",
        fact_key="annual_necessary_expense",
    )


def test_long_memory_whitelist_rejects_unknown_fact() -> None:
    service = LongTermMemoryService(
        postgres_dsn="postgresql://unused", strict_whitelist=True
    )
    try:
        service.validate_fact_key(
            fact_type="private", fact_key="bank_password"
        )
    except ValueError as exc:
        assert "不允许保存" in str(exc)
    else:
        raise AssertionError("未知事实类型没有被拒绝。")


def test_long_memory_original_text_is_redacted() -> None:
    value = sanitize_personal_value(
        {
            "amount": 180000,
            "original_text": "我的手机号是13800138000，年度支出18万元",
        }
    )
    assert "13800138000" not in value["original_text"]
    assert value["amount"] == 180000
