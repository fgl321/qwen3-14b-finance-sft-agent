from __future__ import annotations

import pytest

from app.core.json_utils import extract_json_object, parse_arguments


class TestParseArguments:
    def test_dict_passthrough(self) -> None:
        assert parse_arguments({"a": 1}) == {"a": 1}

    def test_json_string(self) -> None:
        assert parse_arguments('{"a": 1}') == {"a": 1}

    def test_none_and_empty(self) -> None:
        assert parse_arguments(None) == {}
        assert parse_arguments("") == {}
        assert parse_arguments("   ") == {}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_arguments("{not json}")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_arguments("[1, 2]")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_arguments(123)


class TestExtractJsonObject:
    def test_plain_json(self) -> None:
        assert extract_json_object('{"answer": "ok"}') == {
            "answer": "ok"
        }

    def test_fenced_json(self) -> None:
        assert extract_json_object(
            '```json\n{"answer": "ok"}\n```'
        ) == {"answer": "ok"}

    def test_text_wrapped_json(self) -> None:
        assert extract_json_object(
            '以下是结果：{"answer": "ok"} 结束'
        ) == {"answer": "ok"}

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json_object("")

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json_object("没有 JSON 对象")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json_object('{"a": }')
