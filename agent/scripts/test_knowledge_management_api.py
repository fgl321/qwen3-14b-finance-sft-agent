from __future__ import annotations

import json
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"
SAMPLE_FILE = Path("data/raw/finance_sample.txt")


def main() -> None:
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"测试文件不存在：{SAMPLE_FILE}")

    with httpx.Client(
        base_url=BASE_URL,
        timeout=180,
        trust_env=False,
    ) as client:
        print("\n========== health ==========")
        health_response = client.get("/health")
        health_response.raise_for_status()
        print(json.dumps(health_response.json(), ensure_ascii=False, indent=2))

        upload_data = _upload_document(client)

        document_id = upload_data["document"]["document_id"]

        list_before_delete = _list_documents(client)

        _assert_document_exists(
            documents=list_before_delete["documents"],
            document_id=document_id,
        )

        _delete_document(
            client=client,
            document_id=document_id,
        )

        list_after_delete = _list_documents(client)

        _assert_document_not_exists(
            documents=list_after_delete["documents"],
            document_id=document_id,
        )

        upload_data_again = _upload_document(client)

        document_id_again = upload_data_again["document"]["document_id"]

        if document_id_again != document_id:
            raise AssertionError(
                "同一个文件重复上传后 document_id 应该稳定不变。"
            )

        print("\n" + "=" * 80)
        print("知识库管理接口测试通过")
        print("=" * 80)


def _upload_document(
    client: httpx.Client,
) -> dict:
    print("\n========== 上传文档 ==========")

    with SAMPLE_FILE.open("rb") as file:
        response = client.post(
            "/api/knowledge/documents",
            data={
                "tenant_id": "tenant_001",
                "owner_user_id": "u001",
                "knowledge_base_id": "kb_finance_basic",
                "visibility": "private",
            },
            files={
                "file": (
                    SAMPLE_FILE.name,
                    file,
                    "text/plain",
                )
            },
        )

    print("status_code:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if data.get("ok") is not True:
        raise AssertionError("上传响应 ok 应为 true。")

    if data["document"]["file_name"] != SAMPLE_FILE.name:
        raise AssertionError("file_name 应该是原始文件名。")

    return data


def _list_documents(
    client: httpx.Client,
) -> dict:
    print("\n========== 文档列表 ==========")

    response = client.get(
        "/api/knowledge/documents",
        params={
            "tenant_id": "tenant_001",
            "owner_user_id": "u001",
            "knowledge_base_id": "kb_finance_basic",
            "limit": 50,
        },
    )

    print("status_code:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if data.get("ok") is not True:
        raise AssertionError("文档列表响应 ok 应为 true。")

    if "documents" not in data:
        raise AssertionError("文档列表缺少 documents 字段。")

    return data


def _delete_document(
    *,
    client: httpx.Client,
    document_id: str,
) -> dict:
    print("\n========== 删除文档 ==========")

    response = client.delete(
        f"/api/knowledge/documents/{document_id}",
        params={
            "tenant_id": "tenant_001",
            "owner_user_id": "u001",
            "knowledge_base_id": "kb_finance_basic",
        },
    )

    print("status_code:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if data.get("ok") is not True:
        raise AssertionError("删除响应 ok 应为 true。")

    if data.get("deleted_count_estimate", 0) <= 0:
        raise AssertionError("deleted_count_estimate 应该大于 0。")

    return data


def _assert_document_exists(
    *,
    documents: list[dict],
    document_id: str,
) -> None:
    for document in documents:
        if document.get("document_id") == document_id:
            return

    raise AssertionError(f"文档列表中没有找到 document_id：{document_id}")


def _assert_document_not_exists(
    *,
    documents: list[dict],
    document_id: str,
) -> None:
    for document in documents:
        if document.get("document_id") == document_id:
            raise AssertionError(f"文档删除后仍然存在：{document_id}")


if __name__ == "__main__":
    main()
