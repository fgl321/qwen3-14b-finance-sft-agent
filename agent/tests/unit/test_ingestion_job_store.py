from __future__ import annotations

from app.rag.ingestion_jobs import IngestionJobStore


def test_ingestion_job_lifecycle_is_durable(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = IngestionJobStore(database)
    job = store.create(
        file_name="report.pdf",
        stored_path="data/uploads/report.pdf",
        tenant_id="personal",
        owner_user_id="owner",
        knowledge_base_id="kb_finance_basic",
    )

    assert job["status"] == "queued"
    assert job["phase"] == "queued"
    assert job["progress_percent"] == 0
    store.set_processing(job["job_id"])
    assert store.require(job["job_id"])["status"] == "processing"
    store.set_progress(
        job["job_id"],
        phase="embedding",
        percent=63.5,
        message="GPU 向量化 256/400 个唯一子块",
    )
    processing = store.require(job["job_id"])
    assert processing["phase"] == "embedding"
    assert processing["progress_percent"] == 63.5
    assert processing["progress_message"].startswith("GPU 向量化")

    store.set_completed(job["job_id"], {"ok": True, "document": {"document_id": "doc-1"}})
    reopened = IngestionJobStore(database)
    completed = reopened.require(job["job_id"])
    assert completed["status"] == "completed"
    assert completed["phase"] == "completed"
    assert completed["progress_percent"] == 100
    assert completed["result"]["document"]["document_id"] == "doc-1"


def test_interrupted_ingestion_is_marked_failed_after_restart(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = IngestionJobStore(database)
    job = store.create(
        file_name="report.docx",
        stored_path="data/uploads/report.docx",
        tenant_id="personal",
        owner_user_id="owner",
        knowledge_base_id="kb_finance_basic",
    )
    store.set_processing(job["job_id"])

    restarted = IngestionJobStore(database)
    interrupted = restarted.require(job["job_id"])
    assert interrupted["status"] == "failed"
    assert interrupted["phase"] == "failed"
    assert interrupted["error_code"] == "WORKER_RESTARTED"
