import type { DocumentItem } from "../api";

interface Props {
  documents: DocumentItem[];
  uploading: boolean;
  uploadMessage: string;
  raw: string;
  onUpload: (file: File) => void;
  onDelete: (doc: DocumentItem) => void;
  onClearMemory: () => void;
}

export default function KnowledgePanel({
  documents,
  uploading,
  uploadMessage,
  raw,
  onUpload,
  onDelete,
  onClearMemory,
}: Props) {
  return (
    <aside className="side card">
      <h2>知识库</h2>
      <label className="upload">
        <input
          type="file"
          accept=".txt,.md,.markdown,.json,.jsonl,.csv,.pdf,.docx,.png,.jpg,.jpeg"
          disabled={uploading}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
        上传文档
      </label>
      {uploadMessage && <p className="hint">{uploadMessage}</p>}
      <ul className="docs">
        {documents.map((doc) => (
          <li key={doc.document_id}>
            <span className="doc-name">
              {doc.title || doc.file_name || doc.document_id}
            </span>
            <span className="doc-meta">
              {doc.index_status === "degraded" ||
              doc.status === "index_degraded"
                ? "索引异常"
                : doc.status === "active"
                  ? "可检索"
                  : doc.status || "未知"}{" "}
              · {doc.total_chunks ?? 0} 块 ·{" "}
              {doc.ingested_at ? doc.ingested_at.slice(0, 10) : ""}
            </span>
            <button className="link" onClick={() => onDelete(doc)}>
              删除
            </button>
          </li>
        ))}
        {documents.length === 0 && <li className="empty">暂无文档</li>}
      </ul>
      <details>
        <summary>最近一次原始响应</summary>
        <pre>{raw || "暂无"}</pre>
      </details>
      <p className="hint" style={{ marginTop: 12 }}>
        <button className="link" onClick={onClearMemory}>
          清除长期记忆
        </button>
        （跨会话共享，删除会话不影响长期记忆）
      </p>
    </aside>
  );
}
