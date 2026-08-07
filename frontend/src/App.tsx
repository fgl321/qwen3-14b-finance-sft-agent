import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  chat,
  deleteDocument,
  getIdentity,
  health,
  listDocuments,
  uploadDocument,
  type ChatResponse,
  type Citation,
  type DocumentItem,
} from "./api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  meta?: string;
  error?: boolean;
}

export default function App() {
  const identity = getIdentity();
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "你好，我是基于 Qwen3-14B SFT 的金融助手。可以问我金融概念、家庭财务计算，也可以上传文档后基于知识库提问。",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [enableRag, setEnableRag] = useState(true);
  const [ragMode, setRagMode] = useState<"off" | "auto" | "required">("auto");
  const [synthesisProvider, setSynthesisProvider] = useState<"qwen" | "deepseek">("qwen");
  const [currentDocumentId, setCurrentDocumentId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [status, setStatus] = useState("正在检查服务…");
  const [raw, setRaw] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    health().then((h) => {
      if (h.agent && h.qwen) setStatus("服务正常 · Qwen 已连接");
      else if (h.agent) setStatus("Agent 正常 · Qwen 未连接");
      else setStatus("服务未连接");
    });
    refreshDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function refreshDocuments() {
    try {
      setDocuments(await listDocuments());
    } catch {
      setDocuments([]);
    }
  }

  async function submit() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setBusy(true);
    const started = Date.now();
    try {
      const response: ChatResponse = await chat({
        user_message: text,
        ...identity,
        synthesis_llm_provider: synthesisProvider,
        document_ids: currentDocumentId ? [currentDocumentId] : [],
        use_short_memory: true,
        use_long_memory: true,
        save_memory: true,
        extract_long_memory: true,
        enable_rag: enableRag,
        rag_mode: ragMode,
      });
      setRaw(JSON.stringify(response, null, 2));
      const answer = response.final_answer || response.answer || "请求完成，但未找到回答字段。";
      const citations = response.rag?.citations || [];
      const seconds = ((Date.now() - started) / 1000).toFixed(1);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          citations,
          meta: `${seconds}s · ${response.finish_reason || ""}`,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `请求失败：${(error as Error).message}`, error: true },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadMessage(`正在上传并索引 ${file.name}…`);
    try {
      const result = await uploadDocument(file);
      if (!result.ok) throw new Error("上传返回失败");
      const chunks = result.chunks || {};
      const docId = result.document?.document_id;
      if (docId) setCurrentDocumentId(docId);
      setUploadMessage(
        `已入库 ${result.document?.file_name || file.name}（父块 ${chunks.parent_count ?? 0}，子块 ${chunks.child_count ?? 0}），已切换到该文档`,
      );
      await refreshDocuments();
    } catch (error) {
      setUploadMessage(`上传失败：${(error as Error).message}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(doc: DocumentItem) {
    if (!window.confirm(`删除文档「${doc.file_name || doc.document_id}」？删除后不再参与检索。`)) return;
    try {
      await deleteDocument(doc.document_id);
      await refreshDocuments();
    } catch {
      window.alert("删除失败");
    }
  }

  return (
    <main>
      <header>
        <div>
          <h1>Qwen3-14B 金融 Agent</h1>
          <p className="sub">
            最终回答：蒸馏 Qwen3-14B SFT ｜规划/工具/记忆：DeepSeek ｜ RAG：BGE-M3 + Qdrant
          </p>
        </div>
        <div className="status" title={status}>{status}</div>
      </header>

      <section className="layout">
        <div className="chat card">
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}${message.error ? " error" : ""}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
                {message.citations && message.citations.length > 0 && (
                  <div className="citations">
                    <span className="label">引用来源</span>
                    {message.citations.map((citation) => (
                      <span key={citation.citation_id} className="citation">
                        [{citation.citation_id}] {citation.file_name}
                        {citation.page_start ? ` · 第 ${citation.page_start} 页` : ""}
                        <em>{citation.score.toFixed(1)} 分</em>
                      </span>
                    ))}
                  </div>
                )}
                {message.meta && <div className="meta">{message.meta}</div>}
              </div>
            ))}
            {busy && <div className="message assistant">正在分析…</div>}
            <div ref={bottomRef} />
          </div>
          <div className="controls">
            <label className="toggle">
              <input type="checkbox" checked={enableRag} onChange={(e) => setEnableRag(e.target.checked)} />
              RAG
            </label>
            <select value={ragMode} onChange={(e) => setRagMode(e.target.value as "off" | "auto" | "required")} disabled={!enableRag}>
              <option value="auto">自动</option>
              <option value="required">必须知识库</option>
              <option value="off">关闭</option>
            </select>
            <select value={synthesisProvider} onChange={(e) => setSynthesisProvider(e.target.value as "qwen" | "deepseek")}>
              <option value="qwen">蒸馏 Qwen3-14B</option>
              <option value="deepseek">DeepSeek API</option>
            </select>
            <select value={currentDocumentId} onChange={(e) => setCurrentDocumentId(e.target.value)}>
              <option value="">全部文档</option>
              {documents.map((doc) => (
                <option key={doc.document_id} value={doc.document_id}>
                  {doc.file_name || doc.document_id}
                </option>
              ))}
            </select>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.ctrlKey && e.key === "Enter") submit();
              }}
              placeholder="输入金融问题，Ctrl+Enter 发送"
            />
            <button onClick={submit} disabled={busy || !input.trim()}>发送</button>
          </div>
        </div>

        <aside className="side card">
          <h2>知识库</h2>
          <label className="upload">
            <input
              type="file"
              accept=".txt,.md,.markdown,.json,.jsonl,.csv,.pdf,.docx"
              disabled={uploading}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleUpload(file);
                e.target.value = "";
              }}
            />
            上传文档
          </label>
          {uploadMessage && <p className="hint">{uploadMessage}</p>}
          <ul className="docs">
            {documents.map((doc) => (
              <li key={doc.document_id}>
                <span className="doc-name">{doc.file_name || doc.document_id}</span>
                <span className="doc-meta">
                  {doc.total_chunks ?? 0} 块 · {doc.ingested_at ? doc.ingested_at.slice(0, 10) : ""}
                </span>
                <button className="link" onClick={() => handleDelete(doc)}>删除</button>
              </li>
            ))}
            {documents.length === 0 && <li className="empty">暂无文档</li>}
          </ul>
          <details>
            <summary>最近一次原始响应</summary>
            <pre>{raw || "暂无"}</pre>
          </details>
        </aside>
      </section>
    </main>
  );
}
