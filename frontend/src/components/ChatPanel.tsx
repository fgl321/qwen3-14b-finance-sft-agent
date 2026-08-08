import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, DocumentItem, Message } from "../api";

const GREETING =
  "你好，我是基于 Qwen3-14B SFT 的金融助手。可以问我金融概念、家庭财务计算，也可以上传文档后基于知识库提问。";

interface Props {
  messages: Message[];
  busy: boolean;
  input: string;
  enableRag: boolean;
  ragMode: "off" | "auto" | "required";
  synthesisProvider: "qwen" | "deepseek";
  currentDocumentId: string;
  documents: DocumentItem[];
  onInputChange: (value: string) => void;
  onSubmit: () => void;
  onEnableRagChange: (value: boolean) => void;
  onRagModeChange: (value: "off" | "auto" | "required") => void;
  onSynthesisProviderChange: (value: "qwen" | "deepseek") => void;
  onDocumentChange: (value: string) => void;
}

export default function ChatPanel({
  messages,
  busy,
  input,
  enableRag,
  ragMode,
  synthesisProvider,
  currentDocumentId,
  documents,
  onInputChange,
  onSubmit,
  onEnableRagChange,
  onRagModeChange,
  onSynthesisProviderChange,
  onDocumentChange,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat card">
      <div className="messages">
        {messages.length === 0 && !busy && (
          <div className="message assistant">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {GREETING}
            </ReactMarkdown>
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role}${message.error ? " error" : ""}`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
            {message.citations && message.citations.length > 0 && (
              <CitationList citations={message.citations} />
            )}
            {message.meta && <div className="meta">{message.meta}</div>}
          </div>
        ))}
        {busy && <div className="message assistant">正在分析…</div>}
        <div ref={bottomRef} />
      </div>
      <div className="controls">
        <label className="toggle">
          <input
            type="checkbox"
            checked={enableRag}
            onChange={(e) => onEnableRagChange(e.target.checked)}
          />
          RAG
        </label>
        <select
          value={ragMode}
          onChange={(e) =>
            onRagModeChange(e.target.value as "off" | "auto" | "required")
          }
          disabled={!enableRag}
        >
          <option value="auto">自动</option>
          <option value="required">必须知识库</option>
          <option value="off">关闭</option>
        </select>
        <select
          value={synthesisProvider}
          onChange={(e) =>
            onSynthesisProviderChange(e.target.value as "qwen" | "deepseek")
          }
        >
          <option value="qwen">蒸馏 Qwen3-14B</option>
          <option value="deepseek">DeepSeek API</option>
        </select>
        <select
          value={currentDocumentId}
          onChange={(e) => onDocumentChange(e.target.value)}
        >
          <option value="">全部文档</option>
          {documents.map((doc) => (
            <option key={doc.document_id} value={doc.document_id}>
              {doc.file_name || doc.document_id}
            </option>
          ))}
        </select>
        <textarea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === "Enter") onSubmit();
          }}
          placeholder="输入金融问题，Ctrl+Enter 发送"
        />
        <button onClick={onSubmit} disabled={busy || !input.trim()}>
          发送
        </button>
      </div>
    </div>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  return (
    <div className="citations">
      <span className="label">引用来源</span>
      {citations.map((citation) => (
        <span key={citation.citation_id} className="citation">
          [{citation.citation_id}] {citation.file_name}
          {citation.page_start ? ` · 第 ${citation.page_start} 页` : ""}
          <em>{citation.score.toFixed(1)} 分</em>
        </span>
      ))}
    </div>
  );
}
