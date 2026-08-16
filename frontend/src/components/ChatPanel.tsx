import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  AgentEvent,
  Citation,
  DocumentItem,
  DocumentScope,
  Message,
} from "../api";

const GREETING =
  "你好，我是基于 Qwen3-14B SFT 的金融助手。可以问我金融概念、家庭财务计算，也可以上传文档后基于知识库提问。";

interface Props {
  messages: Message[];
  busy: boolean;
  input: string;
  enableRag: boolean;
  ragMode: "off" | "auto" | "required";
  synthesisProvider: "qwen" | "deepseek";
  documentScope: DocumentScope;
  documents: DocumentItem[];
  uploadedDoc: {
    document_id: string;
    title?: string | null;
    file_name?: string | null;
  } | null;
  agentEvents: AgentEvent[];
  onInputChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  onEnableRagChange: (value: boolean) => void;
  onRagModeChange: (value: "off" | "auto" | "required") => void;
  onSynthesisProviderChange: (value: "qwen" | "deepseek") => void;
  onDocumentChange: (value: string) => void;
  onDocumentScopeModeChange: (
    value: "missing" | "none" | "all_uploaded" | "selected",
  ) => void;
  onSwitchToUploaded: () => void;
  onRecover?: (action?: string) => void;
}

export default function ChatPanel({
  messages,
  busy,
  input,
  enableRag,
  ragMode,
  synthesisProvider,
  documentScope,
  documents,
  uploadedDoc,
  agentEvents,
  onInputChange,
  onSubmit,
  onStop,
  onEnableRagChange,
  onRagModeChange,
  onSynthesisProviderChange,
  onDocumentChange,
  onDocumentScopeModeChange,
  onSwitchToUploaded,
  onRecover,
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
            {message.errorAction && (
              <button
                className="recover-button"
                onClick={() => onRecover?.(message.errorAction)}
              >
                {message.errorAction === "upload_document"
                  ? "上传文档"
                  : "选择文档"}
              </button>
            )}
            {message.citations && message.citations.length > 0 && (
              <CitationList
                citations={message.citations}
                evidenceSupportLevel={message.evidenceSupportLevel}
              />
            )}
            {message.tools && message.tools.length > 0 && (
              <details className="run-details">
                <summary>工具执行 · {message.tools.length} 次</summary>
                {message.tools.map((tool, toolIndex) => (
                  <div className="run-detail-row" key={`${tool.tool_name}-${toolIndex}`}>
                    <code>{tool.tool_name || "tool"}</code>
                    <span>{tool.status || "completed"}</span>
                    <span>{tool.duration_ms ?? 0} ms</span>
                  </div>
                ))}
              </details>
            )}
            {message.trace && message.trace.length > 0 && (
              <details className="run-details">
                <summary>执行轨迹 · {message.trace.length} 个节点</summary>
                {message.trace.map((item, traceIndex) => (
                  <div className="run-detail-row" key={`${item.node}-${traceIndex}`}>
                    <code>{item.node}</code>
                    <span>{item.status}</span>
                    <span>{item.elapsed_ms} ms</span>
                  </div>
                ))}
              </details>
            )}
            {message.meta && <div className="meta">{message.meta}</div>}
          </div>
        ))}
        {busy && (
          <div className="execution-panel">
            <strong>Agent 正在执行</strong>
            {(agentEvents.length ? agentEvents : [{ event: "accepted", node: "request_boundary" }]).map(
              (event, index) => (
                <div className="execution-step" key={`${event.event}-${index}`}>
                  <span className="step-dot" />
                  <span>{eventLabel(event)}</span>
                </div>
              ),
            )}
          </div>
        )}
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
          value={documentScope.mode}
          onChange={(e) =>
            onDocumentScopeModeChange(
              e.target.value as
                | "missing"
                | "none"
                | "all_uploaded"
                | "selected",
            )
          }
        >
          <option value="missing">不指定文档</option>
          <option value="selected">指定单个文档</option>
          <option value="all_uploaded">全部上传文档</option>
          <option value="none">不使用文档</option>
        </select>
        {documentScope.mode === "selected" && (
          <select
            value={documentScope.documentId}
            onChange={(e) => onDocumentChange(e.target.value)}
          >
            {documents.map((doc) => (
              <option key={doc.document_id} value={doc.document_id}>
                {doc.title || doc.file_name || doc.document_id}
              </option>
            ))}
          </select>
        )}
        <div className="scope-badge">
          {documentScope.mode === "all_uploaded"
            ? `当前回答范围：全部上传文档（${documents.length}）`
            : documentScope.mode === "none"
              ? "当前回答范围：不使用文档"
              : documentScope.mode === "selected"
                ? `当前回答范围：${
                    documents.find(
                      (d) => d.document_id === documentScope.documentId,
                    )?.title ||
                    documents.find(
                      (d) => d.document_id === documentScope.documentId,
                    )?.file_name ||
                    documentScope.documentId
                  }`
                : "当前回答范围：自动判断"}
        </div>
        {uploadedDoc && (
          <div className="scope-badge uploaded-badge">
            《{uploadedDoc.title || uploadedDoc.file_name}》上传成功
            <button className="link" onClick={onSwitchToUploaded}>
              切换到该文档
            </button>
          </div>
        )}
        <textarea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === "Enter") onSubmit();
          }}
          placeholder="输入金融问题，Ctrl+Enter 发送"
        />
        {busy ? (
          <button className="stop-button" onClick={onStop}>停止生成</button>
        ) : (
          <button onClick={onSubmit} disabled={!input.trim()}>发送</button>
        )}
      </div>
    </div>
  );
}

function eventLabel(event: AgentEvent): string {
  const labels: Record<string, string> = {
    request_boundary: "已接收并检查请求",
    transport: "连接中断，正在自动重试…",
    rag_subgraph: "正在检索、重排并检查文档证据",
    intent_router: "已识别任务能力与执行路径",
    planner: "正在制定下一步行动",
    plan_review: "正在审核工具计划与权限",
    tool_executor: "正在执行金融计算工具",
    observation_validator: "正在校验工具结果",
    result_validator: "正在组装可验证证据",
    answer_synthesis: "正在生成最终回答草稿",
    output_guard: "正在检查数字、引用与金融风险",
    trace_finalizer: "正在完成运行记录",
  };
  return labels[String(event.node || "")] || "正在处理请求";
}

function CitationList({
  citations,
  evidenceSupportLevel,
}: {
  citations: Citation[];
  evidenceSupportLevel?: string;
}) {
  return (
    <div className="citations">
      {evidenceSupportLevel && (
        <span className="citation citation-level">
          文档证据：{evidenceSupportLevel}
        </span>
      )}
      <span className="label">引用来源</span>
      {citations.map((citation) => (
        <span key={citation.citation_id} className="citation">
          [{citation.citation_id}] {citation.file_name}
          {citation.page_start ? ` · 第 ${citation.page_start} 页` : ""}
        </span>
      ))}
    </div>
  );
}
