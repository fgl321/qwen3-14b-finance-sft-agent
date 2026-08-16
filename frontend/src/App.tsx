import { useEffect, useRef, useState } from "react";
import {
  AgentError,
  chatStream,
  clearLongTermMemory,
  deleteChatHistory,
  deleteDocument,
  getIdentity,
  health,
  listDocuments,
  listConversations,
  resetThread,
  saveConversations,
  uploadDocument,
  type ChatResponse,
  type AgentEvent,
  type Conversation,
  type DocumentItem,
  type DocumentScope,
  type Message,
} from "./api";
import ConversationList from "./components/ConversationList";
import ChatPanel from "./components/ChatPanel";
import KnowledgePanel from "./components/KnowledgePanel";

function evidenceSupportLabel(value?: string): string | undefined {
  const labels: Record<string, string> = {
    direct_support: "直接支持",
    partial_support: "部分支持",
    background_support: "背景支持",
    irrelevant: "无直接支持",
  };
  return value ? labels[value] : undefined;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeThreadId, setActiveThreadId] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [enableRag, setEnableRag] = useState(true);
  const [ragMode, setRagMode] = useState<"off" | "auto" | "required">("auto");
  const [synthesisProvider, setSynthesisProvider] = useState<"qwen" | "deepseek">("qwen");
  const [documentScope, setDocumentScope] = useState<DocumentScope>({
    mode: "missing",
  });
  const [uploadedDoc, setUploadedDoc] = useState<{
    document_id: string;
    title?: string | null;
    file_name?: string | null;
  } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [status, setStatus] = useState("正在检查服务…");
  const [raw, setRaw] = useState("");
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([]);
  const activeRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    health().then((h) => {
      if (h.agent && h.qwen) setStatus("服务正常 · Qwen 已连接");
      else if (h.agent) setStatus("Agent 正常 · Qwen 未连接");
      else setStatus("服务未连接");
    });
    refreshDocuments();
    initConversations();
  }, []);

  async function refreshDocuments() {
    try {
      setDocuments(await listDocuments());
    } catch {
      setDocuments([]);
    }
  }

  function initConversations() {
    const stored = listConversations();
    const identity = getIdentity();
    if (stored.length === 0) {
      const first: Conversation = {
        thread_id: identity.thread_id,
        title: "新对话",
        created_at: Date.now(),
        updated_at: Date.now(),
        messages: [],
      };
      saveConversations([first]);
      setConversations([first]);
      setActiveThreadId(identity.thread_id);
      setMessages([]);
      return;
    }
    const found = stored.find((c) => c.thread_id === identity.thread_id);
    if (found) {
      setConversations(stored);
      setActiveThreadId(found.thread_id);
      setMessages(found.messages);
      return;
    }
    const sorted = [...stored].sort((a, b) => b.updated_at - a.updated_at);
    const latest = sorted[0];
    localStorage.setItem("finance_tid", latest.thread_id);
    setConversations(stored);
    setActiveThreadId(latest.thread_id);
    setMessages(latest.messages);
  }

  function commitActive(messages: Message[], title?: string) {
    const next = conversations.map((conv) =>
      conv.thread_id === activeThreadId
        ? { ...conv, title: title ?? conv.title, updated_at: Date.now(), messages }
        : conv,
    );
    setConversations(next);
    saveConversations(next);
  }

  function switchConversation(threadId: string) {
    if (busy || threadId === activeThreadId) return;
    commitActive(messages);
    const target = conversations.find((c) => c.thread_id === threadId);
    localStorage.setItem("finance_tid", threadId);
    setActiveThreadId(threadId);
    setMessages(target?.messages || []);
    setRaw("");
    setAgentEvents([]);
    setInput("");
  }

  function createConversation() {
    if (busy) return;
    commitActive(messages);
    const identity = resetThread();
    const next: Conversation = {
      thread_id: identity.thread_id,
      title: "新对话",
      created_at: Date.now(),
      updated_at: Date.now(),
      messages: [],
    };
    const nextList = [next, ...conversations];
    setConversations(nextList);
    saveConversations(nextList);
    setActiveThreadId(identity.thread_id);
    setMessages([]);
    setRaw("");
    setAgentEvents([]);
    setInput("");
  }

  async function deleteConversation(threadId: string) {
    if (busy) return;
    const target = conversations.find((c) => c.thread_id === threadId);
    if (!window.confirm(`删除会话「${target?.title || "新对话"}」？会同时清空服务器上的该会话短期记忆。`)) return;
    try {
      await deleteChatHistory(threadId);
    } catch {
      // 服务端清理失败不阻塞本地删除。
    }
    const remaining = conversations.filter((c) => c.thread_id !== threadId);
    if (remaining.length === 0) {
      const identity = resetThread();
      const fresh: Conversation = {
        thread_id: identity.thread_id,
        title: "新对话",
        created_at: Date.now(),
        updated_at: Date.now(),
        messages: [],
      };
      saveConversations([fresh]);
      setConversations([fresh]);
      setActiveThreadId(identity.thread_id);
      setMessages([]);
    } else if (threadId === activeThreadId) {
      const fallback = remaining[0];
      localStorage.setItem("finance_tid", fallback.thread_id);
      setConversations(remaining);
      saveConversations(remaining);
      setActiveThreadId(fallback.thread_id);
      setMessages(fallback.messages);
    } else {
      setConversations(remaining);
      saveConversations(remaining);
    }
    setRaw("");
  }

  async function submit() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    const nextMessages: Message[] = [
      ...messages,
      { role: "user", content: text },
    ];
    setMessages(nextMessages);
    setBusy(true);
    setAgentEvents([]);
    const controller = new AbortController();
    activeRequest.current = controller;
    const started = Date.now();
    try {
      const document_scope =
        documentScope.mode === "none"
          ? { mode: "none" as const, document_ids: [] }
          : documentScope.mode === "all_uploaded"
            ? { mode: "all_uploaded" as const, document_ids: [] }
            : documentScope.mode === "selected"
              ? {
                  mode: "selected" as const,
                  document_ids: [documentScope.documentId],
                }
              : undefined;
      const response: ChatResponse = await chatStream({
        user_message: text,
        ...getIdentity(),
        synthesis_llm_provider: synthesisProvider,
        document_ids: [],
        document_scope,
        use_short_memory: true,
        use_long_memory: true,
        save_memory: true,
        extract_long_memory: true,
        enable_rag: enableRag,
        rag_mode: ragMode,
      }, (event) => {
        if (event.event !== "completed") {
          setAgentEvents((current) => [...current.slice(-11), event]);
        }
      }, controller.signal);
      setRaw(JSON.stringify(response, null, 2));
      const answer = response.final_answer || response.answer || "请求完成，但未找到回答字段。";
      const usedCitationIds = new Set(
        [
          ...(response.synthesis_result?.used_citation_ids || []),
          ...(response.final_response_result?.synthesis?.used_citation_ids || []),
        ].map((id) => String(id)),
      );
      const citations =
        usedCitationIds.size > 0
          ? (response.rag?.citations || []).filter((citation) =>
              usedCitationIds.has(String(citation.citation_id)),
            )
          : response.rag?.citations || [];
      const seconds = ((Date.now() - started) / 1000).toFixed(1);
      const finalMessages: Message[] = [
        ...nextMessages,
        {
          role: "assistant",
          content: answer,
          citations,
          evidenceSupportLevel: evidenceSupportLabel(
            response.rag?.evidence_assessment?.support_level,
          ),
          tools: response.agent_loop_result?.tool_traces,
          trace: response.node_trace,
          meta: [
            `${seconds}s`,
            response.synthesis_llm_provider || synthesisProvider,
            response.overall_status || response.finish_reason || "completed",
            response.runtime_revision || "runtime-version-missing",
            `RAG:${response.effective_rag?.mode || ragMode}`,
            `执行轮:${response.execution_round ?? "?"}`,
            `规划:${response.planner_invocation_count ?? "?"}`,
          ].join(" · "),
        },
      ];
      setMessages(finalMessages);
      const current = conversations.find((c) => c.thread_id === activeThreadId);
      const title =
        current?.title && current.title !== "新对话"
          ? current.title
          : text.length > 18
            ? `${text.slice(0, 18)}…`
            : text;
      commitActive(finalMessages, title);
    } catch (error) {
      const wasAborted = (error as Error).name === "AbortError";
      const agentError = error instanceof AgentError ? error : undefined;
      const failed: Message[] = [
        ...nextMessages,
        {
          role: "assistant",
          content: wasAborted ? "本次生成已停止。" : `请求失败：${(error as Error).message}`,
          error: !wasAborted,
          errorAction: agentError?.action,
        },
      ];
      setMessages(failed);
      commitActive(failed);
    } finally {
      activeRequest.current = null;
      setBusy(false);
    }
  }

  function stopGeneration() {
    activeRequest.current?.abort();
  }

  function handleRecover(action?: string) {
    if (action === "clear_document") {
      setDocumentScope({ mode: "none" });
      return;
    }
    if (action === "upload_document" || action === "select_document") {
      const panel = document.querySelector(".side.card");
      panel?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function handleDocumentChange(value: string) {
    if (value) {
      setDocumentScope({ mode: "selected", documentId: value });
    }
  }

  function handleDocumentScopeModeChange(
    mode: "missing" | "none" | "all_uploaded" | "selected",
  ) {
    if (mode === "selected") {
      const first = documents[0];
      setDocumentScope(
        first
          ? { mode: "selected", documentId: first.document_id }
          : { mode: "missing" },
      );
      return;
    }
    setDocumentScope({ mode });
  }

  function handleSwitchToUploaded() {
    if (!uploadedDoc?.document_id) return;
    setDocumentScope({
      mode: "selected",
      documentId: uploadedDoc.document_id,
    });
    setUploadedDoc(null);
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadMessage(`正在上传并索引 ${file.name}…`);
    try {
      const result = await uploadDocument(file, (job) => {
        const label = job.progress_message || (
          job.status === "queued" ? "等待处理" : "正在解析、切块并建立索引"
        );
        const progress = Math.round(job.progress_percent ?? 0);
        setUploadMessage(`${label} ${progress}%：${file.name}`);
      });
      if (!result.ok) throw new Error("上传返回失败");
      const chunks = result.chunks || {};
      const docMeta = result.document;
      if (docMeta?.document_id) {
        setUploadedDoc({
          document_id: docMeta.document_id,
          title: docMeta.title,
          file_name: docMeta.file_name,
        });
      }
      setUploadMessage(
        `《${result.document?.title || result.document?.file_name || file.name}》上传成功（父块 ${chunks.parent_count ?? 0}，子块 ${chunks.child_count ?? 0}）`,
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

  async function handleClearMemory() {
    if (!window.confirm("清除本用户的全部长期记忆？此操作不影响各会话的短期记忆。")) return;
    try {
      const deleted = await clearLongTermMemory();
      window.alert(`已清除 ${deleted} 条长期记忆。`);
    } catch {
      window.alert("清除长期记忆失败");
    }
  }

  return (
    <main>
      <header>
        <div>
          <h1>Qwen3-14B 金融 Agent</h1>
          <p className="sub">
            最终回答可选蒸馏 Qwen3-14B / DeepSeek V4 Flash ｜ LangGraph + RAG + 确定性金融工具
          </p>
        </div>
        <div className="header-actions">
          <div className="status" title={status}>{status}</div>
        </div>
      </header>

      <section className="layout">
        <ConversationList
          conversations={conversations}
          activeThreadId={activeThreadId}
          onCreate={createConversation}
          onSwitch={switchConversation}
          onDelete={deleteConversation}
        />

        <ChatPanel
          messages={messages}
          busy={busy}
          input={input}
          enableRag={enableRag}
          ragMode={ragMode}
          synthesisProvider={synthesisProvider}
          documentScope={documentScope}
          documents={documents}
          uploadedDoc={uploadedDoc}
          agentEvents={agentEvents}
          onInputChange={setInput}
          onSubmit={submit}
          onStop={stopGeneration}
          onEnableRagChange={setEnableRag}
          onRagModeChange={setRagMode}
          onSynthesisProviderChange={setSynthesisProvider}
          onDocumentChange={handleDocumentChange}
          onDocumentScopeModeChange={handleDocumentScopeModeChange}
          onSwitchToUploaded={handleSwitchToUploaded}
          onRecover={handleRecover}
        />

        <KnowledgePanel
          documents={documents}
          uploading={uploading}
          uploadMessage={uploadMessage}
          raw={raw}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onClearMemory={handleClearMemory}
        />
      </section>
    </main>
  );
}
