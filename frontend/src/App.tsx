import { useState, useEffect } from "react";
import {
  chat,
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
  type Conversation,
  type DocumentItem,
  type Message,
} from "./api";
import ConversationList from "./components/ConversationList";
import ChatPanel from "./components/ChatPanel";
import KnowledgePanel from "./components/KnowledgePanel";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeThreadId, setActiveThreadId] = useState("");
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
    setCurrentDocumentId("");
    setRaw("");
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
    const started = Date.now();
    try {
      const response: ChatResponse = await chat({
        user_message: text,
        ...getIdentity(),
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
      const finalMessages: Message[] = [
        ...nextMessages,
        {
          role: "assistant",
          content: answer,
          citations,
          meta: `${seconds}s · ${response.finish_reason || ""}`,
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
      const failed: Message[] = [
        ...nextMessages,
        { role: "assistant", content: `请求失败：${(error as Error).message}`, error: true },
      ];
      setMessages(failed);
      commitActive(failed);
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
            最终回答：蒸馏 Qwen3-14B SFT ｜规划/工具/记忆：DeepSeek ｜ RAG：BGE-M3 + Qdrant
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
          currentDocumentId={currentDocumentId}
          documents={documents}
          onInputChange={setInput}
          onSubmit={submit}
          onEnableRagChange={setEnableRag}
          onRagModeChange={setRagMode}
          onSynthesisProviderChange={setSynthesisProvider}
          onDocumentChange={setCurrentDocumentId}
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
