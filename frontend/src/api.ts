export interface Citation {
  citation_id: number;
  document_id: string;
  file_name: string;
  page_start?: number | null;
  page_end?: number | null;
  score: number;
  score_display?: string | null;
  scores?: {
    dense_score?: number | null;
    sparse_score?: number | null;
    fused_score_raw?: number | null;
    retrieval_score?: number | null;
    rerank_score?: number | null;
    evidence_confidence?: number | null;
    display_score?: number | null;
    display_score_source?: "retrieval" | "reranker" | "evidence";
  };
}

export interface RagPayload {
  answer?: string;
  evidence_assessment?: {
    sufficient?: boolean;
    reason?: string;
    support_level?:
      | "direct_support"
      | "partial_support"
      | "background_support"
      | "irrelevant";
  };
  citations?: Citation[];
  retrieved_chunks?: Array<Record<string, unknown>>;
  retrieved_count?: number;
}

export interface ChatResponse {
  final_answer?: string;
  answer?: string;
  finish_reason?: string;
  rag?: RagPayload;
  synthesis_result?: {
    used_citation_ids?: Array<string | number>;
  };
  final_response_result?: {
    synthesis?: {
      used_citation_ids?: Array<string | number>;
    };
  };
  error?: { code?: string; message?: string };
  detail?: unknown;
  personal_memory?: Record<string, unknown>;
  synthesis_llm_provider?: "qwen" | "deepseek";
  node_trace?: NodeTrace[];
  runtime_revision?: string;
  effective_rag?: {
    enabled?: boolean;
    mode?: "off" | "auto" | "required";
    retrieval_requirement?: string;
    citation_requirement?: string;
  };
  planner_invocation_count?: number;
  plan_attempt_in_round?: number;
  target_execution_round?: number;
  execution_round?: number;
  replan_count?: number;
  execution_round_history?: Array<Record<string, unknown>>;
  required_capabilities?: string[];
  completed_capabilities?: string[];
  failed_capabilities?: string[];
  overall_status?: "completed" | "partial" | "failed";
  agent_loop_result?: {
    tool_traces?: Array<{ tool_name?: string; status?: string; duration_ms?: number }>;
  };
}

export interface NodeTrace {
  node: string;
  status: string;
  elapsed_ms: number;
  summary?: Record<string, unknown>;
}

export interface AgentEvent {
  event: string;
  request_id?: string;
  node?: string | null;
  status?: string;
  detail?: Record<string, unknown>;
  error?: AgentErrorEnvelope;
  result?: ChatResponse;
}

export interface AgentErrorEnvelope {
  code?: string;
  message?: string;
  category?: string;
  retryable?: boolean;
  request_id?: string;
  run_id?: string;
  error_id?: string;
  details?: {
    reason_codes?: string[];
    action?: string;
    user_action_required?: boolean;
  };
}

export class AgentError extends Error {
  code?: string;
  action?: string;
  error_id?: string;

  constructor(message: string, envelope?: AgentErrorEnvelope) {
    super(message);
    this.name = "AgentError";
    this.code = envelope?.code;
    this.action = envelope?.details?.action;
    this.error_id = envelope?.error_id;
  }
}

export class StreamEndedWithoutResultError extends Error {
  constructor() {
    super("流式响应结束但没有最终结果。");
    this.name = "StreamEndedWithoutResultError";
  }
}

function readEnvelope(value: unknown): AgentErrorEnvelope | undefined {
  if (!value || typeof value !== "object") return undefined;
  return value as AgentErrorEnvelope;
}

function formatEnvelope(envelope?: AgentErrorEnvelope): string | undefined {
  if (!envelope) return undefined;
  if (envelope.message) return envelope.message;
  const reason = envelope.details?.reason_codes?.[0] || envelope.code;
  if (reason) {
    return envelope.details?.action
      ? `${reason}（${envelope.details.action}）`
      : reason;
  }
  return undefined;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  tools?: Array<{ tool_name?: string; status?: string; duration_ms?: number }>;
  trace?: NodeTrace[];
  meta?: string;
  error?: boolean;
  errorAction?: string;
  evidenceSupportLevel?: string;
}

export interface Conversation {
  thread_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: Message[];
}

export interface ChatRequest {
  user_message: string;
  user_id: string;
  thread_id: string;
  tenant_id: string;
  knowledge_base_id: string;
  synthesis_llm_provider: "qwen" | "deepseek";
  document_ids: string[];
  document_scope?: {
    mode: "selected" | "none" | "all_uploaded";
    document_ids?: string[];
  };
  use_short_memory: boolean;
  use_long_memory: boolean;
  save_memory: boolean;
  extract_long_memory: boolean;
  enable_rag: boolean;
  rag_mode: "off" | "auto" | "required";
}

export type DocumentScope =
  | { mode: "missing" }
  | { mode: "selected"; documentId: string }
  | { mode: "all_uploaded" }
  | { mode: "none" };

export interface DocumentItem {
  document_id: string;
  file_name?: string | null;
  title?: string | null;
  aliases?: string[];
  status?: string | null;
  index_status?: string | null;
  file_sha256?: string | null;
  total_chunks?: number;
  parent_count?: number;
  child_count?: number;
  ingested_at?: string | null;
}

export interface UploadResponse {
  ok: boolean;
  document?: {
    document_id?: string;
    file_name?: string;
    title?: string | null;
    aliases?: string[];
  };
  chunks?: { total_chunks?: number; parent_count?: number; child_count?: number };
  detail?: unknown;
}

const IDENTITY = {
  tenant_id: "personal",
  user_id: "owner",
  knowledge_base_id: "kb_finance_basic",
};

const CONVERSATIONS_KEY = "finance_conversations";

function uidKey(key: string): string {
  let value = localStorage.getItem(key);
  if (!value) {
    value = `${key}-${uuid()}`;
    localStorage.setItem(key, value);
  }
  return value;
}

// crypto.randomUUID 只在 https/localhost 等安全上下文可用；
// 内网 http 部署时用 Math.random 兜底，避免白屏。
function uuid(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function getIdentity() {
  return {
    ...IDENTITY,
    thread_id: uidKey("finance_tid"),
  };
}

export interface IngestionJob {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  phase?: string;
  progress_percent?: number;
  progress_message?: string;
  file_name: string;
  result?: UploadResponse | null;
  error_code?: string | null;
  error_message?: string | null;
}

// 新建对话：重新生成 thread_id（短期记忆按 thread 隔离），
// 用户身份由服务端固定；这里只重新生成 thread_id 隔离短期记忆。
export function resetThread() {
  localStorage.removeItem("finance_tid");
  return getIdentity();
}

export function listConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveConversations(list: Conversation[]): void {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(list));
}

export async function deleteChatHistory(threadId: string): Promise<void> {
  const identity = getIdentity();
  const params = new URLSearchParams({
    user_id: identity.user_id,
    tenant_id: identity.tenant_id,
  });
  await fetch(
    `/api/chat/history/${encodeURIComponent(threadId)}?${params.toString()}`,
    { method: "DELETE" },
  );
}

export async function clearLongTermMemory(): Promise<number> {
  const identity = getIdentity();
  const params = new URLSearchParams({
    user_id: identity.user_id,
    tenant_id: identity.tenant_id,
  });
  const response = await fetch(`/api/memory/facts?${params.toString()}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = (await response.json()) as { deleted_count?: number };
  return data.deleted_count ?? 0;
}

export async function chat(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/api/chat/graph-v2", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = (await response.json()) as ChatResponse;
  if (!response.ok) {
    const detail = readEnvelope(data.detail);
    throw new AgentError(
      formatEnvelope(detail) || data.error?.message || `HTTP ${response.status}`,
      detail,
    );
  }
  return data;
}

async function chatStreamOnce(
  payload: ChatRequest,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch("/api/chat/graph-v2/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) {
    let message = `HTTP ${response.status}`;
    let envelope: AgentErrorEnvelope | undefined;
    try {
      const data = (await response.json()) as { detail?: unknown };
      envelope = readEnvelope(data.detail);
      message = formatEnvelope(envelope) || message;
    } catch {
      // Non-JSON error body: keep the HTTP status fallback.
    }
    throw new AgentError(message, envelope);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed: ChatResponse | undefined;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const dataLine = frame
        .split(/\r?\n/)
        .find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      const event = JSON.parse(dataLine.slice(5).trim()) as AgentEvent;
      onEvent(event);
      if (event.event === "error") {
        const envelope = readEnvelope(event.error || event.detail);
        const message = formatEnvelope(envelope) || "请求失败";
        throw new AgentError(message, envelope);
      }
      if (event.event === "completed" && event.result) completed = event.result;
    }
    if (done) break;
  }
  if (!completed) throw new StreamEndedWithoutResultError();
  return completed;
}

export async function chatStream(
  payload: ChatRequest,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
  retries = 1,
): Promise<ChatResponse> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    if (attempt > 0) {
      onEvent({
        event: "transport_retry",
        node: "transport",
        status: "running",
        detail: { attempt },
      });
    }
    try {
      return await chatStreamOnce(payload, onEvent, signal);
    } catch (error) {
      lastError = error;
      if (
        signal?.aborted ||
        attempt >= retries ||
        !(error instanceof StreamEndedWithoutResultError)
      ) {
        throw error;
      }
    }
  }
  throw lastError;
}

export async function uploadDocument(
  file: File,
  onStatus?: (job: IngestionJob) => void,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("knowledge_base_id", IDENTITY.knowledge_base_id);
  form.append("visibility", "private");
  const response = await fetch("/api/knowledge/documents/async", {
    method: "POST",
    body: form,
  });
  const data = (await response.json()) as IngestionJob & { detail?: unknown };
  if (!response.ok) {
    const detail = data.detail as { message?: string } | undefined;
    throw new Error(detail?.message || `HTTP ${response.status}`);
  }
  let job = data;
  onStatus?.(job);
  for (let attempt = 0; attempt < 600; attempt += 1) {
    if (job.status === "completed" && job.result) return job.result;
    if (job.status === "failed") {
      throw new Error(job.error_message || job.error_code || "文档索引失败");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    const poll = await fetch(
      `/api/knowledge/jobs/${encodeURIComponent(job.job_id)}`,
    );
    if (!poll.ok) throw new Error(`任务查询失败：HTTP ${poll.status}`);
    job = (await poll.json()) as IngestionJob;
    onStatus?.(job);
  }
  throw new Error("文档处理超时，请稍后在知识库中检查。");
}

export async function listDocuments(): Promise<DocumentItem[]> {
  const params = new URLSearchParams({
    tenant_id: IDENTITY.tenant_id,
    owner_user_id: getIdentity().user_id,
    knowledge_base_id: IDENTITY.knowledge_base_id,
  });
  const response = await fetch(`/api/knowledge/documents?${params.toString()}`);
  const data = (await response.json()) as { documents?: DocumentItem[] };
  return data.documents || [];
}

export async function deleteDocument(documentId: string): Promise<void> {
  const params = new URLSearchParams({
    tenant_id: IDENTITY.tenant_id,
    owner_user_id: getIdentity().user_id,
    knowledge_base_id: IDENTITY.knowledge_base_id,
  });
  await fetch(`/api/knowledge/documents/${documentId}?${params.toString()}`, {
    method: "DELETE",
  });
}

export async function health(): Promise<{ agent: boolean; qwen: boolean }> {
  const agentOk = await fetch("/health").then((r) => r.ok).catch(() => false);
  const qwenOk = await fetch("/health/qwen").then((r) => r.ok).catch(() => false);
  return { agent: agentOk, qwen: qwenOk };
}
