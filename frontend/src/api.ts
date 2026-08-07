export interface Citation {
  citation_id: number;
  document_id: string;
  file_name: string;
  page_start?: number | null;
  page_end?: number | null;
  score: number;
}

export interface RagPayload {
  answer?: string;
  evidence_assessment?: { sufficient?: boolean; reason?: string };
  citations?: Citation[];
  retrieved_chunks?: Array<Record<string, unknown>>;
  retrieved_count?: number;
}

export interface ChatResponse {
  final_answer?: string;
  answer?: string;
  finish_reason?: string;
  rag?: RagPayload;
  error?: { code?: string; message?: string };
  detail?: unknown;
  personal_memory?: Record<string, unknown>;
}

export interface ChatRequest {
  user_message: string;
  user_id: string;
  thread_id: string;
  tenant_id: string;
  knowledge_base_id: string;
  synthesis_llm_provider: "qwen" | "deepseek";
  document_ids: string[];
  use_short_memory: boolean;
  use_long_memory: boolean;
  save_memory: boolean;
  extract_long_memory: boolean;
  enable_rag: boolean;
  rag_mode: "off" | "auto" | "required";
}

export interface DocumentItem {
  document_id: string;
  file_name?: string | null;
  file_sha256?: string | null;
  total_chunks?: number;
  parent_count?: number;
  child_count?: number;
  ingested_at?: string | null;
}

export interface UploadResponse {
  ok: boolean;
  document?: { document_id?: string; file_name?: string };
  chunks?: { total_chunks?: number; parent_count?: number; child_count?: number };
  detail?: unknown;
}

const IDENTITY = {
  tenant_id: "default",
  knowledge_base_id: "kb_finance_basic",
};

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
    user_id: uidKey("finance_uid"),
    thread_id: uidKey("finance_tid"),
  };
}

export async function chat(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/api/chat/graph-v2", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = (await response.json()) as ChatResponse;
  if (!response.ok) {
    const detail = data.detail as { message?: string } | undefined;
    throw new Error(detail?.message || data.error?.message || `HTTP ${response.status}`);
  }
  return data;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("tenant_id", IDENTITY.tenant_id);
  // 上传必须使用与聊天相同的用户 ID，否则租户隔离过滤会检索不到文档。
  form.append("owner_user_id", getIdentity().user_id);
  form.append("knowledge_base_id", IDENTITY.knowledge_base_id);
  form.append("visibility", "private");
  const response = await fetch("/api/knowledge/documents", {
    method: "POST",
    body: form,
  });
  const data = (await response.json()) as UploadResponse;
  if (!response.ok) {
    const detail = data.detail as { message?: string } | undefined;
    throw new Error(detail?.message || `HTTP ${response.status}`);
  }
  return data;
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
