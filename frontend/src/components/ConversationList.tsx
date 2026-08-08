import type { Conversation } from "../api";

interface Props {
  conversations: Conversation[];
  activeThreadId: string;
  onCreate: () => void;
  onSwitch: (threadId: string) => void;
  onDelete: (threadId: string) => void;
}

export default function ConversationList({
  conversations,
  activeThreadId,
  onCreate,
  onSwitch,
  onDelete,
}: Props) {
  return (
    <aside className="conv card">
      <div className="conv-head">
        <h2>会话</h2>
        <button
          className="link"
          onClick={onCreate}
          title="开启新的短期会话，长期记忆保留"
        >
          ＋ 新建对话
        </button>
      </div>
      <ul className="conv-list">
        {conversations.map((conv) => (
          <li
            key={conv.thread_id}
            className={`conv-item${conv.thread_id === activeThreadId ? " active" : ""}`}
            onClick={() => onSwitch(conv.thread_id)}
            title={conv.title}
          >
            <span className="conv-title">{conv.title}</span>
            <span className="conv-meta">
              {new Date(conv.updated_at).toLocaleString("zh-CN", {
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
            <button
              className="link conv-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(conv.thread_id);
              }}
            >
              删除
            </button>
          </li>
        ))}
        {conversations.length === 0 && <li className="empty">暂无会话</li>}
      </ul>
    </aside>
  );
}
