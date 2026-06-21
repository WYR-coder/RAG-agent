"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Loader2, FileText, Search, X, Settings, Upload, BookOpen, Key, Check, AlertTriangle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

interface Source {
  source_file: string;
  section_path: string;
  content_type: string;
  excerpt: string;
  relevance_score: number;
  category: string;
  relevance?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  retrievalTimeMs?: number;
  generationTimeMs?: number;
}

interface ConfigStatus {
  llm_configured: boolean;
  zhipu_configured: boolean;
  volc_configured: boolean;
  bge_m3_downloaded: boolean;
  reranker_downloaded: boolean;
}

interface AuditCheck {
  name: string;
  passed: boolean;
  detail: string;
  level: string;
}

interface PipelineStatus {
  stage: string;
  progress: number;
  message: string;
  audit?: {
    passed: boolean;
    checks: AuditCheck[];
    error?: string;
  };
}

const STEPS = [
  { key: "llm_configured", label: "配置 API Key", icon: Key },
  { key: "bge_m3_downloaded", label: "下载模型", icon: BookOpen },
  { key: "has_terms", label: "上传领域词典", icon: BookOpen },
  { key: "has_docs", label: "上传文档", icon: Upload },
  { key: "has_index", label: "构建知识库", icon: Search },
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chunkTotal, setChunkTotal] = useState<number | null>(null);
  const [configStatus, setConfigStatus] = useState<ConfigStatus | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"keys" | "dictionary" | "documents">("keys");
  const [termsDict, setTermsDict] = useState("");
  const [synonyms, setSynonyms] = useState("");
  const [desensitize, setDesensitize] = useState("");
  const [documents, setDocuments] = useState<{filename: string; stored_as: string; size: number}[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesRef = useRef<Message[]>([]);
  messagesRef.current = messages;

  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Load config and health status on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(r => r.json())
      .then(d => { if (d.milvus?.rows) setChunkTotal(d.milvus.rows); })
      .catch(() => {});
    fetch(`${API_BASE}/api/config/status`)
      .then(r => r.json())
      .then(d => setConfigStatus(d))
      .catch(() => {});
    fetch(`${API_BASE}/api/dictionary`)
      .then(r => r.json())
      .then(d => { setTermsDict(d["terms_dict.txt"] || ""); setSynonyms(d["query_synonyms.txt"] || ""); setDesensitize(d["desensitize_rules.json"] || ""); })
      .catch(() => {});
    fetch(`${API_BASE}/api/documents`)
      .then(r => r.json())
      .then(d => setDocuments(d.documents || []))
      .catch(() => {});
    fetch(`${API_BASE}/api/pipeline/status`)
      .then(r => r.json())
      .then(d => setPipelineStatus(d))
      .catch(() => {});
  }, []);

  const setupComplete = configStatus && (
    configStatus.llm_configured && configStatus.bge_m3_downloaded &&
    termsDict.trim() && documents.length > 0 && chunkTotal != null && chunkTotal > 0
  );

  const handleSend = useCallback(async () => {
    const query = inputRef.current?.value || input;
    if (!query || loading) return;
    setInput("");
    setLoading(true);

    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 3, stream: true, history: messagesRef.current.slice(-10).map(m => ({ role: m.role, content: m.content })) }),
      });

      if (!res.ok) throw new Error(`服务器错误 (${res.status})`);
      const reader = res.body?.getReader();
      if (!reader) throw new Error("服务器无响应");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.error) {
              setMessages((prev) => prev.map((m, i) => i === prev.length - 1 ? { ...m, content: m.content + `\n\n> ⚠️ ${data.error}` } : m));
              continue;
            }
            if (data.token) {
              setMessages((prev) => prev.map((m, i) => i === prev.length - 1 ? { ...m, content: m.content + data.token } : m));
            }
            if (data.done) {
              setMessages((prev) => prev.map((m, i) => i === prev.length - 1
                ? { ...m, content: data.corrected_content || m.content, sources: data.sources, retrievalTimeMs: data.retrieval_time_ms, generationTimeMs: data.generation_time_ms }
                : m));
            }
          } catch { /* skip */ }
        }
      }
    } catch (e) {
      setMessages((prev) => { const next = [...prev]; const last = next[next.length - 1]; if (last) last.content = `请求失败: ${e instanceof Error ? e.message : "未知错误"}`; return next; });
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); handleSend(); }
  }, [handleSend]);

  // Settings actions
  const saveKeys = async () => {
    const llmKey = (document.getElementById("key-llm") as HTMLInputElement)?.value || "";
    const zhipuKey = (document.getElementById("key-zhipu") as HTMLInputElement)?.value || "";
    await fetch(`${API_BASE}/api/config/keys`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ llm_api_key: llmKey, zhipu_api_key: zhipuKey }) });
    const status = await fetch(`${API_BASE}/api/config/status`).then(r => r.json());
    setConfigStatus(status);
  };

  const saveDictionary = async () => {
    await fetch(`${API_BASE}/api/dictionary`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ terms: termsDict, synonyms, desensitize }) });
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`${API_BASE}/api/documents/upload`, { method: "POST", body: formData });
    const docs = await fetch(`${API_BASE}/api/documents`).then(r => r.json());
    setDocuments(docs.documents || []);
  };

  const runPipeline = async () => {
    await fetch(`${API_BASE}/api/pipeline/run`, { method: "POST" });
    // Poll status
    const poll = setInterval(async () => {
      const st = await fetch(`${API_BASE}/api/pipeline/status`).then(r => r.json());
      setPipelineStatus(st);
      if (st.stage === "completed" || st.stage === "failed") {
        clearInterval(poll);
        if (st.stage === "completed") {
          const health = await fetch(`${API_BASE}/api/health`).then(r => r.json());
          if (health.milvus?.rows) setChunkTotal(health.milvus.rows);
        }
      }
    }, 1000);
  };

  const onFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-white flex flex-col shrink-0">
        <div className="p-4 space-y-4">
          <div>
            <h1 className="text-lg font-bold text-gray-900">RAG Agent</h1>
            <p className="text-xs text-gray-500">可私有化部署的知识库问答系统</p>
          </div>
          <hr />
          <div className="text-xs text-gray-400 space-y-1">
            <p>知识库 {chunkTotal ?? "..."} chunks</p>
            {chunkTotal != null && chunkTotal === 0 && <p className="text-amber-600">尚未构建知识库</p>}
            {pipelineStatus && pipelineStatus.stage !== "idle" && pipelineStatus.stage !== "completed" && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full transition-all" style={{ width: `${(pipelineStatus.progress || 0) * 100}%` }} />
                </div>
                <p className="text-blue-600 mt-1">{pipelineStatus.message || pipelineStatus.stage}</p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main chat area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* Settings button */}
        <button onClick={() => setSettingsOpen(!settingsOpen)}
          className="absolute top-3 right-3 z-10 p-2 rounded-lg hover:bg-gray-200 transition-colors"
          title="设置">
          <Settings className="w-5 h-5 text-gray-500" />
        </button>

        <ScrollArea className="flex-1 min-h-0 p-6">
          <div className="max-w-3xl mx-auto space-y-4">
            {/* Onboarding card */}
            {messages.length === 0 && !setupComplete && (
              <div className="text-center py-12">
                <div className="max-w-md mx-auto bg-white rounded-xl shadow-sm border p-8 space-y-4">
                  <Search className="w-10 h-10 mx-auto opacity-30" />
                  <h2 className="text-lg font-semibold text-gray-800">🛠 欢迎使用 RAG Agent</h2>
                  <p className="text-sm text-gray-500">在使用问答前，请先完成以下设置：</p>
                  <div className="space-y-2 text-left">
                    {configStatus && [
                      { done: configStatus.llm_configured, label: "① 配置 API Key — 在右上角 ⚙ 设置中填写各平台 Key" },
                      { done: configStatus.bge_m3_downloaded, label: "② 下载模型 — 启动时自动从镜像站下载嵌入模型" },
                      { done: !!termsDict.trim(), label: "③ 上传领域词典 — 添加你所在领域的专业术语" },
                      { done: documents.length > 0, label: "④ 上传文档 — 拖拽上传你的设计文档、规范、手册" },
                      { done: !!(chunkTotal && chunkTotal > 0), label: "⑤ 构建知识库 — 在设置面板一键完成解析、分块、索引" },
                    ].map((item, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm">
                        {item.done ? <Check className="w-4 h-4 text-green-500 shrink-0" /> : <div className="w-4 h-4 rounded-full border-2 border-gray-300 shrink-0" />}
                        <span className={item.done ? "text-green-700" : "text-gray-500"}>{item.label}</span>
                      </div>
                    ))}
                  </div>
                  {setupComplete && (
                    <p className="text-sm text-green-600 font-medium">✅ 全部就绪，开始提问吧！</p>
                  )}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <ChatBubble key={i} message={msg} loading={loading && i === messages.length - 1} />
            ))}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>

        {/* Input area */}
        <div className="border-t bg-white p-4">
          <div className="max-w-3xl mx-auto flex gap-3 items-end">
            <textarea ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={chunkTotal === 0 ? "请先在设置中上传文档并构建知识库..." : "输入问题，按 Enter 发送，Shift+Enter 换行..."}
              className="flex-1 min-h-[48px] max-h-[120px] resize-none rounded-lg border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={1} disabled={loading || chunkTotal === 0} />
            <Button onClick={handleSend} disabled={loading || !input.trim()}
              size="icon" className="shrink-0 h-[48px] w-[48px]">
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </div>
        </div>

        {/* ── Settings Panel (Slide-over) ── */}
        {settingsOpen && (
          <div className="fixed inset-0 z-50">
            <div className="absolute inset-0 bg-black/30" onClick={() => setSettingsOpen(false)} />
            <div className="absolute right-0 top-0 bottom-0 w-[440px] bg-white shadow-xl flex flex-col">
              <div className="flex items-center justify-between p-4 border-b">
                <h3 className="font-semibold text-lg">设置</h3>
                <button onClick={() => setSettingsOpen(false)} className="p-1 hover:bg-gray-100 rounded"><X className="w-5 h-5" /></button>
              </div>

              {/* Tabs */}
              <div className="flex border-b">
                {["keys", "dictionary", "documents"].map(tab => (
                  <button key={tab} onClick={() => setSettingsTab(tab as typeof settingsTab)}
                    className={`flex-1 py-2 text-sm font-medium border-b-2 transition-colors ${settingsTab === tab ? "border-blue-500 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
                    {tab === "keys" && "API 配置"}
                    {tab === "dictionary" && "词典管理"}
                    {tab === "documents" && "文档管理"}
                  </button>
                ))}
              </div>

              <ScrollArea className="flex-1 p-4">
                {/* Tab: API Keys */}
                {settingsTab === "keys" && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">DeepSeek API Key</label>
                      <input id="key-llm" type="password" className="w-full mt-1 rounded-lg border px-3 py-2 text-sm" placeholder="sk-..." />
                    </div>
                    <div>
                      <label className="text-sm font-medium">智谱 AI API Key</label>
                      <input id="key-zhipu" type="password" className="w-full mt-1 rounded-lg border px-3 py-2 text-sm" placeholder="xxxxxxxx" />
                    </div>
                    <Button onClick={saveKeys} className="w-full">保存 API Key</Button>
                    {configStatus && (
                      <div className="space-y-1 text-xs">
                        <div className="flex items-center gap-1">{configStatus.llm_configured ? <Check className="w-3 h-3 text-green-500" /> : <AlertTriangle className="w-3 h-3 text-amber-500" />} DeepSeek</div>
                        <div className="flex items-center gap-1">{configStatus.zhipu_configured ? <Check className="w-3 h-3 text-green-500" /> : <AlertTriangle className="w-3 h-3 text-amber-500" />} 智谱 AI</div>
                        <div className="flex items-center gap-1">{configStatus.bge_m3_downloaded ? <Check className="w-3 h-3 text-green-500" /> : <AlertTriangle className="w-3 h-3 text-amber-500" />} BGE-M3</div>
                        <div className="flex items-center gap-1">{configStatus.reranker_downloaded ? <Check className="w-3 h-3 text-green-500" /> : <AlertTriangle className="w-3 h-3 text-amber-500" />} BGE-Reranker</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Tab: Dictionary */}
                {settingsTab === "dictionary" && (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">领域分词词典</label>
                      <p className="text-xs text-gray-400 mb-1">每行一个专业术语</p>
                      <textarea value={termsDict} onChange={e => setTermsDict(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm h-32 resize-none font-mono" placeholder="需要系数&#10;一类高层住宅&#10;消防控制室" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">查询同义词</label>
                      <p className="text-xs text-gray-400 mb-1">口语词=标准术语，每行一对</p>
                      <textarea value={synonyms} onChange={e => setSynonyms(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm h-24 resize-none font-mono" placeholder="浪涌=电涌&#10;竖井=电井" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">脱敏规则</label>
                      <p className="text-xs text-gray-400 mb-1">JSON 格式数组</p>
                      <textarea value={desensitize} onChange={e => setDesensitize(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm h-24 resize-none font-mono" />
                    </div>
                    <Button onClick={saveDictionary} className="w-full">保存词典</Button>
                  </div>
                )}

                {/* Tab: Documents */}
                {settingsTab === "documents" && (
                  <div className="space-y-4">
                    <div className="border-2 border-dashed rounded-lg p-6 text-center"
                      onDragOver={e => e.preventDefault()} onDrop={onFileDrop}>
                      <Upload className="w-8 h-8 mx-auto text-gray-300 mb-2" />
                      <p className="text-sm text-gray-500">拖拽文件到此处上传</p>
                      <p className="text-xs text-gray-400 mt-1">支持 DOCX, PDF, PPTX, XLS, XLSX, TXT</p>
                      <label className="mt-3 inline-block cursor-pointer text-sm text-blue-600 hover:underline">
                        或点击选择文件
                        <input type="file" className="hidden" accept=".docx,.pdf,.pptx,.xls,.xlsx,.txt,.md,.doc"
                          onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f); }} />
                      </label>
                    </div>

                    {documents.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-sm font-medium">已上传文档 ({documents.length})</p>
                        {documents.map((doc, i) => (
                          <div key={i} className="flex items-center justify-between text-sm bg-gray-50 rounded px-3 py-2">
                            <span className="truncate">{doc.filename}</span>
                            <span className="text-xs text-gray-400 shrink-0">{(doc.size / 1024).toFixed(0)} KB</span>
                          </div>
                        ))}
                      </div>
                    )}

                    <Button onClick={runPipeline}
                      disabled={documents.length === 0 || (pipelineStatus && pipelineStatus.stage !== "idle" && pipelineStatus.stage !== "completed" && pipelineStatus.stage !== "failed")}
                      className="w-full">
                      {pipelineStatus && pipelineStatus.stage !== "idle" && pipelineStatus.stage !== "completed" && pipelineStatus.stage !== "failed"
                        ? `${pipelineStatus.message || "处理中..."}`
                        : "构建知识库"}
                    </Button>

                    {/* Audit results */}
                    {pipelineStatus?.audit?.checks && pipelineStatus.audit.checks.length > 0 && (
                      <div className="space-y-1 text-xs border rounded-lg p-3 bg-gray-50">
                        <p className="font-medium mb-1">审计结果 {pipelineStatus.audit.passed ? "✅" : "❌"}</p>
                        {pipelineStatus.audit.checks.map((c, i) => (
                          <div key={i} className="flex items-center gap-1">
                            {c.passed ? <Check className="w-3 h-3 text-green-500 shrink-0" />
                              : c.level === "ERROR" ? <AlertTriangle className="w-3 h-3 text-red-500 shrink-0" />
                              : <AlertTriangle className="w-3 h-3 text-amber-500 shrink-0" />}
                            <span className={c.passed ? "text-gray-600" : c.level === "ERROR" ? "text-red-600" : "text-amber-600"}>
                              {c.name}: {c.detail}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </ScrollArea>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function ChatBubble({ message, loading }: { message: Message; loading: boolean }) {
  const isUser = message.role === "user";
  const isStreaming = !isUser && loading;
  const [renderedContent, setRenderedContent] = useState(message.content);
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);

  useEffect(() => {
    if (!isStreaming) { setRenderedContent(message.content); return; }
    const timer = setTimeout(() => setRenderedContent(message.content), 150);
    return () => clearTimeout(timer);
  }, [message.content, isStreaming]);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] ${isUser ? "order-1" : ""}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-gray-500">{isUser ? "你" : "RAG Agent"}</span>
          {message.retrievalTimeMs != null && (
            <span className="text-[10px] text-gray-400">检索 {message.retrievalTimeMs.toFixed(0)}ms · 生成 {message.generationTimeMs?.toFixed(0)}ms</span>
          )}
        </div>
        <Card className={`p-4 ${isUser ? "bg-blue-500 text-white border-0" : "bg-white"}`}>
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="text-sm text-gray-700 space-y-2 markdown-body">
              {renderedContent ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}
                  urlTransform={(url) => url.startsWith("/api/") ? `${API_BASE}${url}` : url}
                  components={{
                    img: ({ src, alt }) => (
                      <img src={src || ""} alt={alt || ""}
                        className="max-w-full rounded-lg cursor-zoom-in hover:shadow-lg transition-shadow border my-2"
                        style={{ maxHeight: "400px" }}
                        onClick={(e) => { e.stopPropagation(); setZoomedImage(typeof src === "string" ? src : null); }} />
                    ),
                  }}>
                  {renderedContent}
                </ReactMarkdown>
              ) : loading ? <span className="text-gray-400">思考中...</span> : <span className="text-gray-400">空响应</span>}
            </div>
          )}
        </Card>

        {message.sources && message.sources.length > 0 && (() => {
          const deduped = [...new Map(message.sources.map(s => [s.source_file, s])).values()];
          const sorted = [...deduped].sort((a, b) => a.relevance === "direct" ? -1 : 1).slice(0, 4);
          return (
            <div className="mt-2 space-y-1">
              <div className="flex items-center gap-1 text-[10px] text-gray-400"><FileText className="w-3 h-3" />参考来源</div>
              {sorted.map((s, j) => (
                <div key={j} className={`text-[11px] rounded px-2 py-1 flex items-center gap-1.5 ${s.relevance === "direct" ? "bg-blue-50" : "bg-gray-50"}`}>
                  <span className={`shrink-0 text-[10px] px-1 py-0.5 rounded font-medium ${s.relevance === "direct" ? "bg-blue-500 text-white" : "bg-gray-300 text-gray-500"}`}>
                    {s.relevance === "direct" ? "直接参考" : "拓展参考"}
                  </span>
                  <span className="font-medium">{s.source_file}</span>
                  {s.section_path && <span className="text-gray-400">· {s.section_path}</span>}
                </div>
              ))}
            </div>
          );
        })()}

        {zoomedImage && (
          <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 cursor-zoom-out" onClick={() => setZoomedImage(null)}>
            <button className="absolute top-4 right-4 text-white hover:text-gray-300" onClick={() => setZoomedImage(null)}><X className="w-8 h-8" /></button>
            <img src={zoomedImage} alt="预览" className="max-w-full max-h-full object-contain rounded-lg" onClick={e => e.stopPropagation()} />
          </div>
        )}
      </div>
    </div>
  );
}
