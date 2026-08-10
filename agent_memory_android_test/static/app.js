/* Agent Memory 测试 App 前端：原生桥 + 聊天 + agent_memory 四类记忆视图。 */

const bridge = window.AiGlassesAndroid || null;
const isNative = () => Boolean(bridge);

const state = {
  ownerId: "",
  voiceEnabled: true,
  ambientRunning: false,
  audio: {},
  memoryTab: "facts",
  memorySearch: "",
  lastDebug: null,
  enrollment: { sessionId: "", running: false },
  surfaces: { memory: false, debug: false, settings: false },
};

const messagesEl = document.getElementById("messages");
const voiceStatusEl = document.getElementById("voice-status");
const ambientModeEl = document.getElementById("ambient-mode-label");
const ambientStatusEl = document.getElementById("ambient-status");
const memoryListEl = document.getElementById("memory-list");
const debugOutputEl = document.getElementById("debug-output");
const toastEl = document.getElementById("toast");
const tooltipEl = document.getElementById("button-tooltip");

const ENROLL_PHRASES = [
  "你好小忆，现在开始录入我的声音。",
  "我在测试 agent memory 的记忆系统。",
  "今天天气不错，适合出门走走。",
];

/* ---------- 基础工具 ---------- */

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message, kind = "info") {
  toastEl.textContent = message;
  toastEl.className = `toast ${kind}`;
  toastEl.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toastEl.hidden = true;
  }, 3000);
}

function setVoiceStatus(text, kind = "") {
  voiceStatusEl.textContent = text;
  voiceStatusEl.className = `voice-status ${kind}`.trim();
}

function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : role === "assistant" ? "忆" : "!";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  node.append(avatar, bubble);
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function ownerId() {
  return state.ownerId || "local-user";
}

/* ---------- 原生桥 ---------- */

function callBridge(name, ...args) {
  if (!bridge || typeof bridge[name] !== "function") return null;
  const raw = bridge[name](...args);
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return raw;
  }
}

function speak(text) {
  if (!state.voiceEnabled || !text) return;
  if (bridge && bridge.speak) bridge.speak(String(text).slice(0, 4000));
}

/* ---------- 聊天 ---------- */

async function sendChat(text) {
  const message = String(text || "").trim();
  if (!message) return;
  appendMessage("user", message);
  const input = document.getElementById("message-input");
  if (input) input.value = "";
  const typing = appendMessage("assistant typing", "正在思考...");
  try {
    const result = await requestJSON("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, user_id: ownerId() }),
    });
    typing.remove();
    appendMessage("assistant", result.reply);
    state.lastDebug = result.debug || null;
    renderDebug();
    speak(result.reply);
    if (state.surfaces.memory) refreshMemoryPanel();
  } catch (error) {
    typing.remove();
    appendMessage("system", error.message || "聊天失败");
  }
}

function pollCompletedReplies() {
  if (!isNative()) return;
  const replies = callBridge("consumeCompletedReplies");
  if (!Array.isArray(replies) || replies.length === 0) return;
  for (const item of replies) {
    if (item.query) appendMessage("user", item.query);
    if (item.reply) {
      appendMessage("assistant", item.reply);
      speak(item.reply);
    }
  }
}

/* ---------- 收音与状态 ---------- */

function applyAudioStatus(status) {
  if (!status || typeof status !== "object") return;
  state.audio = { ...state.audio, ...status };
  const running = Boolean(status.running);
  state.ambientRunning = running;
  const modelState = String(status.model_state || "not_installed");
  const deviceLabel = status.input_device_name
    ? `${status.input_device_name}${status.input_device_source === "bluetooth" ? "（蓝牙）" : ""}`
    : "未确认";
  const labels = [
    `收音=${deviceLabel}`,
    `VAD片段=${Number(status.vad_segment_count || 0)}`,
    `ambient final=${Number(status.ambient_final_count || 0)}`,
    `拒绝=${Number(status.speech_rejected_count || 0)}`,
    `模型=${modelState}`,
  ];
  ambientModeEl.textContent = running ? "全天收音待机中" : "收音待机未启动";
  ambientStatusEl.textContent = labels.join(" · ");
  const toggle = document.getElementById("ambient-standby-toggle");
  toggle.textContent = running ? "停止全天待机" : "开启全天待机";
  const details = document.getElementById("settings-audio-details");
  if (details) details.textContent = labels.join("\n");
  if (running) {
    setVoiceStatus("收音待机中，说“你好小忆”发起语音问答");
  } else if (modelState === "ready") {
    setVoiceStatus("本地语音模型已就绪，可开启全天待机");
  } else {
    setVoiceStatus(`原生麦克风已就绪；本地模型状态：${modelState}`);
  }
  syncEnrollmentFromStatus(status);
}

async function syncAudioStatus() {
  if (!isNative()) return;
  const status = callBridge("audioStatus");
  if (status) applyAudioStatus(status);
}

async function refreshSpeakerSummary() {
  const summaryEl = document.getElementById("speaker-setting-summary");
  try {
    const profile = await requestJSON(`/api/speaker/profile?user_id=${encodeURIComponent(ownerId())}`);
    summaryEl.textContent = profile.enrolled
      ? `已录入 ${profile.sample_count} 段（${profile.model || "campplus"}）`
      : "尚未录入";
  } catch (_) {
    summaryEl.textContent = "读取失败";
  }
}

/* ---------- 记忆面板 ---------- */

function setMemoryTab(tab) {
  state.memoryTab = tab;
  for (const name of ["facts", "states", "actionables", "episodes"]) {
    const button = document.getElementById(`memory-tab-${name}`);
    if (button) button.setAttribute("aria-selected", String(name === tab));
  }
  loadMemoryList();
}

async function loadMemoryStats() {
  try {
    const stats = await requestJSON("/api/agent-memory/stats");
    const counts = stats.counts || {};
    document.getElementById("stat-facts").textContent = counts.facts || 0;
    document.getElementById("stat-states").textContent = counts.states || 0;
    document.getElementById("stat-actionables").textContent = counts.actionables || 0;
    document.getElementById("stat-episodes").textContent = counts.episodes || 0;
    const total = Object.values(counts).reduce((sum, n) => sum + Number(n || 0), 0);
    document.getElementById("memory-count").textContent = `${total} 条`;
    document.getElementById("memory-reflect-info").textContent = stats.last_reflect_at
      ? `最近反思：${stats.last_reflect_at}`
      : "最近反思：暂无";
  } catch (error) {
    showToast(`统计加载失败：${error.message}`, "error");
  }
}

async function loadMemoryList() {
  const q = state.memorySearch.trim();
  const url = `/api/agent-memory/${state.memoryTab}?limit=100${q ? `&q=${encodeURIComponent(q)}` : ""}`;
  memoryListEl.textContent = "加载中…";
  try {
    const result = await requestJSON(url);
    renderMemoryItems(state.memoryTab, result.items || []);
  } catch (error) {
    memoryListEl.textContent = `加载失败：${error.message}`;
  }
}

function renderMemoryItems(tab, items) {
  if (items.length === 0) {
    memoryListEl.textContent = "暂无数据";
    return;
  }
  memoryListEl.textContent = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "memory-card";
    const title = document.createElement("div");
    title.className = "memory-card-title";
    const body = document.createElement("div");
    body.className = "memory-card-body";
    const meta = document.createElement("div");
    meta.className = "memory-card-meta";

    if (tab === "facts") {
      title.textContent = item.summary || `事实 #${item.id}`;
      body.textContent = [item.fact_kind, item.fact_subject, item.fact_root_topic]
        .filter(Boolean)
        .join(" · ");
      meta.textContent = `${item.time_key || "未知时间"} · 重要度 ${Number(item.importance || 0).toFixed(2)} · ${item.created_at || ""}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ghost memory-card-delete";
      remove.dataset.factId = String(item.id);
      remove.textContent = "删除";
      card.append(title, body, meta, remove);
    } else if (tab === "states") {
      title.textContent = item.canonical_name || `状态 #${item.id}`;
      body.textContent = item.summary || "";
      const timeline = Array.isArray(item.time_line) ? item.time_line : [];
      meta.textContent = `${item.state_type || ""} · ${item.state_scope || ""} · ${item.updated_at || ""}`;
      card.append(title, body, meta);
      if (timeline.length) {
        const line = document.createElement("div");
        line.className = "memory-card-timeline";
        line.textContent = `演化：${timeline.join(" -> ")}`;
        card.append(line);
      }
    } else if (tab === "actionables") {
      title.textContent = item.canonical_name || item.summary || `待办 #${item.id}`;
      body.textContent = item.summary || "";
      meta.textContent = `${item.item_type || ""} · 状态 ${item.status || ""} · 负责人 ${item.owner || ""} · 截止 ${item.due_at || "未设置"} · ${item.updated_at || ""}`;
      card.append(title, body, meta);
    } else {
      title.textContent = item.title || `片段 #${item.id}`;
      body.textContent = item.summary || "";
      const participants = Array.isArray(item.participants) ? item.participants : [];
      meta.textContent = `${item.source_type || ""} · ${item.started_at || ""} · 参与者 ${participants.join("、") || "无"}`;
      card.append(title, body, meta);
    }
    memoryListEl.appendChild(card);
  }
}

async function refreshMemoryPanel() {
  await Promise.all([loadMemoryStats(), loadMemoryList()]);
}

async function triggerReflect() {
  const button = document.getElementById("trigger-reflect");
  button.disabled = true;
  button.textContent = "反思中…";
  try {
    await requestJSON("/api/agent-memory/reflect", { method: "POST", body: JSON.stringify({ user_id: ownerId() }) });
    showToast("反思完成，状态/待办已更新");
  } catch (error) {
    showToast(`反思失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "触发反思";
    await refreshMemoryPanel();
  }
}

async function deleteFact(factId) {
  try {
    const result = await requestJSON(`/api/agent-memory/facts/${factId}`, { method: "DELETE" });
    showToast(result.deleted ? "已删除该事实" : "未找到该事实");
  } catch (error) {
    showToast(`删除失败：${error.message}`, "error");
  }
  await refreshMemoryPanel();
}

/* ---------- 召回调试 ---------- */

function renderDebug() {
  if (!state.lastDebug) {
    debugOutputEl.textContent = "等待用户发送 query...";
    return;
  }
  debugOutputEl.textContent = JSON.stringify(state.lastDebug, null, 2);
}

/* ---------- 声纹录入 ---------- */

function syncEnrollmentFromStatus(status) {
  const enrollmentState = String(status.enrollment_state || "idle");
  const sessionId = String(status.enrollment_session_id || "");
  if (sessionId) state.enrollment.sessionId = sessionId;
  const count = Number(status.enrollment_sample_count || 0);
  const total = Number(status.enrollment_sample_total || 3);
  const modal = document.getElementById("speaker-enroll-modal");
  if (!modal || modal.hidden) return;
  const progress = document.getElementById("speaker-enroll-progress");
  const statusLine = document.getElementById("speaker-enroll-status");
  progress.textContent = `当前进度：${count} / ${total}`;
  if (enrollmentState === "completed" || count >= total) {
    statusLine.textContent = "录入完成，声纹已可用于说话人标注。";
    document.getElementById("speaker-enroll-start").disabled = false;
    document.getElementById("speaker-enroll-start").textContent = "重新录入";
  } else if (enrollmentState === "error") {
    statusLine.textContent = `录入出错：${status.last_error || "未知错误"}`;
    document.getElementById("speaker-enroll-start").disabled = false;
  } else if (enrollmentState === "collecting") {
    statusLine.textContent = `正在录入第 ${Math.min(count + 1, total)} 段，请朗读下方内容。`;
    document.getElementById("speaker-enroll-start").disabled = true;
    document.getElementById("speaker-enroll-start").textContent = "录入中…";
  }
  const phraseIndex = Math.min(count, ENROLL_PHRASES.length - 1);
  document.getElementById("speaker-enroll-phrase").textContent = ENROLL_PHRASES[phraseIndex];
  document.getElementById("speaker-enroll-phrase-label").textContent = `第 ${phraseIndex + 1} 段朗读内容`;
}

function openEnrollment() {
  const modal = document.getElementById("speaker-enroll-modal");
  modal.hidden = false;
  document.getElementById("speaker-enroll-progress").textContent = "当前进度：0 / 3";
  document.getElementById("speaker-enroll-phrase").textContent = ENROLL_PHRASES[0];
  document.getElementById("speaker-enroll-phrase-label").textContent = "第 1 段朗读内容";
  document.getElementById("speaker-enroll-status").textContent = "点击开始录入后，完成当前段朗读内容即可；暂时不录也不影响文字聊天。";
  document.getElementById("speaker-enroll-start").disabled = false;
  document.getElementById("speaker-enroll-start").textContent = "开始录入";
  refreshSpeakerSummary();
}

function closeEnrollment() {
  document.getElementById("speaker-enroll-modal").hidden = true;
  document.body.classList.remove("speaker-open");
}

function startEnrollment() {
  const sessionId = `enroll_${Date.now().toString(36)}`;
  state.enrollment.sessionId = sessionId;
  document.getElementById("speaker-enroll-start").disabled = true;
  document.getElementById("speaker-enroll-start").textContent = "录入中…";
  callBridge("startSpeakerEnrollment", sessionId);
}

function cancelEnrollment() {
  callBridge("cancelSpeakerEnrollment");
  closeEnrollment();
}

/* ---------- 面板切换 ---------- */

function toggleSurface(name, force) {
  const next = typeof force === "boolean" ? force : !state.surfaces[name];
  state.surfaces[name] = next;
  const pane = name === "memory" ? document.getElementById("memory-pane")
    : name === "debug" ? document.getElementById("debug-pane")
    : null;
  if (pane) pane.hidden = !next;
  const backdropId = name === "memory" ? "memory-backdrop" : "debug-backdrop";
  const backdrop = document.getElementById(backdropId);
  if (backdrop) backdrop.hidden = !next;
  const button = document.getElementById(`${name}-toggle`);
  if (button) button.setAttribute("aria-expanded", String(next));
  if (name === "memory" && pane) {
    pane.classList.toggle("open", next);
    document.body.classList.toggle("memory-open", next);
  }
  if (name === "debug") {
    document.body.classList.toggle("debug-open", next);
  }
  if (name === "memory" && next) refreshMemoryPanel();
  if (name === "debug" && next) renderDebug();
}

function toggleSettings(force) {
  const next = typeof force === "boolean" ? force : !state.surfaces.settings;
  state.surfaces.settings = next;
  document.body.classList.toggle("settings-open", next);
  const pane = document.getElementById("settings-pane");
  const backdrop = document.getElementById("settings-backdrop");
  if (pane) pane.hidden = !next;
  if (backdrop) backdrop.hidden = !next;
  const button = document.getElementById("settings-toggle");
  if (button) button.setAttribute("aria-expanded", String(next));
}

function aiGlassesHandleBack() {
  const enrollment = document.getElementById("speaker-enroll-modal");
  if (enrollment && !enrollment.hidden) {
    closeEnrollment();
    return true;
  }
  if (state.surfaces.memory) {
    toggleSurface("memory", false);
    return true;
  }
  if (state.surfaces.debug) {
    toggleSurface("debug", false);
    return true;
  }
  if (state.surfaces.settings) {
    toggleSettings(false);
    return true;
  }
  return false;
}
window.aiGlassesHandleBack = aiGlassesHandleBack;

/* ---------- 事件绑定 ---------- */

function bindEvents() {
  document.getElementById("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendChat(document.getElementById("message-input").value);
  });

  document.getElementById("ambient-standby-toggle").addEventListener("click", () => {
    if (!isNative()) {
      showToast("当前不是原生 Android 环境，无法开启收音", "error");
      return;
    }
    if (state.ambientRunning) callBridge("stopAmbient");
    else callBridge("startAmbient");
  });

  document.getElementById("settings-toggle").addEventListener("click", () => {
    if (state.surfaces.memory) toggleSurface("memory", false);
    if (state.surfaces.debug) toggleSurface("debug", false);
    toggleSettings();
  });
  document.getElementById("settings-close").addEventListener("click", () => toggleSettings(false));
  document.getElementById("settings-backdrop").addEventListener("click", () => toggleSettings(false));

  document.getElementById("memory-toggle").addEventListener("click", () => {
    if (state.surfaces.settings) toggleSettings(false);
    toggleSurface("memory");
  });
  document.getElementById("memory-close").addEventListener("click", () => toggleSurface("memory", false));
  document.getElementById("memory-backdrop").addEventListener("click", () => toggleSurface("memory", false));
  document.getElementById("refresh-memory").addEventListener("click", refreshMemoryPanel);

  document.getElementById("debug-toggle").addEventListener("click", () => {
    if (state.surfaces.settings) toggleSettings(false);
    toggleSurface("debug");
  });
  document.getElementById("close-debug").addEventListener("click", () => toggleSurface("debug", false));
  document.getElementById("debug-backdrop").addEventListener("click", () => toggleSurface("debug", false));
  document.getElementById("clear-debug").addEventListener("click", () => {
    state.lastDebug = null;
    renderDebug();
  });

  for (const tab of ["facts", "states", "actionables", "episodes"]) {
    document.getElementById(`memory-tab-${tab}`).addEventListener("click", () => setMemoryTab(tab));
  }

  const searchInput = document.getElementById("memory-search");
  searchInput.addEventListener("input", () => {
    state.memorySearch = searchInput.value;
  });
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadMemoryList();
  });

  document.getElementById("trigger-reflect").addEventListener("click", triggerReflect);
  document.getElementById("memory-import-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("memory-import-text");
    const text = input.value.trim();
    if (!text) return;
    try {
      await requestJSON("/api/agent-memory/import", {
        method: "POST",
        body: JSON.stringify({ text, user_id: ownerId() }),
      });
      input.value = "";
      showToast("已导入到 agent_memory");
      await refreshMemoryPanel();
    } catch (error) {
      showToast(`导入失败：${error.message}`, "error");
    }
  });

  memoryListEl.addEventListener("click", (event) => {
    const button = event.target.closest(".memory-card-delete");
    if (button && button.dataset.factId) deleteFact(button.dataset.factId);
  });

  document.getElementById("speaker-enroll-button").addEventListener("click", openEnrollment);
  document.getElementById("speaker-enroll-close").addEventListener("click", closeEnrollment);
  document.getElementById("speaker-enroll-start").addEventListener("click", startEnrollment);
  document.getElementById("speaker-enroll-retry").addEventListener("click", startEnrollment);
  document.getElementById("speaker-enroll-cancel").addEventListener("click", cancelEnrollment);

  document.getElementById("voice-toggle").addEventListener("click", () => {
    state.voiceEnabled = !state.voiceEnabled;
    const button = document.getElementById("voice-toggle");
    button.textContent = state.voiceEnabled ? "播报开" : "播报关";
    button.setAttribute("aria-checked", String(state.voiceEnabled));
    if (!state.voiceEnabled && bridge && bridge.stopSpeaking) bridge.stopSpeaking();
  });

  const nativeSettings = document.getElementById("native-settings-button");
  nativeSettings.addEventListener("click", () => {
    if (bridge && bridge.openSettings) bridge.openSettings();
    else showToast("当前不是原生 Android 环境", "error");
  });

  document.querySelectorAll("[data-tooltip]").forEach((element) => {
    element.addEventListener("mouseenter", () => {
      tooltipEl.textContent = element.dataset.tooltip || "";
      tooltipEl.hidden = false;
      const rect = element.getBoundingClientRect();
      tooltipEl.style.top = `${rect.top - tooltipEl.offsetHeight - 6}px`;
      tooltipEl.style.left = `${rect.left + rect.width / 2 - tooltipEl.offsetWidth / 2}px`;
    });
    element.addEventListener("mouseleave", () => {
      tooltipEl.hidden = true;
    });
  });
}

/* ---------- 初始化 ---------- */

async function init() {
  bindEvents();
  state.ownerId = isNative() ? String(bridge.ownerId() || "local-user") : "local-user";
  const ownerSummary = document.getElementById("owner-setting-summary");
  const ownerValue = document.getElementById("owner-setting-value");
  if (ownerSummary) ownerSummary.textContent = `ID：${ownerId()}`;
  if (ownerValue) ownerValue.textContent = ownerId();

  if (isNative()) {
    document.body.classList.add(`${String(bridge.platform() || "android")}-native`);
    await syncAudioStatus();
  } else {
    setVoiceStatus("浏览器模式：仅支持文字聊天与记忆面板");
  }
  refreshSpeakerSummary();
  if (state.surfaces.memory) refreshMemoryPanel();
  window.setInterval(syncAudioStatus, 1500);
  window.setInterval(pollCompletedReplies, 2000);
  window.setInterval(fastPollAudioUi, FAST_POLL_MS);
}

init();
