/**
 * IntentRoute — Core Frontend Application Logic
 * Clean SVG iconography · Zero Emojis · Responsive Views
 */

const API = '';

// ── State Management ──────────────────────────────────────────
let currentSession = null;
let currentTheme   = localStorage.getItem('intentroute_theme') || 'dark';
let activeTab      = 'chat';
let chatHistory    = [];
let isSubmitting   = false;
let sessionStats   = {
  queries: 0,
  cacheHits: 0,
  totalMs: 0,
};

// ── DOM References ────────────────────────────────────────────
// Auth elements
const authScreen     = document.getElementById('auth-screen');
const appShell        = document.getElementById('app-shell');
const loginForm      = document.getElementById('login-form');
const loginUserInp   = document.getElementById('login-username');
const loginPwdInp    = document.getElementById('login-password');
const loginBtn       = document.getElementById('login-btn');
const togglePwdBtn   = document.getElementById('toggle-pwd-btn');
const userAvatarText = document.getElementById('user-avatar-text');
const userDispName   = document.getElementById('user-display-name');
const userDispRole   = document.getElementById('user-display-role');
const logoutBtn      = document.getElementById('logout-btn');

// Theme
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeLabel     = document.getElementById('theme-label');

// Navigation
const navTabs        = document.querySelectorAll('.nav-tab');
const viewChat       = document.getElementById('view-chat');
const viewArch       = document.getElementById('view-architecture');
const viewAnalytics  = document.getElementById('view-analytics');
const viewPerformance = document.getElementById('view-performance');

// Chat elements
const chatGreeting   = document.getElementById('chat-greeting-view');
const chatMessages   = document.getElementById('chat-messages');
const chatInput      = document.getElementById('chat-input');
const chatSendBtn    = document.getElementById('chat-send-btn');
const clearChatBtn   = document.getElementById('clear-chat-btn');
const thinkToggleBtn = document.getElementById('think-toggle-btn');
const micBtn         = document.getElementById('mic-btn');
const chatPlusBtn    = document.getElementById('chat-plus-btn');

// Architecture / Simulator
const simQueryInput  = document.getElementById('sim-query-input');
const simRunBtn      = document.getElementById('sim-run-btn');
const simClearBtn    = document.getElementById('clear-sim-btn');
const simResultBox   = document.getElementById('sim-result');

// Analytics elements
const metricQueries  = document.getElementById('metric-queries');
const metricCache    = document.getElementById('metric-cache-rate');
const metricLatency  = document.getElementById('metric-latency');
const metricCpu      = document.getElementById('metric-cpu');
const metricRam      = document.getElementById('metric-ram');
const metricCacheBack = document.getElementById('metric-cache-backend');
const historyList    = document.getElementById('analytics-history-list');
const cacheFlushBtn  = document.getElementById('cache-flush-btn');

// Toast
const toastEl        = document.getElementById('toast');


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initAuth();
  initNavigation();
  initChatInput();
  initVoiceInput();
  initSimulator();
  initAnalytics();
});


// ============================================================
// THEME MANAGEMENT (NORMAL WHITE VS DARK BLACK)
// ============================================================

function initTheme() {
  applyTheme(currentTheme);
  themeToggleBtn.addEventListener('click', () => {
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  });
}

function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('intentroute_theme', theme);
  if (themeToggleBtn) {
    themeToggleBtn.setAttribute('aria-checked', theme === 'dark' ? 'true' : 'false');
    themeToggleBtn.setAttribute('title', theme === 'dark' ? 'Switch to Light mode' : 'Switch to Dark mode');
  }
  if (themeLabel) {
    themeLabel.textContent = theme === 'dark' ? 'Dark' : 'Light';
  }
}


// ============================================================
// AUTHENTICATION SYSTEM
// ============================================================

function initAuth() {
  const savedUser = localStorage.getItem('intentroute_user');
  const savedToken = localStorage.getItem('intentroute_token');

  if (savedUser && savedToken) {
    try {
      currentSession = JSON.parse(savedUser);
      showAppShell();
    } catch {
      showAuthScreen();
    }
  } else {
    showAuthScreen();
  }

  // Login form submission
  loginForm.addEventListener('submit', handleLogin);

  // Toggle password visibility
  if (togglePwdBtn) {
    togglePwdBtn.addEventListener('click', () => {
      const isPwd = loginPwdInp.type === 'password';
      loginPwdInp.type = isPwd ? 'text' : 'password';
    });
  }

  // Logout button
  if (logoutBtn) {
    logoutBtn.addEventListener('click', handleLogout);
  }
}

async function handleLogin(e) {
  if (e) e.preventDefault();
  const username = loginUserInp.value.trim();
  const password = loginPwdInp.value.trim();

  if (!username || !password) {
    showToast('Please enter both username and password.');
    return;
  }

  loginBtn.disabled = true;
  const originalText = loginBtn.innerHTML;
  loginBtn.innerHTML = '<span class="btn-text">Authenticating...</span>';

  try {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    let data;
    if (res.ok) {
      data = await res.json();
    } else {
      // Fallback local authentication
      data = {
        success: true,
        token: `ir_token_${Date.now()}`,
        user: {
          username: username,
          name: username.charAt(0).toUpperCase() + username.slice(1),
          email: `${username.toLowerCase()}@intentroute.ai`,
          role: username.toLowerCase() === 'aakif' ? 'Lead AI Engineer' : 'Research Member',
        }
      };
    }

    currentSession = data.user;
    localStorage.setItem('intentroute_user', JSON.stringify(data.user));
    localStorage.setItem('intentroute_token', data.token);

    showAppShell();
    showToast(`Signed in as ${data.user.name}`);
  } catch {
    // Graceful offline fallback
    const fallbackUser = {
      username: username,
      name: username.charAt(0).toUpperCase() + username.slice(1),
      email: `${username.toLowerCase()}@intentroute.ai`,
      role: 'Engineer',
    };
    currentSession = fallbackUser;
    localStorage.setItem('intentroute_user', JSON.stringify(fallbackUser));
    localStorage.setItem('intentroute_token', `ir_local_${Date.now()}`);
    showAppShell();
    showToast(`Signed in as ${fallbackUser.name}`);
  } finally {
    loginBtn.disabled = false;
    loginBtn.innerHTML = originalText;
  }
}

function quickLogin(username, password) {
  loginUserInp.value = username;
  loginPwdInp.value = password;
  handleLogin();
}

function showAuthScreen() {
  authScreen.classList.remove('hidden');
  appShell.classList.add('hidden');
  authScreen.classList.remove('auth-visible');
  requestAnimationFrame(() => authScreen.classList.add('auth-visible'));
}

function showAppShell() {
  authScreen.classList.add('hidden');
  appShell.classList.remove('hidden');
  appShell.classList.remove('app-entering');
  requestAnimationFrame(() => appShell.classList.add('app-entering'));

  if (currentSession) {
    const name = currentSession.name || currentSession.username || 'User';
    userAvatarText.textContent = name.charAt(0).toUpperCase();
    userDispName.textContent   = name;
    if (userDispRole) userDispRole.textContent = currentSession.role || 'Member';
    restoreChatHistory();
  }
}

function chatStorageKey() {
  const username = currentSession?.username || currentSession?.email || 'guest';
  return `intentroute_chat_${username.toLowerCase()}`;
}

function saveChatHistory() {
  try {
    localStorage.setItem(chatStorageKey(), JSON.stringify(chatHistory.slice(0, 50)));
  } catch (error) {
    console.warn('Unable to save chat history:', error);
  }
}

function restoreChatHistory() {
  if (!chatMessages || !currentSession) return;

  try {
    chatMessages.innerHTML = '';
    chatHistory = [];
    const saved = JSON.parse(localStorage.getItem(chatStorageKey()) || '[]');
    if (!Array.isArray(saved)) return;

    chatHistory = saved;
    saved.slice().reverse().forEach(data => {
      const turnId = `restored-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const turnDiv = document.createElement('div');
      turnDiv.className = 'chat-turn';
      turnDiv.id = turnId;
      turnDiv.innerHTML = `
        <div class="user-msg-row">
          <div class="user-bubble">${escapeHtml(data.question || '')}</div>
        </div>
        <div class="bot-msg-row"></div>
      `;
      chatMessages.appendChild(turnDiv);
      renderBotResponse(turnDiv, turnId, data.question || '', data, 0);
    });

    if (saved.length && chatGreeting) chatGreeting.style.display = 'none';
    scrollChatToBottom(false);
  } catch (error) {
    console.warn('Unable to restore chat history:', error);
  }
}

async function handleLogout() {
  try {
    await fetch(`${API}/api/auth/logout`, { method: 'POST' });
  } catch { /* silent */ }

  currentSession = null;
  localStorage.removeItem('intentroute_user');
  localStorage.removeItem('intentroute_token');
  chatHistory = [];
  if (chatMessages) chatMessages.innerHTML = '';
  if (chatGreeting) chatGreeting.style.display = '';
  showAuthScreen();
  showToast('Signed out successfully.');
}


// ============================================================
// NAVIGATION & VIEW SWITCHER
// ============================================================

function initNavigation() {
  // Check hash on load
  const hash = window.location.hash.replace('#', '') || 'chat';
  if (['chat', 'architecture', 'analytics', 'performance'].includes(hash)) {
    switchTab(hash);
  }

  // Check URL pathname for /architecture
  if (window.location.pathname.includes('/architecture')) {
    switchTab('architecture');
  }

  window.addEventListener('hashchange', () => {
    const h = window.location.hash.replace('#', '');
    if (['chat', 'architecture', 'analytics', 'performance'].includes(h)) {
      switchTab(h);
    }
  });
}

function switchTab(tabId) {
  activeTab = tabId;

  // Update tabs
  navTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabId);
  });

  // Update views
  viewChat.classList.toggle('active', tabId === 'chat');
  viewArch.classList.toggle('active', tabId === 'architecture');
  viewAnalytics.classList.toggle('active', tabId === 'analytics');
  viewPerformance.classList.toggle('active', tabId === 'performance');

  const activeView = document.querySelector('.view-content.active');
  if (activeView) {
    activeView.classList.remove('view-entering');
    requestAnimationFrame(() => activeView.classList.add('view-entering'));
  }

  window.location.hash = tabId;

  if (tabId === 'analytics') {
    pollAnalytics();
  }
}


// ============================================================
// CHATBOT INTERFACE (MATCHING USER'S SAMPLE IMAGES)
// ============================================================

function initChatInput() {
  if (clearChatBtn) clearChatBtn.addEventListener('click', clearChatHistory);
  setGreetingQuote();

  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    updateWordCount();
  });

  // Enter to submit, Shift+Enter for newline
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitChatQuery();
    }
  });

  // Send button click
  chatSendBtn.addEventListener('click', submitChatQuery);

  // Think button toggle (if present)
  if (thinkToggleBtn) {
    thinkToggleBtn.addEventListener('click', () => {
      thinkToggleBtn.classList.toggle('active');
      const isActive = thinkToggleBtn.classList.contains('active');
      showToast(isActive ? 'RoBERTa routing intelligence enabled' : 'RoBERTa routing bypassed (direct)');
    });
  }

  // Plus button
  if (chatPlusBtn) {
    chatPlusBtn.addEventListener('click', () => {
      showToast('Additional tools and context attachments ready');
    });
  }
}

function clearChatHistory() {
  if (clearChatBtn?.disabled) return;

  const clearLabel = clearChatBtn?.querySelector('span');
  const clearIcon = clearChatBtn?.querySelector('.clear-bin-icon');

  if (clearChatBtn) {
    clearChatBtn.disabled = true;
    clearChatBtn.classList.add('is-clearing');
    if (clearLabel) clearLabel.textContent = 'Clearing...';
    if (clearIcon) clearIcon.classList.add('bin-opening');
  }
  if (chatMessages) chatMessages.classList.add('is-clearing');

  setTimeout(() => {
    chatHistory = [];
    if (chatMessages) chatMessages.innerHTML = '';
    if (chatMessages) chatMessages.classList.remove('is-clearing');
    if (chatGreeting) chatGreeting.style.display = '';
    setGreetingQuote();
    if (clearChatBtn) {
      clearChatBtn.disabled = false;
      clearChatBtn.classList.remove('is-clearing');
      if (clearIcon) clearIcon.classList.remove('bin-opening');
      if (clearLabel) clearLabel.textContent = 'Clear chat';
    }
    try {
      localStorage.removeItem(chatStorageKey());
    } catch (error) {
      console.warn('Unable to clear chat history:', error);
    }
    showToast('Chat history cleared');
  }, 980);
}

function setGreetingQuote() {
  const quoteEl = document.querySelector('.chat-greeting');
  if (!quoteEl) return;

  const quotes = [
    'Ready when you are.',
    'Ask a sharper question. Find a clearer answer.',
    'Every good solution starts with curiosity.',
    'Think deeply. Move decisively.',
    'Bring the problem. We will find the path.',
    'Complex questions deserve thoughtful answers.',
    'Start anywhere. Discover what is possible.',
    'Good questions make better systems.',
    'Turn uncertainty into your next insight.',
    'Let the right model do the heavy thinking.',
    'A clear answer is closer than it looks.',
    'Make the next question count.',
  ];

  const currentQuote = quoteEl.textContent.trim();
  const availableQuotes = quotes.filter(quote => quote !== currentQuote);
  quoteEl.textContent = availableQuotes[Math.floor(Math.random() * availableQuotes.length)];
}

function scrollChatToBottom(smooth = true) {
  const scrollArea = document.querySelector('.chat-scroll-area');
  if (scrollArea) {
    scrollArea.scrollTo({
      top: scrollArea.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto'
    });
  } else if (chatMessages?.lastElementChild) {
    chatMessages.lastElementChild.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'end' });
  }
}

function initVoiceInput() {
  if (!micBtn) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.addEventListener('click', () => {
      showToast('Voice input is supported in Chrome, Edge, and Safari.');
    });
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  let isRecording = false;

  micBtn.addEventListener('click', () => {
    if (isRecording) {
      recognition.stop();
      isRecording = false;
      micBtn.classList.remove('active');
    } else {
      try {
        recognition.start();
        isRecording = true;
        micBtn.classList.add('active');
        showToast('Listening... Speak now');
      } catch {
        recognition.stop();
        isRecording = false;
      }
    }
  });

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    chatInput.value = (chatInput.value ? chatInput.value + ' ' : '') + transcript;
    chatInput.dispatchEvent(new Event('input'));
    chatInput.focus();
    isRecording = false;
    micBtn.classList.remove('active');
  };

  recognition.onerror = () => {
    isRecording = false;
    micBtn.classList.remove('active');
  };

  recognition.onend = () => {
    isRecording = false;
    micBtn.classList.remove('active');
  };
}

async function submitChatQuery() {
  const query = chatInput.value.trim();
  if (!query || isSubmitting) return;

  isSubmitting = true;
  chatSendBtn.disabled = true;

  // Hide greeting view
  if (chatGreeting) {
    chatGreeting.style.display = 'none';
  }

  // Clear input
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // 1. Render User Message Row (Top-Right deep blue pill - Matching Image 2)
  const turnId = `turn-${Date.now()}`;
  const turnDiv = document.createElement('div');
  turnDiv.className = 'chat-turn';
  turnDiv.id = turnId;

  turnDiv.innerHTML = `
    <div class="user-msg-row">
      <div class="user-bubble">${escapeHtml(query)}</div>
    </div>
    <div class="bot-msg-row">
      <div class="chat-loading-row" id="loading-${turnId}">
        <div class="chat-spinner"></div>
        <span>Evaluating query complexity with RoBERTa...</span>
      </div>
    </div>
  `;

  chatMessages.appendChild(turnDiv);
  scrollChatToBottom(true);

  const t0 = performance.now();

  try {
    const res = await fetch(`${API}/api/ask-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        use_cache: true,
        conversation_history: chatHistory.slice(0, 5).map(h => ({
          question: h.question,
          answer: h.answer,
        })).reverse(),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = typeof err?.detail === 'string'
        ? err.detail
        : JSON.stringify(err?.detail || err);
      throw new Error(detail || 'The server could not process your question.');
    }

    // Stream the response token-by-token
    await streamBotResponse(turnDiv, turnId, query, res, t0);

  } catch (err) {
    renderBotError(turnDiv, turnId, err.message);
  } finally {
    isSubmitting = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
}

async function streamBotResponse(turnDiv, turnId, query, res, t0) {
  const botRow = turnDiv.querySelector('.bot-msg-row');
  if (!botRow) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let metaData = null;
  let rawTokens = '';
  let answerBodyEl = null;
  let metaRendered = false;

  const buildMetaHTML = (meta) => {
    const modelDisplayName = shortModel(meta.model || '');
    const isSimple = meta.complexity === 'Simple';
    const confPct = ((meta.confidence || 0) * 100).toFixed(1);
    return `
      <div class="model-attribution-bar">
        <div class="meta-strip">
          <div class="meta-chip meta-model">
            <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
              <rect x="9" y="9" width="6" height="6"></rect>
              <line x1="9" y1="1" x2="9" y2="4"></line>
              <line x1="15" y1="1" x2="15" y2="4"></line>
              <line x1="9" y1="20" x2="9" y2="23"></line>
              <line x1="15" y1="20" x2="15" y2="23"></line>
              <line x1="20" y1="9" x2="23" y2="9"></line>
              <line x1="20" y1="14" x2="23" y2="14"></line>
              <line x1="1" y1="9" x2="4" y2="9"></line>
              <line x1="1" y1="14" x2="4" y2="14"></line>
            </svg>
            <span class="meta-chip-label">Model</span>
            <span class="meta-chip-value">${escapeHtml(modelDisplayName)}</span>
          </div>
          <span class="meta-chip-divider"></span>
          <div class="meta-chip meta-complexity ${isSimple ? 'is-simple' : 'is-complex'}">
            <span class="meta-status-dot"></span>
            <span class="meta-chip-value">${meta.complexity || '...'}</span>
            <span class="meta-chip-sub">(${confPct}%)</span>
          </div>
          <span class="meta-chip-divider"></span>
          <div class="meta-chip meta-latency">
            <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <span class="meta-chip-value" id="latency-val-${turnId}">...</span>
          </div>
          ${meta.cache_hit ? `
            <span class="meta-chip-divider"></span>
            <div class="meta-chip meta-cache">
              <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              <span class="meta-chip-value">Cached</span>
            </div>
          ` : ''}
        </div>
      </div>
      <div class="bot-answer-body streaming" id="answer-body-${turnId}"></div>
    `;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        if (event.type === 'meta') {
          metaData = event;
          botRow.innerHTML = buildMetaHTML(metaData);
          answerBodyEl = document.getElementById(`answer-body-${turnId}`);
          metaRendered = true;
          scrollChatToBottom(false);

        } else if (event.type === 'token') {
          if (!metaRendered) continue;
          rawTokens += event.text;
          if (answerBodyEl) {
            // Progressive rendering: update every ~25 tokens to avoid layout thrashing
            // while still showing structured content quickly
            if (rawTokens.length % 25 === 0 || rawTokens.endsWith('\n')) {
              answerBodyEl.innerHTML = `<pre class="streaming-pre">${escapeHtml(rawTokens)}</pre>`;
            } else {
              // Ultra-fast path: just append to the pre element directly
              const pre = answerBodyEl.querySelector('.streaming-pre');
              if (pre) {
                pre.textContent = rawTokens;
              } else {
                answerBodyEl.innerHTML = `<pre class="streaming-pre">${escapeHtml(rawTokens)}</pre>`;
              }
            }
          }
          scrollChatToBottom(false);

        } else if (event.type === 'done') {
          const totalMs = event.latency_ms || (performance.now() - t0);
          const latencyDisplay = totalMs >= 1000
            ? `${(totalMs / 1000).toFixed(1)}s`
            : `${Math.round(totalMs)}ms`;

          const latencyEl = document.getElementById(`latency-val-${turnId}`);
          if (latencyEl) latencyEl.textContent = latencyDisplay;

          if (answerBodyEl && rawTokens) {
            const cleanAnswer = cleanModelOutput(rawTokens);
            const formattedAnswer = formatMarkdown(cleanAnswer, metaData?.domain || 'general');
            answerBodyEl.innerHTML = formattedAnswer;
            answerBodyEl.classList.remove('streaming');
            addCodeCopyButtons(botRow);
          }

          const toolbar = document.createElement('div');
          toolbar.className = 'bot-action-bar';
          const cleanAnswerFinal = cleanModelOutput(rawTokens);
          toolbar.innerHTML = `
            <button class="action-btn" title="Copy response" onclick="copyText(this, ${JSON.stringify(cleanAnswerFinal).replace(/"/g, '&quot;')})"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></button>
            <button class="action-btn" title="Good response" onclick="toggleFeedback(this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg></button>
            <button class="action-btn" title="Bad response" onclick="toggleFeedback(this)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg></button>
            <button class="action-btn" title="Regenerate" onclick="regenerateQuery(${JSON.stringify(query).replace(/"/g, '&quot;')})"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg></button>
          `;
          botRow.appendChild(toolbar);

          if (metaData) {
            const historyEntry = {
              ...metaData,
              question: query,
              answer: cleanAnswerFinal,
              total_ms: totalMs,
            };
            updateSessionStats(historyEntry);
            chatHistory.unshift(historyEntry);
            saveChatHistory();
          }
          scrollChatToBottom(true);

        } else if (event.type === 'error') {
          renderBotError(turnDiv, turnId, event.detail || 'The answer service could not complete this request.');
          return;
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

function renderBotResponse(turnDiv, turnId, query, data, clientMs) {
  const botRow = turnDiv.querySelector('.bot-msg-row');
  if (!botRow) return;

  const modelDisplayName = shortModel(data.model);
  const isSimple = data.complexity === 'Simple';
  const confPct = (data.confidence * 100).toFixed(1);
  const latencyDisplay = data.total_ms >= 1000
    ? `${(data.total_ms / 1000).toFixed(1)}s`
    : `${Math.round(data.total_ms)}ms`;

  // Format cleaned answer with code highlighting and markdown paragraphs.
  const cleanAnswer = cleanModelOutput(data.answer);
  const formattedAnswer = formatMarkdown(cleanAnswer, data.domain);

  botRow.innerHTML = `
    <!-- Human-Crafted Model Attribution Strip -->
    <div class="model-attribution-bar">
      <div class="meta-strip">
        <div class="meta-chip meta-model">
          <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
            <rect x="9" y="9" width="6" height="6"></rect>
            <line x1="9" y1="1" x2="9" y2="4"></line>
            <line x1="15" y1="1" x2="15" y2="4"></line>
            <line x1="9" y1="20" x2="9" y2="23"></line>
            <line x1="15" y1="20" x2="15" y2="23"></line>
            <line x1="20" y1="9" x2="23" y2="9"></line>
            <line x1="20" y1="14" x2="23" y2="14"></line>
            <line x1="1" y1="9" x2="4" y2="9"></line>
            <line x1="1" y1="14" x2="4" y2="14"></line>
          </svg>
          <span class="meta-chip-label">Model</span>
          <span class="meta-chip-value">${escapeHtml(modelDisplayName)}</span>
        </div>

        <span class="meta-chip-divider"></span>

        <div class="meta-chip meta-complexity ${isSimple ? 'is-simple' : 'is-complex'}">
          <span class="meta-status-dot"></span>
          <span class="meta-chip-value">${data.complexity}</span>
          <span class="meta-chip-sub">(${confPct}%)</span>
        </div>

        <span class="meta-chip-divider"></span>

        <div class="meta-chip meta-latency">
          <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <span class="meta-chip-value">${latencyDisplay}</span>
        </div>

        ${data.cache_hit ? `
          <span class="meta-chip-divider"></span>
          <div class="meta-chip meta-cache">
            <svg class="meta-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
            <span class="meta-chip-value">Cached</span>
          </div>
        ` : ''}
      </div>
    </div>

    <!-- Answer Body -->
    <div class="bot-answer-body">
      ${formattedAnswer}
    </div>

    <!-- Action Toolbar Under Answer (Matching Image 2) -->
    <div class="bot-action-bar">
      <!-- Copy button -->
      <button class="action-btn" title="Copy response" onclick="copyText(this, ${JSON.stringify(cleanAnswer).replace(/"/g, '&quot;')})">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>

      <!-- Thumbs up -->
      <button class="action-btn" title="Good response" onclick="toggleFeedback(this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
        </svg>
      </button>

      <!-- Thumbs down -->
      <button class="action-btn" title="Bad response" onclick="toggleFeedback(this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
        </svg>
      </button>

      <!-- Regenerate button -->
      <button class="action-btn" title="Regenerate" onclick="regenerateQuery(${JSON.stringify(query).replace(/"/g, '&quot;')})">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="23 4 23 10 17 10"></polyline>
          <polyline points="1 20 1 14 7 14"></polyline>
          <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
        </svg>
      </button>

      <!-- More options -->
      <button class="action-btn" title="More options" onclick="showMoreOptions(${JSON.stringify(data).replace(/"/g, '&quot;')})">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="1"></circle>
          <circle cx="19" cy="12" r="1"></circle>
          <circle cx="5" cy="12" r="1"></circle>
        </svg>
      </button>
    </div>
  `;

  addCodeCopyButtons(botRow);

  scrollChatToBottom(true);
}

function addCodeCopyButtons(container) {
  container.querySelectorAll('.bot-answer-body pre').forEach(pre => {
    const copyButton = document.createElement('button');
    copyButton.className = 'code-copy-btn';
    copyButton.type = 'button';
    copyButton.title = 'Copy code';
    copyButton.setAttribute('aria-label', 'Copy code');
    copyButton.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;
    copyButton.addEventListener('click', () => copyText(copyButton, pre.querySelector('code')?.textContent || ''));
    pre.appendChild(copyButton);
  });
}

function renderBotError(turnDiv, turnId, message) {
  const botRow = turnDiv.querySelector('.bot-msg-row');
  if (!botRow) return;

  const friendlyMessage = formatUserError(message);
  botRow.innerHTML = `
    <div class="model-attribution-bar">
      <span class="badge" style="background:rgba(239,68,68,0.12);color:#ef4444;">Unable to answer</span>
    </div>
    <div class="bot-answer-body" style="color:#ef4444;">
      ${escapeHtml(friendlyMessage)}
      <div style="margin-top:10px;color:var(--text-secondary);font-size:0.85rem;">
        Please try again in a moment.
      </div>
    </div>
  `;
}

function formatUserError(error) {
  let detail = error;
  if (error && typeof error === 'object') {
    detail = error.detail || error.message || error.error;
    if (typeof detail === 'object') {
      detail = detail.msg || detail.message || JSON.stringify(detail);
    }
  }

  const text = String(detail || '').toLowerCase();
  console.error('QA request failed:', error);

  if (text.includes('failed to fetch') || text.includes('networkerror') || text.includes('connection')) {
    return 'The answer service is not reachable right now.';
  }
  if (text.includes('timed out') || text.includes('timeout')) {
    return 'This question is taking longer than expected to answer.';
  }
  if (text.includes('ollama') || text.includes('model')) {
    return 'The selected answer model is temporarily unavailable.';
  }
  return 'We could not complete that answer right now.';
}

function regenerateQuery(query) {
  chatInput.value = query;
  submitChatQuery();
}

function toggleFeedback(btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll('.action-btn').forEach(b => {
    if (b !== btn) b.classList.remove('active');
  });
  btn.classList.toggle('active');
  showToast(btn.classList.contains('active') ? 'Feedback received' : 'Feedback cleared');
}

function copyText(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    btn.classList.add('active');
    showToast('Copied to clipboard');
    setTimeout(() => btn.classList.remove('active'), 2000);
  }).catch(() => {
    showToast('Unable to copy text');
  });
}

function showMoreOptions(data) {
  const details = `Model: ${data.model}\nComplexity: ${data.complexity} (${(data.confidence*100).toFixed(1)}%)\nDomain: ${data.domain}\nTotal Latency: ${data.total_ms} ms\nRoBERTa: ${data.classify_ms} ms`;
  navigator.clipboard.writeText(details).then(() => {
    showToast('Routing metadata copied to clipboard');
  });
}


// ============================================================
// VIEW 2: ROUTING ARCHITECTURE & LIVE SIMULATOR
// ============================================================

function initSimulator() {
  if (!simRunBtn) return;

  simRunBtn.addEventListener('click', runSimulator);
  simQueryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      runSimulator();
    }
  });

  if (simClearBtn) {
    simClearBtn.addEventListener('click', () => {
      simQueryInput.value = '';
      simResultBox.innerHTML = `
        <div class="sim-placeholder">
          Enter any prompt above and click "Test Route" to observe live RoBERTa confidence scores and routing decisions.
        </div>
      `;
    });
  }
}

async function runSimulator() {
  const query = simQueryInput.value.trim();
  if (!query) {
    showToast('Please type a prompt to simulate.');
    return;
  }

  simRunBtn.disabled = true;
  simResultBox.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;color:var(--text-secondary);padding:10px 0;">
      <div class="chat-spinner"></div>
      <span>RoBERTa classifier performing inference...</span>
    </div>
  `;

  try {
    const res = await fetch(`${API}/api/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query }),
    });

    if (!res.ok) throw new Error('Classification request failed.');

    const data = await res.json();
    const confPct = (data.confidence * 100).toFixed(1);
    const simplePct = (data.simple_prob * 100).toFixed(1);
    const complexPct = (data.complex_prob * 100).toFixed(1);

    const targetModel = (() => {
      if (data.complexity === 'Simple') return 'Qwen2.5:0.5B';
      switch (data.domain) {
        case 'code': return 'Qwen2.5-Coder:3B';
        case 'math': return 'Qwen3:8B';
        default:     return 'Qwen2.5-3B';
      }
    })();

    simResultBox.innerHTML = `
      <div class="sim-output">
        <div class="sim-meta-row">
          <div>
            <span style="font-size:0.8rem;color:var(--text-muted);display:block;margin-bottom:4px;">Predicted Complexity</span>
            <span class="badge ${data.complexity === 'Simple' ? 'badge-simple' : 'badge-complex'}" style="font-size:0.88rem;padding:4px 14px;">
              ${data.complexity} (${confPct}%)
            </span>
          </div>

          <div>
            <span style="font-size:0.8rem;color:var(--text-muted);display:block;margin-bottom:4px;">Assigned Model</span>
            <span style="font-size:0.92rem;font-weight:600;color:var(--accent-blue);">
              ${escapeHtml(targetModel)}
            </span>
          </div>

          <div>
            <span style="font-size:0.8rem;color:var(--text-muted);display:block;margin-bottom:4px;">Domain</span>
            <span class="badge badge-meta" style="font-size:0.82rem;text-transform:capitalize;">
              ${escapeHtml(data.domain)}
            </span>
          </div>

          <div>
            <span style="font-size:0.8rem;color:var(--text-muted);display:block;margin-bottom:4px;">Inference Latency</span>
            <span style="font-size:0.88rem;font-family:'JetBrains Mono', monospace;color:var(--text-primary);">
              ${data.latency_ms.toFixed(1)} ms
            </span>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:6px;">
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-muted);">
            <span>Simple: ${simplePct}%</span>
            <span>Complex: ${complexPct}%</span>
          </div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill" style="width: ${data.complexity === 'Simple' ? simplePct : complexPct}%;"></div>
          </div>
        </div>
      </div>
    `;

  } catch (err) {
    simResultBox.innerHTML = `<div style="color:#ef4444;font-size:0.88rem;">${escapeHtml(formatUserError(err))}</div>`;
  } finally {
    simRunBtn.disabled = false;
  }
}


// ============================================================
// VIEW 3: SYSTEM ANALYTICS & CACHE
// ============================================================

function initAnalytics() {
  if (cacheFlushBtn) {
    cacheFlushBtn.addEventListener('click', async () => {
      try {
        await fetch(`${API}/api/cache`, { method: 'DELETE' });
        sessionStats.cacheHits = 0;
        showToast('Cache cleared successfully');
        pollAnalytics();
      } catch {
        showToast('Failed to flush cache');
      }
    });
  }

  // Auto-poll health and stats periodically
  setInterval(pollAnalytics, 15000);
}

async function pollAnalytics() {
  try {
    const [healthRes, metricsRes] = await Promise.all([
      fetch(`${API}/api/health`),
      fetch(`${API}/api/metrics`).catch(() => null),
    ]);

    if (healthRes.ok) {
      const hData = await healthRes.json();
      if (metricCpu) metricCpu.textContent = `${hData.cpu_percent.toFixed(1)}%`;
      if (metricRam) metricRam.textContent = `${hData.ram_percent.toFixed(1)}% (${hData.ram_used_mb} MB)`;
      if (metricCacheBack && hData.cache) {
        metricCacheBack.textContent = hData.cache.detail || 'In-Memory Cache';
      }
    }

    if (metricsRes && metricsRes.ok) {
      const mData = await metricsRes.json();
      if (metricQueries) metricQueries.textContent = mData.total_queries;
      if (metricLatency) metricLatency.textContent = `${mData.avg_total_ms.toFixed(0)} ms`;
      if (metricCache && mData.cache_stats) {
        metricCache.textContent = `${mData.cache_stats.hit_rate}%`;
      }
    }
  } catch { /* silent */ }

  renderAnalyticsHistory();
}

function updateSessionStats(data) {
  sessionStats.queries++;
  sessionStats.totalMs += data.total_ms;
  if (data.cache_hit) sessionStats.cacheHits++;

  if (metricQueries) metricQueries.textContent = sessionStats.queries;
  if (metricLatency) metricLatency.textContent = `${(sessionStats.totalMs / sessionStats.queries).toFixed(0)} ms`;
  if (metricCache) {
    metricCache.textContent = `${((sessionStats.cacheHits / sessionStats.queries) * 100).toFixed(0)}%`;
  }

  renderAnalyticsHistory();
}

function renderAnalyticsHistory() {
  if (!historyList) return;

  if (chatHistory.length === 0) {
    historyList.innerHTML = '<div class="history-empty">No query history recorded yet.</div>';
    return;
  }

  historyList.innerHTML = chatHistory.slice(0, 20).map(item => `
    <div class="history-item" onclick="loadQueryIntoChat(${JSON.stringify(item.question).replace(/"/g, '&quot;')})">
      <div class="history-q">${escapeHtml(item.question)}</div>
      <div class="history-meta">
        <span class="badge ${item.complexity === 'Simple' ? 'badge-simple' : 'badge-complex'}">${item.complexity}</span>
        <span>${escapeHtml(shortModel(item.model))}</span>
        <span>${item.total_ms.toFixed(0)} ms</span>
        ${item.cache_hit ? '<span class="badge badge-cache">Cached</span>' : ''}
      </div>
    </div>
  `).join('');
}

function loadQueryIntoChat(query) {
  switchTab('chat');
  chatInput.value = query;
  chatInput.focus();
}


// ============================================================
// UTILITIES & HELPERS
// ============================================================

let toastTimer;
function showToast(msg) {
  if (!toastEl) return;
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 3000);
}

function countWords(text) {
  if (!text || !text.trim()) return 0;
  return text.trim().split(/\s+/).length;
}

function updateWordCount() {
  const counter = document.getElementById('word-count-display');
  if (!counter) return;
  // Word limit removed and display kept hidden
  counter.style.display = 'none';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function shortModel(model) {
  if (!model) return 'Optimal Router';
  const map = {
    'local-arithmetic':                  'Local Arithmetic',
    'local-word-problem':                'Local Math Solver',
    'qwen2.5:1.5b':                     'Qwen2.5:0.5B',
    'Qwen2.5:1.5B':                     'Qwen2.5:0.5B',
    'qwen3:8b':                          'Qwen3:8B',
    'qwen2.5-coder:3b':                  'Qwen2.5-Coder:3B',
    'qwen2.5:3b':                        'Ministral-3:3B',
    'ministral-3:3b':                    'Ministral-3:3B',
  };
  return map[model] || model.split('/').pop();
}

function formatMarkdown(text, domain = '') {
  if (!text) return '';

  // 1. Preserve and extract fenced code blocks ```lang ... ```
  const codeBlocks = [];
  let processed = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const placeholder = `%%FENCED_CODE_${codeBlocks.length}%%`;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return `\n\n${placeholder}\n\n`;
  });

  // Code-specialist responses may omit Markdown fences. Keep explanations readable.
  if (domain === 'code' && codeBlocks.length === 0 && !processed.includes('<pre>')) {
    const lines = processed.split('\n');
    const explanationIndex = lines.findIndex(line => /^(explanation|output|usage)\s*:/i.test(line.trim()));
    const codeLines = explanationIndex > 0 ? lines.slice(0, explanationIndex) : lines;
    const explanationLines = explanationIndex > 0 ? lines.slice(explanationIndex) : [];
    const codePart = `<pre><code>${escapeHtml(codeLines.join('\n').trim())}</code></pre>`;
    const explPart = explanationLines.length ? `\n\n${explanationLines.join('\n')}` : '';
    const placeholder = `%%FENCED_CODE_${codeBlocks.length}%%`;
    codeBlocks.push(codePart);
    processed = `${placeholder}${explPart}`;
  }

  // 2. Inline markdown formatting
  function formatInline(str) {
    if (!str) return '';
    let res = escapeHtml(str);
    // Inline code `code`
    res = res.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold **text** or __text__
    res = res.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    res = res.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
    // Italic *text* or _text_
    res = res.replace(/(^|[^\*])\*([^*\n]+)\*([^\*]|$)/g, '$1<em>$2</em>$3');
    // Make common plain-text equations easier to scan.
    res = res.replace(/\^([0-9+\-])/g, '<sup>$1</sup>');
    res = res.replace(/(\b(?:sqrt|sin|cos|tan|log|ln))\s*\(([^()]*)\)/gi, '<span class="math-function">$1($2)</span>');
    return res;
  }

  // 3. Helper to parse a markdown table row into cells safely
  function parseTableCells(rowStr) {
    let clean = rowStr.trim();
    if (clean.startsWith('|')) clean = clean.slice(1);
    if (clean.endsWith('|')) clean = clean.slice(0, -1);
    const cells = [];
    let current = '';
    let inBackticks = false;
    let escaped = false;
    for (let k = 0; k < clean.length; k++) {
      const ch = clean[k];
      if (ch === '`' && !escaped) {
        inBackticks = !inBackticks;
        current += ch;
      } else if (ch === '\\' && !escaped) {
        escaped = true;
      } else if (ch === '|' && !escaped && !inBackticks) {
        cells.push(current.trim());
        current = '';
      } else {
        if (escaped) current += '\\';
        current += ch;
        escaped = false;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  // Helper to build a structured markdown table
  function buildTable(headerLine, delimLine, rowLines) {
    const rawDelims = parseTableCells(delimLine);
    const headerCells = parseTableCells(headerLine);
    const alignments = rawDelims.map(d => {
      const s = d.trim();
      if (s.startsWith(':') && s.endsWith(':')) return 'center';
      if (s.endsWith(':')) return 'right';
      return 'left';
    });
    const colCount = Math.max(alignments.length, headerCells.length, 1);

    let thead = '<thead><tr>';
    for (let c = 0; c < colCount; c++) {
      const content = headerCells[c] !== undefined && headerCells[c] !== '' ? formatInline(headerCells[c]) : '&nbsp;';
      const align = alignments[c] || 'left';
      thead += `<th style="text-align:${align}">${content}</th>`;
    }
    thead += '</tr></thead>';

    let tbody = '<tbody>';
    for (const rLine of rowLines) {
      if (!rLine.trim()) continue;
      const rowCells = parseTableCells(rLine);
      tbody += '<tr>';
      for (let c = 0; c < colCount; c++) {
        const content = rowCells[c] !== undefined && rowCells[c] !== '' ? formatInline(rowCells[c]) : '&nbsp;';
        const align = alignments[c] || 'left';
        tbody += `<td style="text-align:${align}">${content}</td>`;
      }
      tbody += '</tr>';
    }
    tbody += '</tbody>';

    return `<div class="table-wrapper"><table class="markdown-table">${thead}${tbody}</table></div>`;
  }

  // 4. Line-by-line block tokenization
  const isDelimRow = (line) => {
    const trimmed = line.trim();
    if (!trimmed.includes('-')) return false;
    let inner = trimmed;
    if (inner.startsWith('|')) inner = inner.slice(1);
    if (inner.endsWith('|')) inner = inner.slice(0, -1);
    const parts = inner.split('|');
    if (parts.length < 2) return false;
    return parts.every(part => /^\s*:?-+:?\s*$/.test(part));
  };
  const isPipeRow = (line) => line.includes('|');
  const isHr = (line) => /^\s*(?:---|\*\*\*|___)\s*$/.test(line);
  const isHeading = (line) => /^\s*(#{1,6})\s+(.+)$/.test(line);
  const isUList = (line) => /^\s*[-*+]\s+(.+)$/.test(line);
  const isOList = (line) => /^\s*\d+\.\s+(.+)$/.test(line);

  const lines = processed.split('\n');
  const blocks = [];
  let i = 0;

  // Helper: detect "Label: content" pattern (labeled category line)
  // Matches lines like: "Classification: Email spam detection, fraud detection."
  // Label must be 1-5 words (no numbers only), followed by colon and content
  const isLabeledLine = (line) => {
    const m = line.match(/^([A-Z][A-Za-z\s]{0,40}?):\s+(.+)$/);
    return m && m[1].trim().split(/\s+/).length <= 6;
  };

  // Helper: detect a section heading line (single word or short phrase ending with colon, no inline content)
  // e.g. "Example Applications:" or "Key Features:"
  const isSectionLabel = (line) => {
    const m = line.match(/^([A-Z][A-Za-z\s]{0,40}?):\s*$/);
    return m && m[1].trim().split(/\s+/).length <= 5;
  };

  while (i < lines.length) {
    const line = lines[i];

    // Empty line
    if (!line.trim()) {
      i++;
      continue;
    }

    // Code block placeholder
    if (/^%%FENCED_CODE_\d+%%$/.test(line.trim())) {
      blocks.push(line.trim());
      i++;
      continue;
    }

    // Markdown Table: header line followed by delimiter line
    if (i + 1 < lines.length && isDelimRow(lines[i + 1]) && isPipeRow(line)) {
      const headerLine = line;
      const delimLine = lines[i + 1];
      const rowLines = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && isPipeRow(lines[i]) && !isDelimRow(lines[i])) {
        rowLines.push(lines[i]);
        i++;
      }
      blocks.push(buildTable(headerLine, delimLine, rowLines));
      continue;
    }

    // Horizontal Rule
    if (isHr(line)) {
      blocks.push('<hr class="markdown-divider" />');
      i++;
      continue;
    }

    // Heading
    const headingMatch = line.match(/^\s*(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const headingText = headingMatch[2];
      const hTag = level === 1 ? 'h2' : (level === 2 ? 'h3' : 'h4');
      blocks.push(`<${hTag} class="markdown-heading">${formatInline(headingText)}</${hTag}>`);
      i++;
      continue;
    }

    // Unordered List
    if (isUList(line)) {
      let listHtml = '<ul class="markdown-list">';
      while (i < lines.length && isUList(lines[i])) {
        const m = lines[i].match(/^\s*[-*+]\s+(.+)$/);
        listHtml += `<li>${formatInline(m[1])}</li>`;
        i++;
      }
      listHtml += '</ul>';
      blocks.push(listHtml);
      continue;
    }

    // Ordered List
    if (isOList(line)) {
      let listHtml = '<ol class="markdown-list">';
      while (i < lines.length && isOList(lines[i])) {
        const m = lines[i].match(/^\s*\d+\.\s+(.+)$/);
        listHtml += `<li>${formatInline(m[1])}</li>`;
        i++;
      }
      listHtml += '</ol>';
      blocks.push(listHtml);
      continue;
    }

    // Section heading label (e.g. "Example Applications:") — standalone colon-terminated label
    if (isSectionLabel(line)) {
      const label = line.replace(/:\s*$/, '').trim();
      blocks.push(`<div class="answer-section-label">${escapeHtml(label)}</div>`);
      i++;
      continue;
    }

    // Labeled-category line (e.g. "Classification: Email spam detection, fraud detection.")
    if (isLabeledLine(line)) {
      let catHtml = '<div class="answer-category-group">';
      while (i < lines.length && lines[i].trim() && isLabeledLine(lines[i])) {
        const m = lines[i].match(/^([A-Z][A-Za-z\s]{0,40}?):\s+(.+)$/);
        catHtml += `<div class="answer-category-row"><span class="answer-cat-label">${escapeHtml(m[1].trim())}:</span><span class="answer-cat-value">${formatInline(m[2].trim())}</span></div>`;
        i++;
      }
      catHtml += '</div>';
      blocks.push(catHtml);
      continue;
    }

    // Regular Paragraph (or Callout)
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^%%FENCED_CODE_\d+%%$/.test(lines[i].trim()) &&
      !isHr(lines[i]) &&
      !isHeading(lines[i]) &&
      !isUList(lines[i]) &&
      !isOList(lines[i]) &&
      !(i + 1 < lines.length && isDelimRow(lines[i + 1]) && isPipeRow(lines[i]))
    ) {
      paraLines.push(lines[i]);
      i++;
    }

    const paraText = paraLines.join('\n');
    const isCallout = /(?:exact|final)\s+answer|answer\s*:|important\s+note|notes?\s*:/i.test(paraText);
    const formattedContent = paraLines.map(l => formatInline(l)).join('<br/>');

    if (isCallout) {
      blocks.push(`<div class="answer-callout">${formattedContent}</div>`);
    } else {
      blocks.push(`<p>${formattedContent}</p>`);
    }
  }

  // 5. Restore code block placeholders
  let html = blocks.join('\n');
  codeBlocks.forEach((codeHtml, idx) => {
    html = html.replace(`%%FENCED_CODE_${idx}%%`, codeHtml);
  });

  return html;
}

function cleanModelOutput(text) {
  if (!text) return '';
  return normalizeMathNotation(String(text))
    .replace(/\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, '')
    .replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/\r/g, '')
    .trim();
}

function normalizeMathNotation(text) {
  return text
    // Display math delimiters $$...$$ and $...$
    .replace(/\$\$([^]*?)\$\$/g, '$1')
    .replace(/\$([^$\n]+)\$/g, '$1')
    // Inline math \(...\) and display \[...\]
    .replace(/\\\(([^]*?)\\\)/g, '$1')
    .replace(/\\\[([^]*?)\\\]/g, '$1')
    // \boxed{...} — wrap in brackets to highlight the answer
    .replace(/\\boxed\s*\{([^{}]*)\}/g, '[ $1 ]')
    // \frac{a}{b} → (a)/(b)
    .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)')
    // Models sometimes omit the \frac command: {a}{b} or {a}{sqrt(...)}.
    .replace(/\{([^{}]+)\}\s*\{((?:sqrt|sin|cos|tan|log|ln)\s*\([^{}]*\)|[^{}]+)\}/gi, '($1) / ($2)')
    // \left( \right) wrappers — remove, keep inner parens
    .replace(/\\left\s*\(/g, '(').replace(/\\right\s*\)/g, ')')
    .replace(/\\left\s*\[/g, '[').replace(/\\right\s*\]/g, ']')
    .replace(/\\left\s*\{/g, '{').replace(/\\right\s*\}/g, '}')
    .replace(/\\left\./g, '').replace(/\\right\./g, '')
    // \sqrt{...} → sqrt(...)
    .replace(/\\sqrt\s*\{([^{}]*)\}/g, 'sqrt($1)')
    .replace(/\\sqrt\s+/g, 'sqrt ')
    .replace(/\bsqrt\s*\{([^{}]*)\}/gi, 'sqrt($1)')
    // Powers and subscripts
    .replace(/\^\{([^{}]*)\}/g, '^$1')
    .replace(/_\{([^{}]*)\}/g, '_$1')
    // Math operators → readable symbols
    .replace(/\\cdot/g, ' \u00d7 ')
    .replace(/\\times/g, ' \u00d7 ')
    .replace(/\\div/g, ' \u00f7 ')
    .replace(/\\pm/g, ' \u00b1 ')
    .replace(/\\neq/g, ' \u2260 ')
    .replace(/\\leq/g, ' \u2264 ')
    .replace(/\\geq/g, ' \u2265 ')
    .replace(/\\approx/g, ' \u2248 ')
    .replace(/\\equiv/g, ' \u2261 ')
    .replace(/\\infty/g, '\u221e')
    .replace(/\\rightarrow/g, ' \u2192 ')
    .replace(/\\Rightarrow/g, ' \u21d2 ')
    // Greek letters
    .replace(/\\alpha/g, '\u03b1').replace(/\\beta/g, '\u03b2').replace(/\\gamma/g, '\u03b3')
    .replace(/\\delta/g, '\u03b4').replace(/\\Delta/g, '\u0394').replace(/\\epsilon/g, '\u03b5')
    .replace(/\\theta/g, '\u03b8').replace(/\\lambda/g, '\u03bb').replace(/\\mu/g, '\u03bc')
    .replace(/\\pi/g, '\u03c0').replace(/\\Pi/g, '\u03a0').replace(/\\sigma/g, '\u03c3')
    .replace(/\\Sigma/g, '\u03a3').replace(/\\phi/g, '\u03c6').replace(/\\omega/g, '\u03c9')
    // \text{...}, \mathrm{...}, \mathbf{...} → just the content
    .replace(/\\(?:text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}/g, '$1')
    // \begin{...}...\end{...} environment delimiters
    .replace(/\\begin\{[^}]+\}/g, '').replace(/\\end\{[^}]+\}/g, '')
    .replace(/\\hline/g, '').replace(/\\\\(?=\s*$)/gm, '')
    // Remaining bare \command (not a known symbol) — strip
    .replace(/\\[a-zA-Z]+\*/g, '').replace(/\\[a-zA-Z]+/g, '')
    // Collapse multiple spaces
    .replace(/ {2,}/g, ' ')
    .trim();
}
