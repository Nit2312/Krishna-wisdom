// Global variables
let messageCount = 0;
let searchCount = 0;
let isSystemInitialized = false;

// DOM elements
const statusIndicator = document.getElementById('statusIndicator');
const statusDot = statusIndicator.querySelector('.status-dot');
const statusText = statusIndicator.querySelector('.status-text');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const initBtn = document.getElementById('initBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const themeToggle = document.getElementById('themeToggle');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    checkSystemStatus();
    setupEventListeners();
    setupMobileSidebar();
    setupSidebarToggle();
    setupNewChat();
});

// ========== THEME MANAGEMENT ==========
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = savedTheme ? savedTheme === 'dark' : prefersDark;
    
    if (isDark) {
        document.documentElement.classList.add('dark-mode');
        updateThemeIcon(true);
    } else {
        document.documentElement.classList.remove('dark-mode');
        updateThemeIcon(false);
    }
    
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeIcon(isDark);
}

function updateThemeIcon(isDark) {
    if (!themeToggle) return;
    const icon = themeToggle.querySelector('i');
    if (icon) {
        icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Setup event listeners
function setupEventListeners() {
    messageInput.addEventListener('input', handleInputChange);
    messageInput.addEventListener('keypress', handleKeyPress);
}

// Handle input changes
function handleInputChange() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || !isSystemInitialized;
}

// Handle keyboard events
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        updateSystemStatus(data);

        if (!data.initialized) {
            initializeSystem();
        }
    } catch (error) {
        console.error('Error checking system status:', error);
        updateStatus('error', 'Connection failed');
    }
}

// Update system status
function updateSystemStatus(data) {
    isSystemInitialized = data.initialized;
    
    if (isSystemInitialized) {
        updateStatus('connected', 'Connected');
        if (initBtn) {
            initBtn.innerHTML = '<i class="fas fa-check"></i> System Ready';
            initBtn.disabled = true;
            initBtn.classList.add('btn-primary');
            initBtn.classList.remove('btn-secondary');
        }
    } else {
        updateStatus('disconnected', 'Not initialized');
        if (initBtn) {
            initBtn.innerHTML = '<i class="fas fa-play"></i> Initialize System';
            initBtn.disabled = false;
            initBtn.classList.remove('btn-primary');
            initBtn.classList.add('btn-secondary');
        }
    }
    
    handleInputChange();
}

// Update status indicator
function updateStatus(status, message) {
    statusDot.className = 'status-dot';
    
    switch(status) {
        case 'connected':
            statusDot.classList.add('connected');
            break;
        case 'initializing':
            statusDot.classList.add('initializing');
            break;
        case 'error':
            // Keep default red color
            break;
    }
    
    statusText.textContent = message;
}

// Initialize system
async function initializeSystem() {
    if (isSystemInitialized) return;
    
    updateStatus('initializing', 'Initializing...');
    if (initBtn) {
        initBtn.disabled = true;
        initBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';
    }
    
    try {
        const response = await fetch('/api/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateSystemStatus({
                initialized: true,
                total_cases: data.total_cases,
                total_chunks: data.total_chunks
            });
            
            showNotification('System initialized successfully!', 'success');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        console.error('Error initializing system:', error);
        updateStatus('error', 'Initialization failed');
        if (initBtn) {
            initBtn.disabled = false;
            initBtn.innerHTML = '<i class="fas fa-play"></i> Initialize System';
        }
        showNotification('Failed to initialize system: ' + error.message, 'error');
    }
}

function setupMobileSidebar() {
    const media = window.matchMedia('(max-width: 768px)');
    const detailsList = document.querySelectorAll('.sidebar .collapsible');

    const applyState = () => {
        if (media.matches) {
            detailsList.forEach((detail) => {
                detail.removeAttribute('open');
            });
        }
    };

    applyState();
    media.addEventListener('change', applyState);
}

function setupSidebarToggle() {
    const toggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    const appContainer = document.querySelector('.app-container');
    if (!toggle || !sidebar || !appContainer) return;

    const media = window.matchMedia('(max-width: 768px)');
    const storageKey = 'sidebarCollapsed';

    const applyDesktopState = () => {
        if (media.matches) {
            appContainer.classList.remove('sidebar-collapsed');
            return;
        }
        const isCollapsed = localStorage.getItem(storageKey) === 'true';
        appContainer.classList.toggle('sidebar-collapsed', isCollapsed);
    };

    applyDesktopState();
    media.addEventListener('change', () => {
        sidebar.classList.remove('open');
        applyDesktopState();
    });

    toggle.addEventListener('click', function() {
        if (media.matches) {
            sidebar.classList.toggle('open');
            return;
        }
        const nextCollapsed = !appContainer.classList.contains('sidebar-collapsed');
        appContainer.classList.toggle('sidebar-collapsed', nextCollapsed);
        localStorage.setItem(storageKey, String(nextCollapsed));
    });
}

function setupNewChat() {
    const btn = document.querySelector('.new-chat-btn');
    if (!btn) return;
    btn.addEventListener('click', function() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        chatMessages.innerHTML = '';
        const welcome = document.createElement('div');
        welcome.className = 'welcome-message';
        welcome.innerHTML = `
            <div class="welcome-icon"><i class="fas fa-om"></i></div>
            <h2>Seek Divine Wisdom from Krishna's Teachings</h2>
            <p>I am a devoted servant of Shri Krishna. Ask me about the divine teachings found in the Bhagavad Gita, Upanishads, Mahabharata, and Srimad Bhagavatam. All my answers come solely from these sacred texts.</p>
            <div class="welcome-suggestions">
                <button type="button" class="suggestion-chip" onclick="sendExampleMessage('What does the Bhagavad Gita teach about devotion to Krishna?')">Krishna devotion teachings</button>
                <button type="button" class="suggestion-chip" onclick="sendExampleMessage('Explain the path of Bhakti as described in the sacred texts')">Bhakti Yoga path</button>
                <button type="button" class="suggestion-chip" onclick="sendExampleMessage('What is the nature of the eternal soul according to Krishna')">Nature of the soul</button>
            </div>
        `;
        chatMessages.appendChild(welcome);
        messageInput.focus();
    });
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || !isSystemInitialized) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    messageInput.value = '';
    handleInputChange();
    
    // Update counters
    messageCount++;
    searchCount++;
    updateCounters();
    
    // Show loading
    showLoading(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message
            })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status} ${response.statusText}`);
        }
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Add AI response to chat (pass question for Verify answer)
        addMessage(data.response, 'assistant', data.sources || [], false, data.retrieval_metrics, message);
        
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('Sorry, I encountered an error: ' + error.message, 'assistant', [], true);
    } finally {
        showLoading(false);
    }
}

// Send example message
function sendExampleMessage(message) {
    messageInput.value = message;
    handleInputChange();
    sendMessage();
}

// Add message to chat
function addMessage(content, sender, sources = [], isError = false, retrievalMetrics = null, question = null) {
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-hands-praying"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    if (isError) {
        bubbleDiv.style.background = '#fef2f2';
        bubbleDiv.style.color = '#991b1b';
        bubbleDiv.style.borderColor = '#fecaca';
    }
    
    const formattedContent = formatAIResponse(content);
    bubbleDiv.innerHTML = formattedContent;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();
    
    contentDiv.appendChild(bubbleDiv);
    contentDiv.appendChild(timeDiv);
    
    if (sender === 'assistant' && !isError && question) {
        const verifyWrap = document.createElement('div');
        verifyWrap.className = 'verify-answer-wrap';
        const verifyBtn = document.createElement('button');
        verifyBtn.type = 'button';
        verifyBtn.className = 'verify-answer-btn';
        verifyBtn.innerHTML = '<i class="fas fa-shield-alt"></i> Verify answer';
        verifyBtn.dataset.question = question;
        verifyBtn.dataset.response = content;
        verifyBtn.dataset.sources = JSON.stringify(sources || []);
        verifyBtn.addEventListener('click', handleVerifyAnswer);
        verifyWrap.appendChild(verifyBtn);
        contentDiv.appendChild(verifyWrap);
    }
    
    if (sources && sources.length > 0) {
        const sourcesDiv = createSourcesSection(sources);
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format AI response for better readability
function formatAIResponse(content) {
    const escaped = escapeHtml(content || '');
    const lines = escaped.split('\n');

    const blocks = [];
    let paragraphLines = [];
    let ulItems = [];
    let olItems = [];

    const flushParagraph = () => {
        if (!paragraphLines.length) return;
        const text = paragraphLines.join('\n').trim();
        if (text) blocks.push({ type: 'p', text });
        paragraphLines = [];
    };
    const flushUl = () => {
        if (!ulItems.length) return;
        blocks.push({ type: 'ul', items: ulItems.slice() });
        ulItems = [];
    };
    const flushOl = () => {
        if (!olItems.length) return;
        blocks.push({ type: 'ol', items: olItems.slice() });
        olItems = [];
    };

    for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, '');
        const trimmed = line.trim();

        // Blank line = paragraph break
        if (!trimmed) {
            flushUl();
            flushOl();
            flushParagraph();
            continue;
        }

        // Unordered list: • item, - item
        const ulMatch = trimmed.match(/^([•-])\s+(.*)$/);
        if (ulMatch) {
            flushOl();
            flushParagraph();
            ulItems.push(ulMatch[2]);
            continue;
        }

        // Ordered list: 1. item or 1) item
        const olMatch = trimmed.match(/^(\d+)[.)]\s+(.*)$/);
        if (olMatch) {
            flushUl();
            flushParagraph();
            olItems.push(olMatch[2]);
            continue;
        }

        // Normal paragraph line
        flushUl();
        flushOl();
        paragraphLines.push(trimmed);
    }

    flushUl();
    flushOl();
    flushParagraph();

    const inlineFormat = (text) => {
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/CaseID:\s*(\w+)/g, '<span class="case-id">CaseID: $1</span>')
            .replace(/Job_Name:\s*([^,\n]+)/g, '<span class="job-name">Job Name: $1</span>');
    };

    const html = blocks.map((b) => {
        if (b.type === 'p') {
            return '<p>' + inlineFormat(b.text).replace(/\n/g, '<br>') + '</p>';
        }
        if (b.type === 'ul') {
            return '<ul>' + b.items.map(i => '<li>' + inlineFormat(i) + '</li>').join('') + '</ul>';
        }
        if (b.type === 'ol') {
            return '<ol>' + b.items.map(i => '<li>' + inlineFormat(i) + '</li>').join('') + '</ol>';
        }
        return '';
    }).join('');

    return html || '<p></p>';
}

// Create expandable sources section
function createSourcesSection(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';
    
    const sourcesHeader = document.createElement('div');
    sourcesHeader.className = 'sources-header';
    sourcesHeader.innerHTML = `
        <div class="sources-header-left">
            <i class="fas fa-file-alt"></i>
            <span>📄 Source Documents (${sources.length})</span>
        </div>
        <i class="fas fa-chevron-down expand-icon"></i>
    `;
    
    const sourcesContent = document.createElement('div');
    sourcesContent.className = 'sources-content';
    
    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        
        const sourceMeta = document.createElement('div');
        sourceMeta.className = 'source-meta';
        
        // Handle different source types
        if (source.type === 'case_record') {
            sourceMeta.innerHTML = `
                <strong>CaseID:</strong> ${source.case_id} | 
                <strong>Job Name:</strong> ${source.job_name}
            `;
        } else if (source.type === 'pdf_document') {
            sourceMeta.innerHTML = `
                <strong>PDF:</strong> ${source.filename}
            `;
        } else {
            sourceMeta.innerHTML = `
                <strong>Source:</strong> ${JSON.stringify(source.metadata || {})}
            `;
        }
        
        const sourceContent = document.createElement('div');
        sourceContent.className = 'source-content';
        
        // Format source content better
        const content = source.content
            .replace(/Problem:\s*/i, '<strong>Problem:</strong> ')
            .replace(/Resolution:\s*/i, '<br><strong>Resolution:</strong> ');
        sourceContent.innerHTML = content;
        
        sourceItem.appendChild(sourceMeta);
        sourceItem.appendChild(sourceContent);
        sourcesContent.appendChild(sourceItem);
    });
    
    // Add toggle functionality
    sourcesHeader.addEventListener('click', function() {
        const isExpanded = sourcesContent.classList.contains('expanded');
        
        if (isExpanded) {
            sourcesContent.classList.remove('expanded');
            sourcesHeader.classList.remove('expanded');
        } else {
            sourcesContent.classList.add('expanded');
            sourcesHeader.classList.add('expanded');
        }
    });
    
    sourcesDiv.appendChild(sourcesHeader);
    sourcesDiv.appendChild(sourcesContent);
    
    return sourcesDiv;
}

async function handleVerifyAnswer(event) {
    const btn = event.target.closest('.verify-answer-btn');
    if (!btn || btn.disabled) return;
    const verifyWrap = btn.closest('.verify-answer-wrap');
    const contentDiv = verifyWrap.closest('.message-content');
    let existingEval = contentDiv.querySelector('.evaluation-wrap');
    if (existingEval) existingEval.remove();
    const question = btn.dataset.question || '';
    const response = btn.dataset.response || '';
    let sources = [];
    try {
        sources = JSON.parse(btn.dataset.sources || '[]');
    } catch (_) {}
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking...';
    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, response, sources })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Evaluation failed');
        const verdict = data.verdict || 'warning';
        const scoreStr = data.score != null && !isNaN(Number(data.score)) ? Number(data.score) : '—';
        const faithfulnessStr = data.faithfulness != null && !isNaN(Number(data.faithfulness)) ? Number(data.faithfulness) : '—';
        const relevanceStr = data.answer_relevance != null && !isNaN(Number(data.answer_relevance)) ? Number(data.answer_relevance) : '—';
        const issuesList = (data.issues || []).map(i => '<li>' + escapeHtml(i) + '</li>').join('');
        const suggestionsList = (data.suggestions || []).map(s => '<li>' + escapeHtml(s) + '</li>').join('');
        const strengthsList = (data.strengths || []).map(s => '<li>' + escapeHtml(s) + '</li>').join('');
        const evaluationWrap = document.createElement('div');
        evaluationWrap.className = 'evaluation-wrap evaluation-wrap--' + verdict;
        const header = document.createElement('div');
        header.className = 'evaluation-header';
        header.innerHTML = `
            <div class="evaluation-header-left">
                <i class="fas fa-shield-alt"></i>
                <span>Answer evaluation</span>
                <span class="evaluation-verdict">${escapeHtml((verdict).toUpperCase())}</span>
                <span class="evaluation-score">${scoreStr}/100</span>
            </div>
            <i class="fas fa-chevron-down evaluation-expand-icon"></i>
        `;
        const content = document.createElement('div');
        content.className = 'evaluation-content';
        content.innerHTML = `
            <div class="evaluation-body">
                <div class="evaluation-metrics">
                    <div class="evaluation-metric">
                        <span class="evaluation-metric-label">Faithfulness score</span>
                        <span class="evaluation-metric-value">${faithfulnessStr}/100</span>
                        <span class="evaluation-metric-desc">Monitors hallucinations.</span>
                    </div>
                    <div class="evaluation-metric">
                        <span class="evaluation-metric-label">Answer relevance</span>
                        <span class="evaluation-metric-value">${relevanceStr}/100</span>
                        <span class="evaluation-metric-desc">Ensures answers are on-topic.</span>
                    </div>
                </div>
                <p class="evaluation-summary">${escapeHtml(data.summary || '')}</p>
                ${issuesList ? '<div class="evaluation-section"><strong>Issues</strong><ul>' + issuesList + '</ul></div>' : ''}
                ${suggestionsList ? '<div class="evaluation-section"><strong>Suggestions</strong><ul>' + suggestionsList + '</ul></div>' : ''}
                ${strengthsList ? '<div class="evaluation-section"><strong>Strengths</strong><ul>' + strengthsList + '</ul></div>' : ''}
            </div>
        `;
        header.addEventListener('click', function () {
            const isExpanded = content.classList.contains('expanded');
            if (isExpanded) {
                content.classList.remove('expanded');
                header.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                header.classList.add('expanded');
            }
        });
        evaluationWrap.appendChild(header);
        evaluationWrap.appendChild(content);
        verifyWrap.after(evaluationWrap);
    } catch (err) {
        const evaluationWrap = document.createElement('div');
        evaluationWrap.className = 'evaluation-wrap evaluation-wrap--fail';
        const header = document.createElement('div');
        header.className = 'evaluation-header';
        header.innerHTML = `
            <div class="evaluation-header-left">
                <i class="fas fa-shield-alt"></i>
                <span>Answer evaluation</span>
                <span class="evaluation-verdict">FAIL</span>
            </div>
            <i class="fas fa-chevron-down evaluation-expand-icon"></i>
        `;
        const content = document.createElement('div');
        content.className = 'evaluation-content';
        content.innerHTML = '<div class="evaluation-body"><p class="evaluation-summary">' + escapeHtml(err.message) + '</p></div>';
        header.addEventListener('click', function () {
            const isExpanded = content.classList.contains('expanded');
            if (isExpanded) {
                content.classList.remove('expanded');
                header.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                header.classList.add('expanded');
            }
        });
        evaluationWrap.appendChild(header);
        evaluationWrap.appendChild(content);
        verifyWrap.after(evaluationWrap);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-shield-alt"></i> Verify answer';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Get current time
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit'
    });
}

// Update counters
function updateCounters() {
    return;
}

// Show/hide loading overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1001;
        animation: slideIn 0.3s ease-out;
        max-width: 400px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// ========== VOICE RECORDING FUNCTIONALITY ==========
let mediaRecorder = null;
let audioChunks = [];
let currentAudio = null;  // Track current audio playback
let isRecording = false;
const voiceBtn = document.getElementById('voiceBtn');

async function toggleVoiceRecording() {
    if (!isSystemInitialized) {
        showNotification('Please wait for system to initialize', 'error');
        return;
    }

    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });
        
        mediaRecorder.addEventListener('stop', async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendVoiceMessage(audioBlob);
            
            // Stop all audio tracks
            stream.getTracks().forEach(track => track.stop());
        });
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        messageInput.placeholder = 'Recording... Click mic to stop';
        
        showNotification('Recording started', 'info');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showNotification('Microphone access denied', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        voiceBtn.classList.remove('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        messageInput.placeholder = 'Ask for Krishna\'s wisdom...';
    }
}

async function sendVoiceMessage(audioBlob) {
    try {
        showLoading(true);
        
        // Create form data with audio file
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        const response = await fetch('/api/chat/voice', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Voice processing failed');
        }
        
        // Display transcribed question
        addMessage(data.input_text, 'user');
        
        addMessage(
            data.response,
            'assistant',
            data.sources || [],
            false,
            data.retrieval_metrics,
            data.input_text
        );
        
        // Play audio response
        if (data.audio_base64) {
            playAudioResponse(data.audio_base64);
        }
        
        showNotification('Voice response ready', 'success');
        
    } catch (error) {
        console.error('Error sending voice message:', error);
        addMessage('Sorry, voice processing failed: ' + error.message, 'assistant', [], true);
        showNotification('Voice processing error', 'error');
    } finally {
        showLoading(false);
    }
}

function playAudioResponse(base64Audio) {
    try {
        // Stop any currently playing audio
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }
        
        // Decode base64 and create audio element
        const audioData = atob(base64Audio);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        
        for (let i = 0; i < audioData.length; i++) {
            view[i] = audioData.charCodeAt(i);
        }
        
        const blob = new Blob([arrayBuffer], { type: 'audio/mp3' });
        const audioUrl = URL.createObjectURL(blob);
        
        const audio = new Audio(audioUrl);
        currentAudio = audio;  // Store reference to current audio
        
        // Show audio control button
        const audioControlBtn = document.getElementById('audioControlBtn');
        audioControlBtn.style.display = 'flex';
        audioControlBtn.classList.add('playing');
        
        // Update button state based on playback state
        audio.addEventListener('play', () => {
            audioControlBtn.classList.add('playing');
            updateAudioControlButton();
        });
        
        audio.addEventListener('pause', () => {
            updateAudioControlButton();
        });
        
        audio.addEventListener('ended', () => {
            audioControlBtn.style.display = 'none';
            audioControlBtn.classList.remove('playing');
            currentAudio = null;
            URL.revokeObjectURL(audioUrl);
        });
        
        audio.addEventListener('error', () => {
            audioControlBtn.style.display = 'none';
            audioControlBtn.classList.remove('playing');
            currentAudio = null;
            URL.revokeObjectURL(audioUrl);
        });
        
        audio.play();
        
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

// Toggle audio playback (pause/resume)
function toggleAudioPlayback() {
    if (!currentAudio) return;
    
    if (currentAudio.paused) {
        currentAudio.play();
    } else {
        currentAudio.pause();
    }
}

// Update audio control button icon
function updateAudioControlButton() {
    const audioControlBtn = document.getElementById('audioControlBtn');
    const icon = audioControlBtn.querySelector('i');
    
    if (currentAudio && !currentAudio.paused) {
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
        audioControlBtn.setAttribute('aria-label', 'Pause audio');
    } else {
        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
        audioControlBtn.setAttribute('aria-label', 'Resume audio');
    }
}

// Auto-refresh status every 30 seconds
setInterval(checkSystemStatus, 30000);
