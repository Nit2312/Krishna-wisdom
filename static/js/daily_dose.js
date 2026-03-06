/**
 * Daily Dose — Sanatan Sutra
 * Handles loading, navigation, and interaction for the /daily-dose page.
 */

// ─── State ────────────────────────────────────────────────────────────────────
let currentDay  = null;   // null = auto (today's topic from server), number = explicit
let todayDay    = null;   // the real calendar day number returned by server

// ─── Bootstrap ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    applyStoredTheme();
    loadDailyDose();
});

// ─── Theme (synced with main app) ────────────────────────────────────────────
function applyStoredTheme() {
    const stored = localStorage.getItem('theme');
    if (stored === 'dark') {
        document.documentElement.classList.add('dark-mode');
        const icon = document.querySelector('#themeToggle i');
        if (icon) { icon.classList.replace('fa-moon', 'fa-sun'); }
    }
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    const icon = document.querySelector('#themeToggle i');
    if (icon) {
        icon.classList.toggle('fa-moon', !isDark);
        icon.classList.toggle('fa-sun',  isDark);
    }
}

// ─── Load dose from API ───────────────────────────────────────────────────────
async function loadDailyDose(dayNumber = null) {
    showLoading();

    try {
        const url = dayNumber ? `/api/daily-dose?day=${dayNumber}` : '/api/daily-dose';
        const res  = await fetch(url);

        if (!res.ok) throw new Error(`Server returned ${res.status}`);

        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to load.');

        const data = json.data;
        currentDay = data.day;
        todayDay   = data.today_day;

        renderDose(data);
        updateSidebarState(data);
        updateStatusBar(true);

    } catch (err) {
        showError(err.message || "Could not load today's dose. Please try again.");
        updateStatusBar(false);
    }
}

// ─── Render the card ──────────────────────────────────────────────────────────
function renderDose(data) {
    // Hero header
    const isToday = (data.day === data.today_day);
    el('doseDayBadge').textContent  = isToday
        ? `Day ${data.day} of 100 · Today`
        : `Day ${data.day} of 100`;
    el('doseDate').textContent       = data.date || '';
    el('journeyStart').textContent   = data.journey_start
        ? `Journey started ${data.journey_start}` : '';

    // Meta
    el('doseTheme').textContent      = data.theme   || '';
    el('doseTitle').textContent      = data.title   || '';
    el('doseSourceText').textContent = data.source  || '';
    el('doseQuestion').textContent   = `"${data.question || ''}"`;

    // Message — split into proper paragraphs
    const paras = (data.message || '')
        .split(/\n{2,}/)
        .map(p => p.trim())
        .filter(Boolean);
    el('doseMessage').innerHTML = paras.map(p => `<p>${escHtml(p)}</p>`).join('');

    // Progress bar (Day 5 of 100 → 5 %)
    const pct = Math.round((data.day / 100) * 100);
    el('doseProgressFill').style.width = pct + '%';

    showCard();
}

// ─── Sidebar state ────────────────────────────────────────────────────────────
function updateSidebarState(data) {
    const labelEl = document.getElementById('sidebarDayLabel');
    const badgeEl = document.getElementById('sidebarDayBadge');
    if (labelEl) labelEl.textContent = data.title
        ? data.title.substring(0, 28) + (data.title.length > 28 ? '…' : '')
        : "Today's Teaching";
    if (badgeEl) badgeEl.textContent = `D${data.today_day || data.day}`;
}

// ─── Navigation ───────────────────────────────────────────────────────────────
function goToToday() {
    loadDailyDose(null);   // null = server picks today's day
    document.querySelector('.dose-wrapper')?.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── Copy & Share ─────────────────────────────────────────────────────────────
function copyMessage() {
    const title   = el('doseTitle')?.textContent || '';
    const message = el('doseMessage')?.innerText || '';
    const text    = `${title}\n\n${message}\n\n— Sanatan Sutra`;

    navigator.clipboard.writeText(text)
        .then(() => {
            const btn = el('copyBtn');
            if (btn) {
                btn.classList.add('copied');
                btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    btn.classList.remove('copied');
                    btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                }, 2200);
            }
        })
        .catch(() => showToast('Select the text and copy manually.'));
}

function shareMessage() {
    const title   = el('doseTitle')?.textContent || '';
    const message = el('doseMessage')?.innerText || '';
    if (navigator.share) {
        navigator.share({
            title: `Daily Dose: ${title}`,
            text: message.substring(0, 300) + '…',
            url: window.location.href,
        }).catch(() => {});
    } else {
        copyMessage();
    }
}

function showToast(msg) {
    document.getElementById('doseToast')?.remove();
    const toast = document.createElement('div');
    toast.id = 'doseToast';
    toast.textContent = msg;
    Object.assign(toast.style, {
        position: 'fixed', bottom: '1.5rem', left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(30,30,30,0.92)', color: '#fff',
        backdropFilter: 'blur(6px)',
        padding: '0.6rem 1.3rem',
        borderRadius: '0.5rem',
        fontSize: '0.875rem',
        zIndex: '9999',
        boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
        transition: 'opacity 0.3s ease',
    });
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 350);
    }, 2500);
}

// ─── UI state helpers ─────────────────────────────────────────────────────────
function showLoading() {
    el('doseLoading').style.display = 'flex';
    el('doseError').style.display   = 'none';
    el('doseCard').style.display    = 'none';
}

function showCard() {
    el('doseLoading').style.display = 'none';
    el('doseError').style.display   = 'none';
    el('doseCard').style.display    = 'block';
}

function showError(msg) {
    el('doseLoading').style.display  = 'none';
    el('doseError').style.display    = 'flex';
    el('doseCard').style.display     = 'none';
    el('doseErrorMsg').textContent   = msg;
}

function updateStatusBar(ok) {
    const dot  = document.querySelector('#statusIndicator .status-dot');
    const text = document.querySelector('#statusIndicator .status-text');
    if (dot)  dot.className  = ok ? 'status-dot connected' : 'status-dot';
    if (text) text.textContent = ok ? 'Ready' : 'Error';
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escAttr(str) {
    return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

