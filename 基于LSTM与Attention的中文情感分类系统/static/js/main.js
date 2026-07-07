/**
 * 中文情感分析系统 — 前端交互逻辑
 */

/* ── DOM 引用 ────────────────────────────────── */
const inputText = document.getElementById('inputText');
const predictBtn = document.getElementById('predictBtn');
const sampleBtn = document.getElementById('sampleBtn');
const inputStatus = document.getElementById('inputStatus');
const resultsCard = document.getElementById('resultsCard');
const tokenDisplay = document.getElementById('tokenDisplay');
const modelCards = document.getElementById('modelCards');
const attentionSection = document.getElementById('attentionSection');
const attentionChartCtx = document.getElementById('attentionChart')?.getContext('2d');
const probSection = document.getElementById('probSection');
const probChartCtx = document.getElementById('probChart')?.getContext('2d');
const galleryGrid = document.getElementById('galleryGrid');
const galleryTabs = document.getElementById('galleryTabs');

let attentionChart = null;
let probChart = null;
let allGalleryImages = [];

/* ── 常量 ────────────────────────────────────── */
const MODEL_LABELS = { rnn: 'RNN', lstm: 'BiLSTM', 'attention_lstm': 'Attention-LSTM', 'cnn_lstm': 'CNN-BiLSTM', 'bert': 'BERT' };
const LABEL_COLORS = { 0: '#dc2626', 1: '#16a34a', '-1': '#94a3b8' };
const LABEL_NAMES = { 0: '负面', 1: '正面', '-1': '不确定' };

const MODEL_COLORS = { rnn: '#f59e0b', lstm: '#3b82f6', attention_lstm: '#8b5cf6', cnn_lstm: '#06b6d4', bert: '#ec4899' };

// 颜色按比例混白
function lighten(hex, ratio) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    const mix = (c) => Math.round(c + (255 - c) * ratio);
    return '#' + [mix(r), mix(g), mix(b)].map(c => c.toString(16).padStart(2,'0')).join('');
}

/* ── API ─────────────────────────────────────── */
async function callPredict(text) {
    const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });
    return resp.json();
}

async function callSample() {
    const resp = await fetch('/api/sample');
    return resp.json();
}

async function loadGallery() {
    try {
        const resp = await fetch('/api/gallery');
        return resp.json();
    } catch { return []; }
}

/* ── 渲染 ─────────────────────────────────────── */

function renderTokens(tokens) {
    tokenDisplay.innerHTML = tokens
        .map(t => `<span class="token-tag">${t}</span>`)
        .join('');
    if (!tokens.length) {
        tokenDisplay.innerHTML = '<span style="color:#94a3b8;font-size:.85rem;">无有效分词</span>';
    }
}

function renderModelCards(results, bertStatus) {
    const order = ['rnn', 'lstm', 'attention_lstm', 'cnn_lstm', 'bert'];
    modelCards.innerHTML = order.map(name => {
        const r = results[name];
        if (!r) {
            if (name === 'bert' && bertStatus && bertStatus !== 'ready') {
                return '<div class=\"model-result\"><div class=\"model-info\"><h4>BERT</h4><span class=\"confidence\" style=\"color:#94a3b8;\">后台加载中...</span></div><div class=\"model-badge\" style=\"background:#94a3b8;\">等待</div></div>';
            }
            return '';
        }
        const label = MODEL_LABELS[name] || name;
        const badgeColor = r.color || '#4f46e5';
        const probLabel = LABEL_NAMES[r.prediction] || '';
        const finalColor = r.confidence_level === 'medium' ? lighten(badgeColor, 0.5) : badgeColor;
        return `
            <div class="model-result">
                <div class="model-info">
                    <h4>${label}</h4>
                    <span class="confidence">置信度 ${(r.confidence * 100).toFixed(1)}%${r.confidence_level === "low" ? " · 不确定" : r.confidence_level === "medium" ? " · 偏低" : ""}</span>
                    <div class="prob-bar-row">
                        <span class="prob-bar-label" style="color:#dc2626;">负</span>
                        <span style="flex:1;height:3px;background:#fee2e2;border-radius:2px;align-self:center;">
                            <span style="display:block;width:${(r.probabilities['负面']*100)}%;height:100%;background:#dc2626;border-radius:2px;"></span>
                        </span>
                        <span class="prob-bar-label" style="color:#16a34a;">正</span>
                        <span style="flex:1;height:3px;background:#dcfce7;border-radius:2px;align-self:center;">
                            <span style="display:block;width:${(r.probabilities['正面']*100)}%;height:100%;background:#16a34a;border-radius:2px;"></span>
                        </span>
                    </div>
                </div>
                <div class="model-badge" style="background:${finalColor}">${probLabel}</div>
            </div>
        `;
    }).join('');
}

function renderAttentionChart(tokens, attentionWeights) {
    if (!attentionChartCtx) return;
    if (attentionChart) attentionChart.destroy();

    const n = Math.min(tokens.length, attentionWeights.length);
    const labels = tokens.slice(-n);
    const weights = attentionWeights.slice(-n);

    if (!weights.length) return;

    const maxW = Math.max(...weights, 0.001);

    attentionChart = new Chart(attentionChartCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '注意力权重',
                data: weights,
                backgroundColor: weights.map(w => {
                    const alpha = Math.min(w / maxW, 1);
                    return `rgba(79, 70, 229, ${0.25 + alpha * 0.75})`;
                }),
                borderColor: 'rgba(79, 70, 229, 0.9)',
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `权重: ${ctx.raw.toFixed(4)}` } },
            },
            scales: {
                x: { ticks: { font: { size: 10 }, maxRotation: 45 } },
                y: { beginAtZero: true, title: { display: true, text: '权重', font: { size: 11 } } },
            },
        },
    });
}

function renderProbChart(results) {
    if (!probChartCtx) return;
    if (probChart) probChart.destroy();

    const order = ['rnn', 'lstm', 'attention_lstm', 'cnn_lstm', 'bert'];
    const datasets = order.map(name => {
        const r = results[name];
        if (!r) return null;
        return {
            label: MODEL_LABELS[name],
            data: [
                r.probabilities['负面'],
                r.probabilities['正面'],
            ],
            backgroundColor: [
                'rgba(220, 38, 38, 0.55)',
                'rgba(22, 163, 74, 0.55)',
            ],
            borderColor: MODEL_COLORS[name],
            borderWidth: 2,
        };
    }).filter(Boolean);

    probChart = new Chart(probChartCtx, {
        type: 'bar',
        data: { labels: ['负面', '正面'], datasets: datasets },
        options: {
            responsive: true,
            plugins: {
                tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%` } },
            },
            scales: {
                y: { beginAtZero: true, max: 1, ticks: { callback: v => (v * 100).toFixed(0) + '%' } },
            },
        },
    });
}

function renderGallery(images, category = 'all') {
    const filtered = category === 'all' ? images : images.filter(i => i.category === category);
    if (!filtered.length) {
        galleryGrid.innerHTML = '<div class="gallery-loading">暂无图表，请先运行 main.py 生成</div>';
        return;
    }
    galleryGrid.innerHTML = filtered.map(img => {
        const imgUrl = '/results-img/' + img.path.split('/').pop();
        return `<img src="${imgUrl}" alt="${img.name}" loading="lazy" onclick="openModal('${imgUrl}')">`;
    }).join('');
}

/* ── Modal ───────────────────────────────────── */
function openModal(src) {
    let modal = document.getElementById('imageModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'imageModal';
        modal.className = 'modal-overlay';
        modal.innerHTML = '<img class="modal-img" src="">';
        modal.onclick = () => modal.classList.remove('active');
        document.body.appendChild(modal);
    }
    modal.querySelector('.modal-img').src = src;
    modal.classList.add('active');
}

/* ── 事件处理 ────────────────────────────────── */

async function handlePredict() {
    const text = inputText.value.trim();
    if (!text || text.length < 3) {
        inputStatus.textContent = '请至少输入 3 个字符';
        inputStatus.style.color = '#dc2626';
        return;
    }
    inputStatus.textContent = '分析中...';
    inputStatus.style.color = '#4f46e5';
    predictBtn.disabled = true;

    try {
        const data = await callPredict(text);
        if (data.error) {
            inputStatus.textContent = data.error;
            inputStatus.style.color = '#dc2626';
            predictBtn.disabled = false;
            return;
        }
        resultsCard.style.display = 'block';
        renderTokens(data.tokens || []);
        renderModelCards(data.results, data.bert_status);

        const attn = data.results['attention_lstm'];
        if (attn && attn.attention && attn.attention.length > 0) {
            attentionSection.style.display = 'block';
            renderAttentionChart(data.tokens, attn.attention);
        } else {
            attentionSection.style.display = 'none';
        }

        probSection.style.display = 'block';
        renderProbChart(data.results);
        inputStatus.textContent = '分析完成';
        inputStatus.style.color = '#16a34a';
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        inputStatus.textContent = '服务器未响应，请确认 app.py 已启动';
        inputStatus.style.color = '#dc2626';
    } finally {
        predictBtn.disabled = false;
    }
}

async function handleSample() {
    inputStatus.textContent = '加载样本...';
    try {
        const data = await callSample();
        if (data.text) {
            inputText.value = data.text;
            inputStatus.textContent = '已加载测试样本 (真实标签: ' + data.label_name + ')';
            inputStatus.style.color = '#4f46e5';
        }
    } catch (err) {
        inputStatus.textContent = '加载样本失败，请稍后重试';
        inputStatus.style.color = '#dc2626';
    }
}

/* ── 事件绑定 ────────────────────────────────── */
predictBtn.addEventListener('click', handlePredict);
sampleBtn.addEventListener('click', handleSample);
inputText.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handlePredict();
    }
});

galleryTabs?.addEventListener('click', e => {
    if (!e.target.classList.contains('tab')) return;
    galleryTabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    e.target.classList.add('active');
    renderGallery(allGalleryImages, e.target.dataset.cat);
});

/* ── 初始化 ──────────────────────────────────── */
(async () => {
    const images = await loadGallery();
    allGalleryImages = images;
    renderGallery(images);
})();
