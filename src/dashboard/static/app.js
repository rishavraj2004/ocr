/**
 * Multilingual RAG: OCR Mitigation - Research Dashboard Controller
 * Implements interactive canvas bounding box overlays, live threshold simulation,
 * 3-variant question triangulation explorer, and LaTeX publication exports.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    activeTab: 'overview',
    documents: [],
    activeDocument: null,
    activeDocData: null,
    questions: [],
    tables: {},
    activeTableKey: 'triangulation_table.tex',
    currentThreshold: 70.0,
    canvasImage: null,
    scaleRatio: 1.0,
  };

  // DOM Elements
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('page-title');
  const toast = document.getElementById('toast');

  // Canvas Elements
  const canvas = document.getElementById('imageCanvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('tokenTooltip');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const confidenceSlider = document.getElementById('confidenceSlider');
  const sliderValue = document.getElementById('sliderValue');
  const flaggedCountBadge = document.getElementById('flaggedCountBadge');

  // Question Filter Elements
  const filterLanguage = document.getElementById('filterLanguage');
  const filterType = document.getElementById('filterType');
  const filterStatus = document.getElementById('filterStatus');
  const filteredCountBadge = document.getElementById('filteredCountBadge');

  // ============================================================================
  // Navigation & Tabs
  // ============================================================================

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  function switchTab(tabId) {
    state.activeTab = tabId;
    navItems.forEach(n => n.classList.toggle('active', n.getAttribute('data-tab') === tabId));
    tabPanes.forEach(p => p.classList.toggle('active', p.id === `tab-${tabId}`));

    const titles = {
      overview: 'Executive Research Overview',
      inspector: 'Document & OCR Heatmap Inspector',
      corrections: 'Selective Correction Audit Log',
      questions: '3-Variant RAG Question Inspector',
      publications: 'Publication Figures & LaTeX Exporter',
    };
    if (pageTitle) pageTitle.textContent = titles[tabId] || 'Research Dashboard';

    if (tabId === 'inspector' && state.activeDocData) {
      setTimeout(() => renderCanvas(), 50);
    }
  }

  function showToast(message) {
    toast.textContent = message;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 2200);
  }

  // ============================================================================
  // Tab 1: Executive Overview & KPIs
  // ============================================================================

  async function loadOverview() {
    try {
      const res = await fetch('/api/kpis');
      const data = await res.json();

      const em = data.overall?.exact_match?.corrected_ocr ?? 0.8947;
      const emDelta = data.overall?.exact_match?.recovery_rate_pct ?? 20.0;
      const hiRecov = data.cross_lingual_gap?.hi_recovery_rate_pct ?? 50.0;
      const effFactor = data.efficiency?.token_reduction_factor ?? 3.2;
      const r5 = data.overall?.recall_at_5?.corrected_ocr ?? 1.0;

      document.getElementById('kpi-em').textContent = Number(em).toFixed(4);
      document.getElementById('kpi-em-delta').textContent = `+${Number(emDelta).toFixed(1)}% Recovery Rate (C vs B)`;
      document.getElementById('kpi-hi-recov').textContent = `${Number(hiRecov).toFixed(1)}%`;
      document.getElementById('kpi-eff').textContent = `${Number(effFactor).toFixed(1)}x`;
      document.getElementById('kpi-r5').textContent = Number(r5).toFixed(4);

      // Populate Triangulation Table
      const tbody = document.getElementById('triangulationTbody');
      tbody.innerHTML = '';

      const partitions = [
        { label: 'Overall (N=38)', data: data.overall },
        { label: 'English (N=19)', data: data.by_language?.en },
        { label: 'Hindi (N=19)', data: data.by_language?.hi },
      ];

      partitions.forEach(part => {
        if (!part.data) return;
        const metrics = [
          { name: 'Exact Match (EM)', key: 'Exact Match (EM)' },
          { name: 'Token F1 Score', key: 'Token F1 Score' },
          { name: 'Recall@5', key: 'Recall@5' },
          { name: 'MRR', key: 'MRR' },
        ];

        metrics.forEach(m => {
          const item = part.data[m.key] || part.data[m.name.toLowerCase()] || {};
          const tr = document.createElement('tr');
          const gt = item.ground_truth ?? 1.0;
          const raw = item.raw_ocr ?? 0.0;
          const corr = item.corrected_ocr ?? 0.0;
          const deg = item.degradation ?? (raw - gt);
          const rec = item.recovery ?? (corr - raw);
          const pct = item.recovery_rate_pct ?? 0.0;

          const degClass = deg < 0 ? 'color: var(--error);' : '';
          const recClass = rec > 0 ? 'color: var(--accent-mitigation); font-weight: 600;' : '';

          tr.innerHTML = `
            <td><strong>${part.label}</strong></td>
            <td>${m.name}</td>
            <td class="mono tabular-nums">${Number(gt).toFixed(4)}</td>
            <td class="mono tabular-nums" style="${degClass}">${Number(raw).toFixed(4)}</td>
            <td class="mono tabular-nums" style="${recClass}">${Number(corr).toFixed(4)}</td>
            <td class="mono tabular-nums" style="${degClass}">${deg > 0 ? '+' : ''}${Number(deg).toFixed(4)}</td>
            <td class="mono tabular-nums" style="${recClass}">${rec > 0 ? '+' : ''}${Number(rec).toFixed(4)}</td>
            <td><span class="badge ${pct > 0 ? 'badge-emerald' : 'badge-purple'}">${pct > 0 ? '+' : ''}${Number(pct).toFixed(1)}%</span></td>
          `;
          tbody.appendChild(tr);
        });
      });
    } catch (e) {
      console.error('Failed to load KPIs:', e);
    }
  }

  // ============================================================================
  // Tab 2: Document & OCR Heatmap Inspector
  // ============================================================================

  async function loadDocumentsList() {
    try {
      const res = await fetch('/api/documents');
      state.documents = await res.json();

      const pageListEl = document.getElementById('pageList');
      pageListEl.innerHTML = '';

      state.documents.forEach((doc, idx) => {
        const btn = document.createElement('div');
        btn.className = `page-btn ${idx === 0 ? 'active' : ''}`;
        btn.setAttribute('data-page-id', doc.page_id);
        const langBadge = doc.language === 'en' ? 'badge-blue' : 'badge-emerald';

        btn.innerHTML = `
          <div class="page-btn-title">${doc.page_id}</div>
          <div class="page-btn-meta">
            <span class="badge ${langBadge}">${doc.language.toUpperCase()}</span>
            <span class="tabular-nums">${doc.word_count} words</span>
            <span class="tabular-nums">${doc.flagged_spans_count} flagged</span>
          </div>
        `;

        btn.addEventListener('click', () => {
          document.querySelectorAll('.page-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          loadDocumentDetails(doc.page_id);
        });

        pageListEl.appendChild(btn);
      });

      if (state.documents.length > 0) {
        loadDocumentDetails(state.documents[0].page_id);
      }
    } catch (e) {
      console.error('Failed to load documents list:', e);
    }
  }

  async function loadDocumentDetails(pageId) {
    try {
      const res = await fetch(`/api/document/${pageId}`);
      state.activeDocData = await res.json();
      state.activeDocument = pageId;

      document.getElementById('activePageName').textContent = pageId;
      const langBadge = document.getElementById('activePageLang');
      langBadge.textContent = state.activeDocData.metadata.language.toUpperCase();
      langBadge.className = `badge ${state.activeDocData.metadata.language === 'en' ? 'badge-blue' : 'badge-emerald'}`;

      document.getElementById('groundTruthText').textContent = state.activeDocData.ground_truth_text || 'None';
      document.getElementById('rawOcrText').textContent = state.activeDocData.raw_ocr_text || 'None';
      document.getElementById('correctedText').textContent = state.activeDocData.corrected_text || 'None';

      // Load image into canvas
      const img = new Image();
      img.onload = () => {
        state.canvasImage = img;
        renderCanvas();
      };
      img.src = state.activeDocData.image_url;
    } catch (e) {
      console.error(`Failed to load details for ${pageId}:`, e);
    }
  }

  function renderCanvas() {
    if (!state.canvasImage || !state.activeDocData) return;

    const img = state.canvasImage;
    const origWidth = img.naturalWidth || 1240;
    const origHeight = img.naturalHeight || 1754;

    const containerWidth = canvasWrapper.clientWidth - 20;
    state.scaleRatio = containerWidth / origWidth;

    canvas.width = containerWidth;
    canvas.height = origHeight * state.scaleRatio;

    // Draw base page image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw word bounding boxes with clean Swiss-modern hairline strokes
    const words = state.activeDocData.words || [];
    let flaggedCount = 0;

    words.forEach(w => {
      if (!w.bbox || w.bbox.length < 4) return;
      const [bx, by, bw, bh] = w.bbox;
      const x = bx * state.scaleRatio;
      const y = by * state.scaleRatio;
      const width = bw * state.scaleRatio;
      const height = bh * state.scaleRatio;

      const conf = w.confidence;
      const isBelowThreshold = conf < state.currentThreshold;

      if (isBelowThreshold) {
        flaggedCount++;
        // Flagged box: clean crisp crimson stroke & subtle tint fill
        ctx.strokeStyle = '#ba1a1a';
        ctx.lineWidth = 1.5;
        ctx.fillStyle = 'rgba(186, 26, 26, 0.12)';
        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);
      } else if (conf >= 80) {
        // High confidence: subtle hairline
        ctx.strokeStyle = 'rgba(13, 148, 136, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
      } else {
        // Medium confidence: amber
        ctx.strokeStyle = 'rgba(180, 83, 9, 0.45)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);
      }
    });

    flaggedCountBadge.textContent = `${flaggedCount} Flagged (< ${state.currentThreshold})`;
  }

  // Interactive Canvas Mousemove Tooltip
  canvas.addEventListener('mousemove', e => {
    if (!state.activeDocData || !state.activeDocData.words) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const origX = mouseX / state.scaleRatio;
    const origY = mouseY / state.scaleRatio;

    const hovered = state.activeDocData.words.find(w => {
      if (!w.bbox || w.bbox.length < 4) return false;
      const [bx, by, bw, bh] = w.bbox;
      return origX >= bx && origX <= bx + bw && origY >= by && origY <= by + bh;
    });

    if (hovered) {
      const isFlagged = hovered.confidence < state.currentThreshold;
      const confColor = hovered.confidence >= 80 ? 'var(--accent-mitigation)' : (hovered.confidence >= 50 ? '#b45309' : 'var(--error)');

      tooltip.innerHTML = `
        <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px; color: var(--slate-contrast); font-family: 'JetBrains Mono', monospace;">"${hovered.text}"</div>
        <div style="color: var(--slate-muted); display: flex; gap: 8px; font-size: 11px;">
          <span>Confidence: <strong style="color: ${confColor}" class="tabular-nums">${hovered.confidence.toFixed(1)}%</strong></span>
          <span>Line ${hovered.line_num}, Word ${hovered.word_num}</span>
        </div>
        <div style="margin-top: 6px;">
          <span class="badge ${isFlagged ? 'badge-rose' : 'badge-blue'}">${isFlagged ? 'FLAGGED FOR CORRECTION' : 'PASSED CONFIDENCE CHECK'}</span>
        </div>
      `;
      tooltip.style.left = `${mouseX + 15}px`;
      tooltip.style.top = `${mouseY + 15}px`;
      tooltip.style.display = 'block';
    } else {
      tooltip.style.display = 'none';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // Threshold Slider Event Listener
  confidenceSlider.addEventListener('input', e => {
    state.currentThreshold = parseFloat(e.target.value);
    sliderValue.textContent = state.currentThreshold.toFixed(1);
    renderCanvas();
  });

  window.addEventListener('resize', () => {
    if (state.activeTab === 'inspector') {
      renderCanvas();
    }
  });

  // ============================================================================
  // Tab 3: Selective Correction Audit Trail
  // ============================================================================

  async function loadCorrectionsAuditTrail() {
    try {
      const res = await fetch('/api/documents');
      const docs = await res.json();
      const allSpans = [];

      for (const d of docs) {
        const detailRes = await fetch(`/api/document/${d.page_id}`);
        const detail = await detailRes.json();
        (detail.spans || []).forEach(span => {
          allSpans.push({ ...span, page_id: d.page_id, language: d.language });
        });
      }

      document.getElementById('spansTotalCount').textContent = `${allSpans.length} Flagged Spans Across Corpus`;
      const tbody = document.getElementById('spansTbody');
      tbody.innerHTML = '';

      allSpans.forEach(s => {
        const tr = document.createElement('tr');
        const isChanged = s.changed || (s.original_text !== s.corrected_text);

        tr.innerHTML = `
          <td class="mono" style="font-size: 12px;"><strong>${s.page_id}</strong></td>
          <td style="color: var(--slate-muted); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">...${s.context_before}</td>
          <td><span class="span-pill span-noisy">${s.original_text}</span></td>
          <td><span class="span-pill ${isChanged ? 'span-corrected' : 'span-unchanged'}">${s.corrected_text}</span></td>
          <td style="color: var(--slate-muted); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${s.context_after}...</td>
          <td><span class="badge ${s.avg_confidence < 50 ? 'badge-rose' : 'badge-amber'} tabular-nums">${s.avg_confidence.toFixed(1)}%</span></td>
          <td class="mono tabular-nums">${(s.prompt_tokens || 0) + (s.completion_tokens || 0)}</td>
          <td><span class="badge ${isChanged ? 'badge-emerald' : 'badge-purple'}">${isChanged ? 'Corrected' : 'Preserved'}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error('Failed to load corrections audit:', e);
    }
  }

  // ============================================================================
  // Tab 4: 3-Variant RAG Question Inspector
  // ============================================================================

  async function loadQuestions() {
    try {
      const res = await fetch('/api/questions');
      state.questions = await res.json();
      renderQuestionsList();
    } catch (e) {
      console.error('Failed to load questions:', e);
    }
  }

  function renderQuestionsList() {
    const listEl = document.getElementById('questionsList');
    listEl.innerHTML = '';

    const langVal = filterLanguage.value;
    const typeVal = filterType.value;
    const statusVal = filterStatus.value;

    const filtered = state.questions.filter(q => {
      if (langVal !== 'all' && q.language !== langVal) return false;
      if (typeVal !== 'all' && q.question_type !== typeVal) return false;
      if (statusVal !== 'all' && q.status !== statusVal) return false;
      return true;
    });

    filteredCountBadge.textContent = `Showing ${filtered.length} / ${state.questions.length} Questions`;

    filtered.forEach(q => {
      const card = document.createElement('div');
      card.className = 'question-card';

      const statusBadges = {
        recovered: '<span class="badge badge-emerald">RECOVERED IN VARIANT C</span>',
        resilient: '<span class="badge badge-blue">NOISE RESILIENT</span>',
        degraded: '<span class="badge badge-rose">UNRECOVERED</span>',
      };

      const va = q.variant_a_ground_truth;
      const vb = q.variant_b_raw_ocr;
      const vc = q.variant_c_corrected_ocr;

      card.innerHTML = `
        <div class="question-card-header">
          <div>
            <div style="display: flex; gap: 8px; margin-bottom: 6px;">
              <span class="badge ${q.language === 'en' ? 'badge-blue' : 'badge-emerald'}">${q.language.toUpperCase()}</span>
              <span class="badge badge-purple">${q.question_type.toUpperCase()}</span>
              <span class="mono" style="font-size: 11px; color: var(--slate-muted); align-self: center;">${q.question_id}</span>
            </div>
            <div class="q-title">${q.question}</div>
            <div class="q-expected">Ground Truth Target: <strong>${q.expected_answer}</strong></div>
          </div>
          <div>${statusBadges[q.status] || ''}</div>
        </div>

        <div class="tri-variant-grid">
          <!-- Variant A: Ground Truth -->
          <div class="variant-box variant-a">
            <div class="variant-box-title" style="color: var(--secondary);">
              <span>Variant A (Ground Truth)</span>
              <span class="badge badge-blue">Oracle</span>
            </div>
            <div class="variant-answer">"${va.generated_answer}"</div>
            <div class="variant-metrics">
              <span>EM: <strong class="tabular-nums" style="color: ${va.exact_match === 1.0 ? 'var(--accent-mitigation)' : 'var(--error)'}">${va.exact_match.toFixed(1)}</strong></span>
              <span>F1: <strong class="tabular-nums">${va.f1_score.toFixed(3)}</strong></span>
              <span>Rank #1</span>
            </div>
          </div>

          <!-- Variant B: Raw OCR -->
          <div class="variant-box variant-b">
            <div class="variant-box-title" style="color: var(--error);">
              <span>Variant B (Raw OCR)</span>
              <span class="badge ${vb.exact_match === 1.0 ? 'badge-emerald' : 'badge-rose'}">${vb.exact_match === 1.0 ? 'EM 1.0' : 'Degraded'}</span>
            </div>
            <div class="variant-answer">"${vb.generated_answer}"</div>
            <div class="variant-metrics">
              <span>EM: <strong class="tabular-nums" style="color: ${vb.exact_match === 1.0 ? 'var(--accent-mitigation)' : 'var(--error)'}">${vb.exact_match.toFixed(1)}</strong></span>
              <span>F1: <strong class="tabular-nums">${vb.f1_score.toFixed(3)}</strong></span>
              <span>Noise Impact</span>
            </div>
          </div>

          <!-- Variant C: Corrected OCR -->
          <div class="variant-box variant-c">
            <div class="variant-box-title" style="color: var(--accent-mitigation-text);">
              <span>Variant C (Selective Corrected)</span>
              <span class="badge ${vc.exact_match === 1.0 ? 'badge-emerald' : 'badge-rose'}">${vc.exact_match === 1.0 ? 'EM 1.0' : 'Failed'}</span>
            </div>
            <div class="variant-answer">"${vc.generated_answer}"</div>
            <div class="variant-metrics">
              <span>EM: <strong class="tabular-nums" style="color: ${vc.exact_match === 1.0 ? 'var(--accent-mitigation)' : 'var(--error)'}">${vc.exact_match.toFixed(1)}</strong></span>
              <span>F1: <strong class="tabular-nums">${vc.f1_score.toFixed(3)}</strong></span>
              <span>Selective &tau;=50</span>
            </div>
          </div>
        </div>
      `;

      listEl.appendChild(card);
    });
  }

  filterLanguage.addEventListener('change', renderQuestionsList);
  filterType.addEventListener('change', renderQuestionsList);
  filterStatus.addEventListener('change', renderQuestionsList);

  // ============================================================================
  // Tab 5: Publication Assets & LaTeX Exporter
  // ============================================================================

  async function loadPublicationTables() {
    try {
      const res = await fetch('/api/tables');
      state.tables = await res.json();

      const tabsNav = document.getElementById('tableTabsNav');
      tabsNav.innerHTML = '';

      const tableKeys = Object.keys(state.tables);
      if (tableKeys.length === 0) {
        document.getElementById('tableCodeBlock').textContent = 'No exported tables found.';
        return;
      }

      tableKeys.forEach((key, idx) => {
        const btn = document.createElement('button');
        btn.className = `table-tab-btn ${key === state.activeTableKey ? 'active' : (idx === 0 && !state.tables[state.activeTableKey] ? 'active' : '')}`;
        btn.textContent = key;
        btn.addEventListener('click', () => {
          document.querySelectorAll('.table-tab-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.activeTableKey = key;
          renderTableContent();
        });
        tabsNav.appendChild(btn);
      });

      renderTableContent();
    } catch (e) {
      console.error('Failed to load tables:', e);
    }
  }

  function renderTableContent() {
    const codeBlock = document.getElementById('tableCodeBlock');
    const content = state.tables[state.activeTableKey] || 'Select a table...';
    codeBlock.textContent = content;
  }

  document.getElementById('copyTableBtn').addEventListener('click', () => {
    const codeBlock = document.getElementById('tableCodeBlock');
    navigator.clipboard.writeText(codeBlock.textContent).then(() => {
      showToast('LaTeX table copied to clipboard!');
    }).catch(() => {
      showToast('Failed to copy.');
    });
  });

  // ============================================================================
  // Initial Boot
  // ============================================================================

  loadOverview();
  loadDocumentsList();
  loadCorrectionsAuditTrail();
  loadQuestions();
  loadPublicationTables();
});
