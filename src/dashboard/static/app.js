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
      'live-lab': 'Live Q&A & Pipeline Testing Lab',
      publications: 'Publication Figures & LaTeX Exporter',
    };
    if (pageTitle) pageTitle.textContent = titles[tabId] || 'Research Dashboard';

    if (tabId === 'inspector' && state.activeDocData) {
      setTimeout(() => renderCanvas(), 50);
    }
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
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
  // Tab 6: Live Q&A & Pipeline Testing Lab Controller
  // ============================================================================

  function initLiveLab() {
    const btnModeUploadDoc = document.getElementById('btnModeUploadDoc');
    const panelLiveQnA = document.getElementById('panelLiveQnA');
    const panelLayerLab = document.getElementById('panelLayerLab');
    const panelUploadDoc = document.getElementById('panelUploadDoc');

    if (!btnModeLiveQnA || !btnModeLayerLab) return;

    // Mode Switcher
    btnModeLiveQnA.addEventListener('click', () => {
      btnModeLiveQnA.classList.add('active');
      btnModeLayerLab.classList.remove('active');
      if (btnModeUploadDoc) btnModeUploadDoc.classList.remove('active');
      panelLiveQnA.style.display = 'block';
      panelLayerLab.style.display = 'none';
      if (panelUploadDoc) panelUploadDoc.style.display = 'none';
    });

    btnModeLayerLab.addEventListener('click', () => {
      btnModeLayerLab.classList.add('active');
      btnModeLiveQnA.classList.remove('active');
      if (btnModeUploadDoc) btnModeUploadDoc.classList.remove('active');
      panelLiveQnA.style.display = 'none';
      panelLayerLab.style.display = 'block';
      if (panelUploadDoc) panelUploadDoc.style.display = 'none';
      populateLayerDocSelect();
    });

    if (btnModeUploadDoc && panelUploadDoc) {
      btnModeUploadDoc.addEventListener('click', () => {
        btnModeUploadDoc.classList.add('active');
        btnModeLiveQnA.classList.remove('active');
        btnModeLayerLab.classList.remove('active');
        panelLiveQnA.style.display = 'none';
        panelLayerLab.style.display = 'none';
        panelUploadDoc.style.display = 'block';
      });
    }

    // Populate Sample Question Chips
    loadLiveSamplePresets();
    initUploadModule();

    // Query Execution
    const btnExecuteLiveQuery = document.getElementById('btnExecuteLiveQuery');
    const liveQuestionInput = document.getElementById('liveQuestionInput');
    btnExecuteLiveQuery.addEventListener('click', () => executeLiveRAGQuery());
    liveQuestionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') executeLiveRAGQuery();
    });

    // Layer Lab Interactivity
    const liveTauSlider = document.getElementById('liveTauSlider');
    const liveTauLabel = document.getElementById('liveTauLabel');
    if (liveTauSlider && liveTauLabel) {
      liveTauSlider.addEventListener('input', () => {
        liveTauLabel.textContent = Number(liveTauSlider.value).toFixed(1);
      });
    }

    const liveCtxSlider = document.getElementById('liveCtxSlider');
    const liveCtxLabel = document.getElementById('liveCtxLabel');
    if (liveCtxSlider && liveCtxLabel) {
      liveCtxSlider.addEventListener('input', () => {
        liveCtxLabel.textContent = `\u00B1${liveCtxSlider.value} words`;
      });
    }

    const btnRunPreprocess = document.getElementById('btnRunPreprocess');
    if (btnRunPreprocess) {
      btnRunPreprocess.addEventListener('click', executeLivePreprocessingTest);
    }

    const btnRunLiveCorrection = document.getElementById('btnRunLiveCorrection');
    if (btnRunLiveCorrection) {
      btnRunLiveCorrection.addEventListener('click', executeLiveCorrectionSimulation);
    }
  }

  async function loadLiveSamplePresets() {
    const chipsContainer = document.getElementById('livePresetChips');
    if (!chipsContainer) return;
    try {
      const res = await fetch('/api/live/samples');
      const samples = await res.json();
      chipsContainer.innerHTML = '';

      samples.forEach(s => {
        const chip = document.createElement('button');
        chip.className = 'preset-chip';
        
        const langBadge = s.language === 'hi' ? '\u0939\u093F' : 'EN';
        chip.innerHTML = `<span class="badge ${s.language === 'hi' ? 'badge-purple' : 'badge-blue'}" style="font-size: 9px; padding: 1px 4px;">${langBadge}</span> <span>${escapeHtml(s.question)}</span>`;
        
        chip.addEventListener('click', () => {
          document.getElementById('liveQuestionInput').value = s.question;
          document.getElementById('liveLanguageSelect').value = s.language;
          executeLiveRAGQuery(s.expected_answer);
        });
        chipsContainer.appendChild(chip);
      });
    } catch (e) {
      console.error('Failed to load sample presets:', e);
    }
  }

  async function executeLiveRAGQuery(expectedAnswer = null) {
    const input = document.getElementById('liveQuestionInput');
    const question = input.value.trim();
    if (!question) {
      showToast('Please enter a research question.');
      return;
    }

    const lang = document.getElementById('liveLanguageSelect').value;
    const topK = parseInt(document.getElementById('liveTopKSelect').value, 10) || 3;
    const btn = document.getElementById('btnExecuteLiveQuery');

    btn.disabled = true;
    btn.textContent = 'Querying FAISS & LLM...';

    // Show loading state in answer blocks
    ['liveAnsA', 'liveAnsB', 'liveAnsC'].forEach(id => {
      document.getElementById(id).textContent = 'Generating answer...';
    });

    try {
      const res = await fetch('/api/live/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          top_k: topK,
          language: lang,
          expected_answer: expectedAnswer,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      renderLiveRAGResults(data);
      showToast('Live 3-variant query completed!');
    } catch (e) {
      console.error('Live RAG query error:', e);
      showToast('Query failed. Check server logs.');
      ['liveAnsA', 'liveAnsB', 'liveAnsC'].forEach(id => {
        document.getElementById(id).textContent = `Error: ${e.message}`;
      });
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run Live RAG';
    }
  }

  function renderLiveRAGResults(data) {
    const analysis = data.analysis || {};
    const variants = data.variants || {};

    // Diagnostics Banner
    const banner = document.getElementById('liveDiagnosticsBanner');
    banner.style.display = 'block';

    const statusBadge = document.getElementById('liveStatusBadge');
    statusBadge.textContent = (analysis.status || 'unknown').toUpperCase();
    if (analysis.status === 'recovered') {
      statusBadge.className = 'badge badge-emerald';
    } else if (analysis.status === 'resilient') {
      statusBadge.className = 'badge badge-blue';
    } else {
      statusBadge.className = 'badge';
      statusBadge.style.cssText = 'background: var(--accent-error-bg); color: var(--accent-error-text); border: 1px solid var(--accent-error-border);';
    }

    document.getElementById('liveRawF1').textContent = Number(analysis.raw_ocr_token_f1_vs_gt || 0).toFixed(2);
    document.getElementById('liveCorrF1').textContent = Number(analysis.corrected_ocr_token_f1_vs_gt || 0).toFixed(2);
    
    const retLat = variants.ground_truth?.retrieval_latency_sec ?? 0;
    const genLat = variants.ground_truth?.generation_latency_sec ?? 0;
    document.getElementById('liveRetLat').textContent = `${Number(retLat).toFixed(3)}s`;
    document.getElementById('liveGenLat').textContent = `${Number(genLat).toFixed(3)}s`;

    // Render Variant A (Ground Truth)
    const varA = variants.ground_truth || {};
    document.getElementById('liveAnsA').textContent = varA.answer || '(No answer produced)';
    renderChunksList(document.getElementById('liveChunksA'), varA.retrieved_chunks || []);

    // Render Variant B (Raw OCR)
    const varB = variants.raw_ocr || {};
    document.getElementById('liveAnsB').textContent = varB.answer || '(No answer produced)';
    renderChunksList(document.getElementById('liveChunksB'), varB.retrieved_chunks || []);

    // Render Variant C (Corrected OCR)
    const varC = variants.corrected_ocr || {};
    document.getElementById('liveAnsC').textContent = varC.answer || '(No answer produced)';
    renderChunksList(document.getElementById('liveChunksC'), varC.retrieved_chunks || []);
  }

  function renderChunksList(container, chunks) {
    container.innerHTML = '';
    if (!chunks || chunks.length === 0) {
      container.innerHTML = '<div style="color: var(--slate-muted); font-style: italic;">No passages retrieved.</div>';
      return;
    }

    chunks.forEach(c => {
      const item = document.createElement('div');
      item.style.cssText = 'background: var(--surface-container-low); border: 1px solid var(--border-hairline); border-radius: var(--radius-sm); padding: 8px 10px; margin-bottom: 8px;';
      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--slate-muted); font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;">
          <span>Passage #${c.rank} &bull; ${c.page_id}</span>
          <span style="color: var(--secondary); font-weight: 600;">Cosine: ${Number(c.score).toFixed(4)}</span>
        </div>
        <div style="font-size: 11px; line-height: 1.4; color: var(--on-surface);">${escapeHtml(c.text)}</div>
      `;
      container.appendChild(item);
    });
  }

  function populateLayerDocSelect() {
    const select = document.getElementById('layerDocSelect');
    if (!select || select.options.length > 0) return;
    state.documents.forEach(doc => {
      const opt = document.createElement('option');
      opt.value = doc.page_id;
      opt.textContent = `${doc.page_id} (${doc.language.toUpperCase()} &bull; ${doc.word_count} words)`;
      select.appendChild(opt);
    });
  }

  async function executeLivePreprocessingTest() {
    const pageId = document.getElementById('layerDocSelect').value;
    if (!pageId) return;

    const denoise = document.getElementById('prepDenoise').checked;
    const deskew = document.getElementById('prepDeskew').checked;
    const threshMethod = document.getElementById('prepMethod').value;
    const container = document.getElementById('preprocessResultContainer');

    container.innerHTML = '<div style="color: var(--secondary); font-size: 12px;">Running OpenCV transforms...</div>';

    try {
      const res = await fetch('/api/live/test-preprocessing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page_id: pageId,
          denoise: denoise,
          deskew: deskew,
          threshold_method: threshMethod,
        }),
      });

      if (!res.ok) throw new Error('Preprocessing failed');
      const data = await res.json();

      container.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 8px; width: 100%;">
          <span>Ops: <strong>${data.operations_applied.join(' &rarr; ')}</strong></span>
          <span>Time: <strong>${data.execution_time_sec}s</strong></span>
        </div>
        <img src="${data.comparison_url}?t=${Date.now()}" style="max-width: 100%; max-height: 240px; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline);" alt="Comparison" />
      `;
      showToast('OpenCV Preprocessing updated!');
    } catch (e) {
      container.innerHTML = `<div style="color: var(--error); font-size: 12px;">Preprocessing failed: ${e.message}</div>`;
    }
  }

  async function executeLiveCorrectionSimulation() {
    const pageId = document.getElementById('layerDocSelect').value;
    if (!pageId) return;

    const tau = parseFloat(document.getElementById('liveTauSlider').value) || 70.0;
    const ctxWin = parseInt(document.getElementById('liveCtxSlider').value, 10) || 5;
    const statsContainer = document.getElementById('liveCorrectionStats');
    const spansContainer = document.getElementById('liveSpansContainer');

    statsContainer.style.display = 'block';
    spansContainer.innerHTML = '<div style="color: var(--secondary); font-size: 12px; text-align: center; padding: 12px;">Simulating selective correction...</div>';

    try {
      const res = await fetch('/api/live/test-correction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page_id: pageId,
          threshold: tau,
          context_window: ctxWin,
        }),
      });

      if (!res.ok) throw new Error('Correction simulation failed');
      const data = await res.json();

      document.getElementById('liveFlaggedSpans').textContent = data.flagged_spans_count;
      document.getElementById('liveFlaggedPct').textContent = `${data.flagged_percentage}%`;
      document.getElementById('liveTokenSavings').textContent = `${data.efficiency.token_savings_pct}%`;

      spansContainer.innerHTML = '';
      if (!data.spans || data.spans.length === 0) {
        spansContainer.innerHTML = `<div style="color: var(--slate-muted); font-style: italic; text-align: center; padding: 12px;">No words fell below &tau; = ${tau}. Corpus is clean.</div>`;
        return;
      }

      data.spans.forEach(s => {
        const item = document.createElement('div');
        item.style.cssText = 'background: var(--surface-container-low); border: 1px solid var(--border-hairline); border-radius: var(--radius-sm); padding: 8px 10px; margin-bottom: 6px;';
        item.innerHTML = `
          <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--slate-muted); margin-bottom: 4px;">
            <span>Span: <code style="color: var(--error); font-weight: 600;">${escapeHtml(s.original_text)}</code> &rarr; <code style="color: var(--accent-mitigation-text); font-weight: 600;">${escapeHtml(s.corrected_text)}</code></span>
            <span>Avg Conf: <strong>${s.avg_confidence}</strong></span>
          </div>
          <div style="font-size: 11px; color: var(--slate-muted); font-style: italic;">
            ... ${escapeHtml(s.context_before)} <mark style="background: #fef08a; padding: 1px 3px; border-radius: 2px;">${escapeHtml(s.original_text)}</mark> ${escapeHtml(s.context_after)} ...
          </div>
        `;
        spansContainer.appendChild(item);
      });

      showToast(`Simulation complete: ${data.flagged_spans_count} spans flagged!`);
    } catch (e) {
      spansContainer.innerHTML = `<div style="color: var(--error); font-size: 12px;">Simulation failed: ${e.message}</div>`;
    }
  }

  // ============================================================================
  // Mode 3: Upload & Ingest Own Document Logic
  // ============================================================================

  let selectedUploadFile = null;

  function initUploadModule() {
    const tabUploadFile = document.getElementById('tabUploadFile');
    const tabUploadPaste = document.getElementById('tabUploadPaste');
    const sectionUploadFile = document.getElementById('sectionUploadFile');
    const sectionUploadPaste = document.getElementById('sectionUploadPaste');

    if (!tabUploadFile || !tabUploadPaste) return;

    tabUploadFile.addEventListener('click', () => {
      tabUploadFile.classList.add('active');
      tabUploadPaste.classList.remove('active');
      sectionUploadFile.style.display = 'block';
      sectionUploadPaste.style.display = 'none';
    });

    tabUploadPaste.addEventListener('click', () => {
      tabUploadPaste.classList.add('active');
      tabUploadFile.classList.remove('active');
      sectionUploadFile.style.display = 'none';
      sectionUploadPaste.style.display = 'block';
    });

    const dropzoneArea = document.getElementById('dropzoneArea');
    const fileUploadInput = document.getElementById('fileUploadInput');
    const selectedFileName = document.getElementById('selectedFileName');

    if (dropzoneArea && fileUploadInput) {
      dropzoneArea.addEventListener('click', () => fileUploadInput.click());

      dropzoneArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzoneArea.style.borderColor = 'var(--secondary)';
        dropzoneArea.style.background = 'var(--surface-container)';
      });

      ['dragleave', 'dragend'].forEach(type => {
        dropzoneArea.addEventListener(type, () => {
          dropzoneArea.style.borderColor = 'var(--outline-variant)';
          dropzoneArea.style.background = 'var(--surface-container-low)';
        });
      });

      dropzoneArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzoneArea.style.borderColor = 'var(--outline-variant)';
        dropzoneArea.style.background = 'var(--surface-container-low)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          selectedUploadFile = e.dataTransfer.files[0];
          showSelectedFile(selectedUploadFile.name);
        }
      });

      fileUploadInput.addEventListener('change', () => {
        if (fileUploadInput.files && fileUploadInput.files.length > 0) {
          selectedUploadFile = fileUploadInput.files[0];
          showSelectedFile(selectedUploadFile.name);
        }
      });
    }

    function showSelectedFile(name) {
      if (selectedFileName) {
        selectedFileName.textContent = `Selected: ${name}`;
        selectedFileName.style.display = 'block';
      }
    }

    const uploadThresholdInput = document.getElementById('uploadThresholdInput');
    const uploadTauLabel = document.getElementById('uploadTauLabel');
    if (uploadThresholdInput && uploadTauLabel) {
      uploadThresholdInput.addEventListener('input', () => {
        uploadTauLabel.textContent = Number(uploadThresholdInput.value).toFixed(1);
      });
    }

    const btnSubmitUpload = document.getElementById('btnSubmitUpload');
    if (btnSubmitUpload) {
      btnSubmitUpload.addEventListener('click', executeDocumentUpload);
    }

    const btnAskOnUploadedDoc = document.getElementById('btnAskOnUploadedDoc');
    if (btnAskOnUploadedDoc) {
      btnAskOnUploadedDoc.addEventListener('click', () => {
        const sec = document.getElementById('dedicatedDocQnASection');
        if (sec) sec.scrollIntoView({ behavior: 'smooth' });
        const input = document.getElementById('activeDocQueryInput');
        if (input) input.focus();
        showToast('Dedicated Q&A console for this document ready below!');
      });
    }

    const btnToggleDocPreview = document.getElementById('btnToggleDocPreview');
    const activeDocPreviewWrap = document.getElementById('activeDocPreviewWrap');
    if (btnToggleDocPreview && activeDocPreviewWrap) {
      btnToggleDocPreview.addEventListener('click', () => {
        const isHidden = activeDocPreviewWrap.style.display === 'none';
        activeDocPreviewWrap.style.display = isHidden ? 'block' : 'none';
        btnToggleDocPreview.textContent = isHidden ? 'Hide Text Preview' : 'Toggle Text Preview';
      });
    }

    const btnExecuteDocQuery = document.getElementById('btnExecuteDocQuery');
    if (btnExecuteDocQuery) {
      btnExecuteDocQuery.addEventListener('click', () => {
        if (!activeCustomDoc) return;
        const input = document.getElementById('activeDocQueryInput');
        const q = input.value.trim();
        if (!q) {
          showToast('Please type a question for this document.');
          return;
        }
        executeActiveDocQuery(activeCustomDoc.page_id, q);
      });
    }

    const activeDocQueryInput = document.getElementById('activeDocQueryInput');
    if (activeDocQueryInput) {
      activeDocQueryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const btn = document.getElementById('btnExecuteDocQuery');
          if (btn) btn.click();
        }
      });
    }

    loadCustomDocumentsList();
  }

  let customDocumentsList = [];
  let activeCustomDoc = null;

  async function loadCustomDocumentsList(selectPageId = null) {
    try {
      const res = await fetch('/api/live/custom-documents');
      if (!res.ok) return;
      customDocumentsList = await res.json();

      const countBadge = document.getElementById('docQnACountBadge');
      if (countBadge) {
        countBadge.textContent = `${customDocumentsList.length} Document${customDocumentsList.length === 1 ? '' : 's'} Ingested`;
      }

      const tabsContainer = document.getElementById('docSelectorTabs');
      if (!tabsContainer) return;

      tabsContainer.innerHTML = '';

      if (customDocumentsList.length === 0) {
        tabsContainer.innerHTML = `
          <div style="color: var(--slate-muted); font-size: 13px; font-style: italic;">
            No custom documents uploaded yet. Ingest a document above to unlock its dedicated Q&amp;A console.
          </div>
        `;
        const consoleEl = document.getElementById('activeDocConsole');
        if (consoleEl) consoleEl.style.display = 'none';
        activeCustomDoc = null;
        return;
      }

      // Determine active document
      if (selectPageId) {
        activeCustomDoc = customDocumentsList.find(d => d.page_id === selectPageId) || customDocumentsList[customDocumentsList.length - 1];
      } else if (!activeCustomDoc || !customDocumentsList.some(d => d.page_id === activeCustomDoc.page_id)) {
        activeCustomDoc = customDocumentsList[customDocumentsList.length - 1];
      }

      customDocumentsList.forEach(doc => {
        const btn = document.createElement('button');
        const isActive = activeCustomDoc && doc.page_id === activeCustomDoc.page_id;
        btn.className = `custom-select lab-mode-btn ${isActive ? 'active' : ''}`;
        btn.style.cssText = 'padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px;';
        btn.innerHTML = `📄 ${escapeHtml(doc.title || doc.page_id)} <span class="badge ${doc.language === 'hi' ? 'badge-purple' : 'badge-blue'}" style="font-size: 10px;">${(doc.language || 'en').toUpperCase()}</span>`;
        btn.addEventListener('click', () => {
          activeCustomDoc = doc;
          document.querySelectorAll('#docSelectorTabs .lab-mode-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderActiveDocConsole(doc);
        });
        tabsContainer.appendChild(btn);
      });

      if (activeCustomDoc) {
        renderActiveDocConsole(activeCustomDoc);
      }
    } catch (e) {
      console.error('Failed to load custom documents:', e);
    }
  }

  async function renderActiveDocConsole(doc) {
    const consoleEl = document.getElementById('activeDocConsole');
    if (!consoleEl) return;
    consoleEl.style.display = 'block';

    document.getElementById('activeDocTitle').textContent = doc.title || doc.page_id;
    document.getElementById('activeDocLangBadge').textContent = (doc.language || 'en').toUpperCase();
    document.getElementById('activeDocIdBadge').textContent = doc.page_id;
    document.getElementById('activeDocSubtitle').textContent =
      `Single-column • Total Words: ${doc.word_count || 0} • Flagged Low-Confidence Spans: ${doc.flagged_spans_count || 0}`;

    const queryInput = document.getElementById('activeDocQueryInput');
    queryInput.value = '';
    queryInput.placeholder = `Ask a question specifically about "${doc.title || doc.page_id}"...`;

    // Reset results grid
    document.getElementById('activeDocDiagnosticsBanner').style.display = 'none';
    document.getElementById('activeDocAnsA').textContent = '(Click Ask Document or select a suggested question)';
    document.getElementById('activeDocAnsB').textContent = '(Click Ask Document or select a suggested question)';
    document.getElementById('activeDocAnsC').textContent = '(Click Ask Document or select a suggested question)';
    document.getElementById('activeDocChunksA').innerHTML = '<div style="color: var(--slate-muted); font-style: italic;">No passages retrieved yet.</div>';
    document.getElementById('activeDocChunksB').innerHTML = '<div style="color: var(--slate-muted); font-style: italic;">No passages retrieved yet.</div>';
    document.getElementById('activeDocChunksC').innerHTML = '<div style="color: var(--slate-muted); font-style: italic;">No passages retrieved yet.</div>';

    // Populate suggested questions
    const chipsContainer = document.getElementById('activeDocSuggestedChips');
    chipsContainer.innerHTML = '';
    const sampleQuestions = doc.sample_questions || [];

    if (sampleQuestions.length === 0) {
      chipsContainer.innerHTML = '<div style="color: var(--slate-muted); font-size: 12px; font-style: italic;">Type any question below to test live triangulation on this document.</div>';
    } else {
      sampleQuestions.forEach(qItem => {
        const chip = document.createElement('div');
        chip.className = 'preset-chip';
        chip.textContent = qItem.question;
        chip.title = 'Click to query this question';
        chip.addEventListener('click', () => {
          queryInput.value = qItem.question;
          executeActiveDocQuery(doc.page_id, qItem.question, qItem.expected_answer);
        });
        chipsContainer.appendChild(chip);
      });
    }

    // Load document preview text in background
    try {
      const res = await fetch(`/api/document/${doc.page_id}`);
      if (res.ok) {
        const dData = await res.json();
        document.getElementById('activeDocGtText').textContent = dData.ground_truth_text || '(No ground truth recorded)';
        document.getElementById('activeDocRawText').textContent = dData.raw_ocr_text || '(No OCR text recorded)';
      }
    } catch (e) {
      console.warn('Could not load text preview:', e);
    }
  }

  async function executeActiveDocQuery(pageId, questionText, expectedAnswer = null) {
    const btn = document.getElementById('btnExecuteDocQuery');
    const topK = parseInt(document.getElementById('activeDocTopKSelect').value) || 2;

    btn.disabled = true;
    btn.textContent = 'Querying Document...';

    try {
      const res = await fetch('/api/live/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: questionText,
          top_k: topK,
          page_id: pageId,
          expected_answer: expectedAnswer,
        }),
      });

      if (!res.ok) throw new Error(`Query failed with status ${res.status}`);
      const data = await res.json();

      // Render Diagnostic Banner
      const diagBanner = document.getElementById('activeDocDiagnosticsBanner');
      diagBanner.style.display = 'block';

      const statusBadge = document.getElementById('activeDocStatusBadge');
      const st = data.analysis.status;
      if (st === 'recovered') {
        statusBadge.textContent = 'RECOVERED BY VARIANT C';
        statusBadge.className = 'badge badge-emerald';
      } else if (st === 'resilient') {
        statusBadge.textContent = 'NOISE RESILIENT';
        statusBadge.className = 'badge badge-blue';
      } else {
        statusBadge.textContent = 'UNRECOVERED NOISE';
        statusBadge.className = 'badge badge-red';
      }

      document.getElementById('activeDocRawF1').textContent = data.analysis.raw_ocr_token_f1_vs_gt;
      document.getElementById('activeDocCorrF1').textContent = data.analysis.corrected_ocr_token_f1_vs_gt;
      document.getElementById('activeDocLatency').textContent = `${data.variants.corrected_ocr.total_latency_sec}s`;

      // Render Answers
      document.getElementById('activeDocAnsA').textContent = data.variants.ground_truth.answer;
      document.getElementById('activeDocAnsB').textContent = data.variants.raw_ocr.answer;
      document.getElementById('activeDocAnsC').textContent = data.variants.corrected_ocr.answer;

      // Render Retrieved Chunks
      const renderDocChunks = (containerId, chunks) => {
        const c = document.getElementById(containerId);
        c.innerHTML = '';
        if (!chunks || chunks.length === 0) {
          c.innerHTML = '<div style="color: var(--slate-muted); font-style: italic;">No passages retrieved.</div>';
          return;
        }
        chunks.forEach(item => {
          const div = document.createElement('div');
          div.style.cssText = 'background: var(--surface-container-low); padding: 8px; border-radius: 4px; margin-bottom: 6px; border-left: 2px solid var(--secondary); font-size: 11px;';
          div.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--slate-muted); margin-bottom: 2px;">
              <span>Score: <strong>${item.score}</strong></span>
              <span>Passage #${item.rank}</span>
            </div>
            <div style="line-height: 1.35; color: var(--on-surface);">${escapeHtml(item.text)}</div>
          `;
          c.appendChild(div);
        });
      };

      renderDocChunks('activeDocChunksA', data.variants.ground_truth.retrieved_chunks);
      renderDocChunks('activeDocChunksB', data.variants.raw_ocr.retrieved_chunks);
      renderDocChunks('activeDocChunksC', data.variants.corrected_ocr.retrieved_chunks);

      showToast('Document-specific 3-variant triangulation complete!');
    } catch (e) {
      console.error('Doc query error:', e);
      showToast(`Query failed: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Ask Document \u2192';
    }
  }

  async function executeDocumentUpload() {
    const isFileMode = document.getElementById('tabUploadFile').classList.contains('active');
    const titleInput = document.getElementById('uploadTitleInput').value.trim();
    const langSelect = document.getElementById('uploadLanguageSelect').value;
    const tau = parseFloat(document.getElementById('uploadThresholdInput').value) || 70.0;
    const btn = document.getElementById('btnSubmitUpload');

    const steps = [
      document.getElementById('stepPreprocess'),
      document.getElementById('stepOCR'),
      document.getElementById('stepCorrection'),
      document.getElementById('stepFAISS'),
    ];
    const stepperBadge = document.getElementById('uploadStepperBadge');
    const resultContainer = document.getElementById('uploadResultContainer');

    steps.forEach(s => {
      s.style.color = 'var(--slate-muted)';
      s.querySelector('.stepper-icon').textContent = '\u2022';
    });
    stepperBadge.textContent = 'Processing...';
    stepperBadge.className = 'badge badge-purple';
    resultContainer.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Ingesting Document...';

    const setStepActive = (idx) => {
      if (idx > 0) {
        steps[idx - 1].style.color = 'var(--accent-mitigation-text)';
        steps[idx - 1].querySelector('.stepper-icon').textContent = '\u2713';
      }
      if (idx < steps.length) {
        steps[idx].style.color = 'var(--secondary)';
        steps[idx].querySelector('.stepper-icon').textContent = '\u2192';
      }
    };

    setStepActive(0);

    try {
      let res;
      if (isFileMode) {
        if (!selectedUploadFile) {
          showToast('Please select or drop a file first.');
          btn.disabled = false;
          btn.textContent = 'Run Pipeline & Ingest into RAG';
          stepperBadge.textContent = 'Idle';
          stepperBadge.className = 'badge badge-emerald';
          return;
        }

        setTimeout(() => setStepActive(1), 300);
        setTimeout(() => setStepActive(2), 600);

        const formData = new FormData();
        formData.append('file', selectedUploadFile);
        if (titleInput) formData.append('title', titleInput);
        formData.append('language', langSelect);
        formData.append('threshold', tau);

        res = await fetch('/api/live/upload-file', {
          method: 'POST',
          body: formData,
        });
      } else {
        const textContent = document.getElementById('pasteTextInput').value.trim();
        if (!textContent) {
          showToast('Please paste document text content.');
          btn.disabled = false;
          btn.textContent = 'Run Pipeline & Ingest into RAG';
          stepperBadge.textContent = 'Idle';
          stepperBadge.className = 'badge badge-emerald';
          return;
        }

        setTimeout(() => setStepActive(1), 300);
        setTimeout(() => setStepActive(2), 600);

        res = await fetch('/api/live/upload-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: titleInput || 'Pasted Document',
            text: textContent,
            language: langSelect,
            threshold: tau,
          }),
        });
      }

      if (!res.ok) throw new Error(`Upload server error: ${res.status}`);
      const data = await res.json();

      setStepActive(3);
      setTimeout(() => {
        steps[3].style.color = 'var(--accent-mitigation-text)';
        steps[3].querySelector('.stepper-icon').textContent = '\u2713';
      }, 300);

      stepperBadge.textContent = 'Indexed in FAISS';
      stepperBadge.className = 'badge badge-emerald';

      resultContainer.style.display = 'block';
      document.getElementById('uploadResultTitle').textContent = data.title || data.page_id;
      document.getElementById('uploadResultLang').textContent = (data.language || 'en').toUpperCase();
      document.getElementById('uploadWordCount').textContent = data.word_count;
      document.getElementById('uploadFlaggedSpans').textContent = data.flagged_spans_count;
      document.getElementById('uploadChunksCount').textContent = data.chunks_indexed?.ground_truth || data.chunks_indexed?.raw_ocr || 1;
      document.getElementById('uploadRawSnippet').textContent = data.raw_ocr_snippet || '(None)';
      document.getElementById('uploadCorrSnippet').textContent = data.corrected_snippet || '(None)';

      loadDocumentsList();
      populateLayerDocSelect();
      await loadCustomDocumentsList(data.page_id);

      const sec = document.getElementById('dedicatedDocQnASection');
      if (sec) sec.scrollIntoView({ behavior: 'smooth' });

      showToast('Document successfully indexed! Dedicated Q&A section ready below.');
    } catch (e) {
      console.error('Upload error:', e);
      showToast(`Upload failed: ${e.message}`);
      stepperBadge.textContent = 'Error';
      stepperBadge.className = 'badge badge-red';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run Pipeline & Ingest into RAG';
    }
  }

  // ============================================================================
  // Initial Boot
  // ============================================================================

  loadOverview();
  loadDocumentsList();
  loadCorrectionsAuditTrail();
  loadQuestions();
  initLiveLab();
  loadPublicationTables();
});

