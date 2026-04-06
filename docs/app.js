/* ═══════════════════════════════════════════════
   MemMe — Neural Cartography
   Interactions & Visualizations
   ═══════════════════════════════════════════════ */

// ── Pipeline Step Data ──
const pipelineSteps = [
  {
    title: 'Events',
    badge: 'append_events()',
    desc: 'Raw conversational data streams in. Each event carries content, role, timestamp, and session ID. No embedding computed yet \u2014 raw provenance preserved.',
    transform: {
      label_in: 'Raw Input',
      input: '"Hey Alice, are we still meeting at the Shanghai office tomorrow?"',
      label_out: 'Stored Event',
      output: '{ role: "user", content: "Hey Alice...", session: "s_7f3a", time: "2026-04-02T14:30:00Z" }'
    },
    code: `<span class="fn">store</span>.<span class="fn">append_events</span>(
  [{ <span class="kw">role</span>: <span class="str">"user"</span>, <span class="kw">content</span>: <span class="str">"Hey Alice..."</span> }],
  <span class="kw">session_id</span>=<span class="str">"s_7f3a"</span>,
  <span class="kw">user_id</span>=<span class="str">"u_001"</span>
)`
  },
  {
    title: 'Sessions',
    badge: 'automatic grouping',
    desc: 'Events are grouped into sessions by session_id. A session represents one continuous interaction \u2014 a chat window, a meeting, a phone call. When the event count reaches compact_threshold (default 20), compaction is triggered.',
    transform: {
      label_in: '12 raw events',
      input: 'e_01 "Hey Alice..." \u2192 e_02 "Sure, 3pm works" \u2192 e_03 "BTW she said..." \u2192 ... \u2192 e_12 "See you tomorrow"',
      label_out: 'Session',
      output: 'Session { id: "s_7f3a", events: 12, started: "14:30", ended: "14:47", status: "active" }'
    },
    code: `<span class="cmt">// Automatic lifecycle</span>
<span class="kw">first event</span>  \u2192  session created  \u2192  events accumulate
                                        \u2502
              compact_threshold (20)  \u2500\u2500\u2518  \u2192  triggers compact`
  },
  {
    title: 'Compact',
    badge: 'compact(session_id)',
    desc: 'LLM-powered purification. Three transformations happen: pronoun resolution ("she" \u2192 "Alice"), temporal disambiguation ("tomorrow" \u2192 "2026-04-03"), location grounding ("there" \u2192 "Shanghai office"). Embeddings computed on purified text.',
    transform: {
      label_in: 'Before (raw)',
      input: '"She said we should meet there tomorrow to discuss it"',
      label_out: 'After (purified)',
      output: '"Alice said we should meet at the Shanghai office on 2026-04-03 to discuss the Q2 roadmap"'
    },
    code: `<span class="fn">store</span>.<span class="fn">compact</span>(<span class="kw">session_id</span>=<span class="str">"s_7f3a"</span>)
<span class="cmt">// Pronoun:  "she"       \u2192 "Alice"</span>
<span class="cmt">// Temporal: "tomorrow"  \u2192 "2026-04-03"</span>
<span class="cmt">// Location: "there"     \u2192 "Shanghai office"</span>
<span class="cmt">// + Embedding computed on purified text</span>`
  },
  {
    title: 'Episodes',
    badge: 'narrative traces',
    desc: 'Compact groups related events into episodes \u2014 coherent narrative units with title, summary, and significance score. Think of them as "chapters" in the user\'s story. Each episode tracks which events contributed to it.',
    transform: {
      label_in: '12 purified events',
      input: 'Meeting planning, Q2 discussion, Shanghai trip, team updates...',
      label_out: 'Episode',
      output: '{ title: "Q2 Planning with Alice", summary: "Discussed roadmap, Shanghai office meeting on Apr 3", significance: 0.85, events: 12 }'
    },
    code: `Episode {
  <span class="kw">id</span>:           <span class="str">"ep_a1b2"</span>,
  <span class="kw">title</span>:        <span class="str">"Q2 Planning with Alice"</span>,
  <span class="kw">summary</span>:      <span class="str">"Discussed roadmap and Shanghai trip"</span>,
  <span class="kw">significance</span>: <span class="str">0.85</span>,
  <span class="kw">event_ids</span>:    [<span class="str">"e_01"</span>, <span class="str">"e_02"</span>, ... <span class="str">"e_12"</span>]
}`
  },
  {
    title: 'Meditate',
    badge: 'meditate(user_id)',
    desc: 'The orchestrator. Four phases run in sequence: decay old memories, extract atomic facts from each episode, store via add() with vector dedup (cosine similarity handles duplicates automatically), then build knowledge graph and link entities via Aho-Corasick.',
    transform: {
      label_in: '1 episode, 47 existing memories',
      input: 'Episode "Q2 Planning with Alice" + 47 existing memories in store',
      label_out: 'Result',
      output: '3 memories created, 1 deduped (similar to existing), 2 entities linked, 1 relation added'
    },
    code: `<span class="fn">store</span>.<span class="fn">meditate</span>(<span class="kw">user_id</span>=<span class="str">"u_001"</span>)
<span class="cmt">// Phase 1: Decay \u2014 47 memories, 3 below threshold</span>
<span class="cmt">// Phase 2: Extract \u2014 4 atomic facts from episode</span>
<span class="cmt">// Phase 3: Store + Dedup \u2014 3 new, 1 duplicate skipped</span>
<span class="cmt">// Phase 4: Graph + Link \u2014 2 entities, 1 relation</span>`
  },
  {
    title: 'Memories',
    badge: 'atomic facts',
    desc: 'The final product. Each memory is an atomic, self-contained fact with its own embedding vector, importance score, stability (for forgetting curve), temporal metadata, and entity links. Searchable, updatable, and naturally forgettable over time.',
    transform: {
      label_in: 'Extracted fact',
      input: '"Alice plans to visit Shanghai"',
      label_out: 'Stored memory',
      output: '{ content: "Alice plans to visit Shanghai office on 2026-04-03", importance: 0.85, stability: 1.0, embedding: [0.012, -0.034, ...], entities: ["Alice", "Shanghai"] }'
    },
    code: `Memory {
  <span class="kw">id</span>:         <span class="str">"m_x9k2"</span>,
  <span class="kw">content</span>:    <span class="str">"Alice plans to visit Shanghai office on 2026-04-03"</span>,
  <span class="kw">importance</span>: <span class="str">0.85</span>,
  <span class="kw">stability</span>:  <span class="str">1.0</span>,          <span class="cmt">// grows with each access</span>
  <span class="kw">embedding</span>:  [<span class="str">0.012, -0.034, ...</span>],  <span class="cmt">// 1536d vector</span>
  <span class="kw">event_time</span>: <span class="str">"2026-04-02"</span>,
  <span class="kw">entities</span>:   [<span class="str">"Alice"</span>, <span class="str">"Shanghai"</span>]
}`
  },
  {
    title: 'Graph',
    badge: 'knowledge graph',
    desc: 'Entities (people, places, organizations, concepts) and their relationships extracted alongside memories. Enables multi-hop reasoning through spreading activation \u2014 query "Alice" and discover related entities within 2 hops.',
    transform: {
      label_in: 'Memory text',
      input: '"Alice plans to visit Shanghai office on 2026-04-03 to discuss Q2 roadmap"',
      label_out: 'Extracted graph',
      output: 'Entities: [Alice (person), Shanghai (place), Q2 roadmap (concept)]\nRelations: Alice \u2192plans_visit\u2192 Shanghai, Alice \u2192works_on\u2192 Q2 roadmap'
    },
    code: `<span class="cmt">// Extracted in a single LLM call</span>
<span class="kw">entities</span>: [
  { <span class="kw">name</span>: <span class="str">"Alice"</span>,      <span class="kw">type</span>: <span class="str">"person"</span> },
  { <span class="kw">name</span>: <span class="str">"Shanghai"</span>,   <span class="kw">type</span>: <span class="str">"place"</span>  },
  { <span class="kw">name</span>: <span class="str">"Q2 roadmap"</span>, <span class="kw">type</span>: <span class="str">"concept"</span> }
]
<span class="kw">relations</span>: [
  (<span class="str">"Alice"</span>, <span class="kw">plans_visit</span>, <span class="str">"Shanghai"</span>),
  (<span class="str">"Alice"</span>, <span class="kw">works_on</span>,    <span class="str">"Q2 roadmap"</span>)
]`
  },
  {
    title: 'Identity',
    badge: 'personality traits',
    desc: 'High-level behavioral patterns and personality traits distilled from the entire memory corpus. Used to enhance personalization, provide context in RAG pipelines, and tailor response generation to the user\'s known preferences and habits.',
    transform: {
      label_in: '50 memories analyzed',
      input: 'Memories about meetings, communication style, travel, collaboration patterns...',
      label_out: 'Identity traits',
      output: '["Prefers concise communication", "Frequently collaborates with Alice", "Regular visitor to Shanghai", "Focused on Q2 planning"]'
    },
    code: `Identity {
  <span class="kw">user_id</span>: <span class="str">"u_001"</span>,
  <span class="kw">traits</span>: [
    <span class="str">"Prefers concise, direct communication"</span>,
    <span class="str">"Frequently collaborates with Alice on projects"</span>,
    <span class="str">"Regular visitor to Shanghai office"</span>,
    <span class="str">"Currently focused on Q2 2026 planning"</span>
  ],
  <span class="kw">updated_at</span>: <span class="str">"2026-04-02T15:00:00Z"</span>
}`
  }
];

// ── Hero Particle Canvas ──
function initHeroCanvas() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  let w, h;
  let particles = [];
  let mouse = { x: -1000, y: -1000 };
  let raf;

  function resize() {
    w = canvas.parentElement.offsetWidth;
    h = canvas.parentElement.offsetHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
  }

  function createParticles() {
    const count = Math.min(80, Math.floor(w * h / 12000));
    particles = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.4 + 0.1,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);

    // Gradient background
    const grd = ctx.createRadialGradient(w * 0.5, h * 0.4, 0, w * 0.5, h * 0.4, w * 0.7);
    grd.addColorStop(0, 'rgba(139, 124, 246, 0.06)');
    grd.addColorStop(1, 'transparent');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(139, 124, 246, ${0.12 * (1 - dist / 150)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw and update particles
    for (const p of particles) {
      // Mouse interaction
      const mdx = p.x - mouse.x;
      const mdy = p.y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < 200) {
        const force = (200 - mdist) / 200 * 0.02;
        p.vx += mdx / mdist * force;
        p.vy += mdy / mdist * force;
      }

      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.99;
      p.vy *= 0.99;

      // Wrap
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(139, 124, 246, ${p.opacity * 1.5})`;
      ctx.fill();
    }

    raf = requestAnimationFrame(draw);
  }

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
  });

  canvas.addEventListener('mouseleave', () => {
    mouse.x = -1000;
    mouse.y = -1000;
  });

  resize();
  createParticles();
  draw();

  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    resize();
    createParticles();
    draw();
  });
}

// ── Pipeline Interaction ──
function initPipeline() {
  const nodes = document.querySelectorAll('.pipe-stage');
  const connectors = document.querySelectorAll('.pipe-connector');
  const detail = document.getElementById('pipelineDetail');
  const playBtn = document.getElementById('playPause');
  const prevBtn = document.getElementById('prevStep');
  const nextBtn = document.getElementById('nextStep');
  const stepIndicator = document.getElementById('stepIndicator');
  const iconPlay = playBtn.querySelector('.icon-play');
  const iconPause = playBtn.querySelector('.icon-pause');
  const speedBtns = document.querySelectorAll('.speed-btn');

  let current = 0;
  let interval = null;
  let playing = false;
  let speed = 1;
  const baseInterval = 4000;

  let prevStepIdx = 0;

  function firePacket(connectorIdx) {
    if (connectorIdx < 0 || connectorIdx >= connectors.length) return;
    const conn = connectors[connectorIdx];
    conn.classList.remove('firing');
    void conn.offsetWidth; // reflow to restart animation
    conn.classList.add('firing');
    setTimeout(() => conn.classList.remove('firing'), 500);
  }

  function showStep(idx) {
    const direction = idx > prevStepIdx ? 1 : -1;

    // Fire data packet along the connector between prev and current
    if (idx !== prevStepIdx) {
      const connIdx = direction > 0 ? prevStepIdx : idx;
      firePacket(connIdx);
    }

    prevStepIdx = idx;
    current = idx;
    nodes.forEach((n, i) => {
      n.classList.toggle('active', i === idx);
      n.classList.toggle('visited', i < idx);
    });
    connectors.forEach((c, i) => {
      c.classList.toggle('visited', i < idx);
    });

    const step = pipelineSteps[idx];

    // Directional fade-slide
    detail.style.opacity = '0';
    detail.style.transform = `translateY(${direction * 14}px)`;

    setTimeout(() => {
      const t = step.transform;
      detail.innerHTML = `
        <div class="detail-top">
          <div>
            <div class="detail-badge">${step.badge}</div>
            <h3>${step.title}</h3>
            <p>${step.desc}</p>
          </div>
        </div>
        <div class="detail-transform">
          <div class="transform-box transform-in">
            <small>${t.label_in}</small>
            <div class="transform-content">${t.input}</div>
          </div>
          <div class="transform-arrow">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
          </div>
          <div class="transform-box transform-out">
            <small>${t.label_out}</small>
            <div class="transform-content">${t.output}</div>
          </div>
        </div>
        <div class="detail-code">${step.code}</div>
      `;
      detail.style.opacity = '1';
      detail.style.transform = 'translateY(0)';
    }, 200);

    stepIndicator.textContent = idx + 1;

  }

  // Add transition to detail
  detail.style.transition = 'opacity 0.25s ease, transform 0.25s ease';

  function startAutoPlay() {
    if (interval) clearInterval(interval);
    playing = true;
    playBtn.classList.add('playing');
    iconPlay.style.display = 'none';
    iconPause.style.display = 'block';
    interval = setInterval(() => {
      current = (current + 1) % pipelineSteps.length;
      showStep(current);
    }, baseInterval / speed);
  }

  function stopAutoPlay() {
    playing = false;
    playBtn.classList.remove('playing');
    iconPlay.style.display = 'block';
    iconPause.style.display = 'none';
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
  }

  playBtn.addEventListener('click', () => {
    playing ? stopAutoPlay() : startAutoPlay();
  });

  prevBtn.addEventListener('click', () => {
    stopAutoPlay();
    current = (current - 1 + pipelineSteps.length) % pipelineSteps.length;
    showStep(current);
  });

  nextBtn.addEventListener('click', () => {
    stopAutoPlay();
    current = (current + 1) % pipelineSteps.length;
    showStep(current);
  });

  speedBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      speedBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      speed = parseFloat(btn.dataset.speed);
      if (playing) startAutoPlay();
    });
  });

  nodes.forEach((node) => {
    node.addEventListener('click', () => {
      stopAutoPlay();
      showStep(parseInt(node.dataset.step));
    });
  });

  // Keyboard
  document.addEventListener('keydown', (e) => {
    const section = document.getElementById('pipeline');
    const rect = section.getBoundingClientRect();
    if (rect.top > window.innerHeight || rect.bottom < 0) return;

    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      stopAutoPlay();
      current = (current - 1 + pipelineSteps.length) % pipelineSteps.length;
      showStep(current);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      stopAutoPlay();
      current = (current + 1) % pipelineSteps.length;
      showStep(current);
    } else if (e.key === ' ') {
      const active = document.activeElement;
      if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;
      e.preventDefault();
      playing ? stopAutoPlay() : startAutoPlay();
    }
  });

  showStep(0);
}

// ── Forgetting Curve Canvas ──
function initForgettingCurve() {
  const canvas = document.getElementById('forgettingCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 48; // padding
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    canvas.width = w * dpr;
    canvas.height = 320 * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = '320px';
    ctx.scale(dpr, dpr);
  }

  const slider = document.getElementById('stabilitySlider');
  const valueDisplay = document.getElementById('stabilityValue');

  let animatedS = 5;
  let targetS = 5;
  let curveRaf = null;

  function retention(t, S) {
    return Math.pow(1 + t / (5.0 * S), -0.5);
  }

  function draw() {
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const pad = { top: 25, right: 40, bottom: 45, left: 48 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + plotH * (1 - i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#4e5264';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText((i * 25) + '%', pad.left - 8, y + 3);
    }

    // X-axis
    [0, 7, 14, 21, 30].forEach(d => {
      const x = pad.left + (d / 30) * plotW;
      ctx.fillStyle = '#4e5264';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(d + 'd', x, h - pad.bottom + 18);
    });

    // Labels
    ctx.fillStyle = '#8b8fa0';
    ctx.font = '11px "DM Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Days since last access', pad.left + plotW / 2, h - 6);

    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Retention', 0, 0);
    ctx.restore();

    const S = animatedS;

    // Reference curves
    const refs = [
      { s: 1,  color: 'rgba(248, 113, 113, 0.25)' },
      { s: 5,  color: 'rgba(139, 124, 246, 0.25)' },
      { s: 15, color: 'rgba(94, 234, 212, 0.25)' },
    ];

    refs.forEach(ref => {
      if (ref.s === S) return;
      ctx.beginPath();
      ctx.strokeStyle = ref.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let t = 0; t <= 30; t += 0.5) {
        const x = pad.left + (t / 30) * plotW;
        const y = pad.top + plotH * (1 - retention(t, ref.s));
        t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      const endY = pad.top + plotH * (1 - retention(30, ref.s));
      ctx.fillStyle = ref.color;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('S=' + ref.s, pad.left + plotW + 5, endY + 3);
    });

    // Active curve with glow
    ctx.beginPath();
    ctx.strokeStyle = '#8b7cf6';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(139, 124, 246, 0.3)';
    ctx.shadowBlur = 8;
    for (let t = 0; t <= 30; t += 0.2) {
      const x = pad.left + (t / 30) * plotW;
      const y = pad.top + plotH * (1 - retention(t, S));
      t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Fill under curve
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, 'rgba(139, 124, 246, 0.12)');
    grad.addColorStop(1, 'rgba(139, 124, 246, 0.01)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Label
    const endY = pad.top + plotH * (1 - retention(30, S));
    ctx.fillStyle = '#8b7cf6';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('S=' + S, pad.left + plotW + 5, endY + 3);

    // Key points
    [1, 7, 30].forEach(d => {
      const r = retention(d, S);
      const x = pad.left + (d / 30) * plotW;
      const y = pad.top + plotH * (1 - r);

      // Outer ring
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(139, 124, 246, 0.15)';
      ctx.fill();

      // Inner dot
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#8b7cf6';
      ctx.fill();

      ctx.fillStyle = '#1a1a2e';
      ctx.font = 'bold 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(Math.round(r * 100) + '%', x, y - 12);
    });
  }

  function animateCurve() {
    const diff = targetS - animatedS;
    if (Math.abs(diff) < 0.05) {
      animatedS = targetS;
      draw();
      curveRaf = null;
      return;
    }
    animatedS += diff * 0.15;
    draw();
    curveRaf = requestAnimationFrame(animateCurve);
  }

  slider.addEventListener('input', () => {
    valueDisplay.textContent = slider.value;
    targetS = parseInt(slider.value);
    if (!curveRaf) {
      curveRaf = requestAnimationFrame(animateCurve);
    }
  });

  animatedS = 5;
  targetS = 5;
  resize();
  draw();

  window.addEventListener('resize', () => {
    resize();
    draw();
  });
}

// ── Scroll Reveal ──
function initScrollReveal() {
  const sections = document.querySelectorAll('.section');
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.08, rootMargin: '0px 0px -50px 0px' }
  );

  sections.forEach(s => observer.observe(s));
}

// ── Nav Highlight ──
function initNavHighlight() {
  const links = document.querySelectorAll('.nav-links a[href^="#"]');
  const sections = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          links.forEach(l => {
            l.classList.toggle('active', l.getAttribute('href') === '#' + id);
          });
        }
      });
    },
    { threshold: 0.3 }
  );

  sections.forEach(s => observer.observe(s));
}

// ── Benchmark Bars ──
function initBenchmarkBars() {
  const bars = document.querySelectorAll('.bench-bar');
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const bar = entry.target;
          const targetWidth = bar.dataset.width;
          requestAnimationFrame(() => {
            bar.style.width = targetWidth + '%';
          });
          observer.unobserve(bar);
        }
      });
    },
    { threshold: 0.3 }
  );

  bars.forEach(b => observer.observe(b));
}

// ── Nav Scroll State ──
function initNavScroll() {
  const nav = document.querySelector('.nav');
  let ticking = false;

  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        nav.classList.toggle('scrolled', window.scrollY > 40);
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  initHeroCanvas();
  initPipeline();
  initForgettingCurve();
  initScrollReveal();
  initNavHighlight();
  initBenchmarkBars();
  initNavScroll();
});
