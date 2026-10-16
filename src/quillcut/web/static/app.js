/* Quillcut Studio frontend. Plain JavaScript, no build step. Talks to the local server only. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STORE_KEY = "quillcut.options.v3";
  const COUNTS = [1, 2, 3, 5, 10];
  const DEFAULT_TARGET = 45;

  const state = {
    upload: null,          // {upload_id, name, size, duration, width, height}
    logoId: null,
    analysis: null,        // result of the conversation analysis, or null while unknown
    analysisJob: null,
    presets: [],
    formats: [],
    reels: { presets: [30, 35, 45, 60, 75, 90], min: 15, max: 180, max_reels: 20 },
    targets: [DEFAULT_TARGET, DEFAULT_TARGET, DEFAULT_TARGET],
    xhr: null,
    pollTimer: null,
    analysisTimer: null,
    activeJob: null,
    lastJob: null,
  };

  /* ---------- helpers ---------- */
  const fmtSize = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : (b / 1e6).toFixed(1) + " MB");
  const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  const el = (tag, props = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key.startsWith("aria-")) node.setAttribute(key, value);
      else node[key] = value;
    }
    kids.flat().forEach((k) => node.append(k));
    return node;
  };
  let toastTimer;
  function toast(msg) {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
  }
  async function api(path, opts) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch {
      throw new Error("Can't reach the Quillcut app. Is it still running in your terminal?");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const save = (o) => { try { localStorage.setItem(STORE_KEY, JSON.stringify(o)); } catch { /* private mode */ } };
  const load = () => { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { return {}; } };
  const hex = (id) => $(id).value.replace("#", "").toUpperCase();
  const selected = (name) => (document.querySelector(`input[name="${name}"]:checked`) || {}).value;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* ---------- options ---------- */
  function currentOptions() {
    return {
      preset: selected("preset"),
      colors: { base: hex("c-base"), active: hex("c-active"), stroke: hex("c-stroke") },
      caption_size: Number($("size").value),
      caption_y: clamp(Number($("caption-y").value), 35, 65) / 100,
      accent: hex("c-accent"),
      intensity: selected("intensity") || "dynamic",
      aspect: selected("aspect") || "9:16",
      content_style: $("content-style").value,
      include_question: $("include-question").value,
      strict: $("strict").checked,
      music: selected("music") || "none",
      cta: $("cta").value.trim() || null,
      quality: selected("quality") || "draft",
      rights: $("rights").value,
      rights_confirmed: $("rights-ok").checked,
      reels: state.targets.map((target) => ({ target })),
      rules: $("rules").value.trim(),
    };
  }

  const INTENSITY_HINTS = {
    clean: "Framing change every 3-4 s with a gentle 1.05x zoom. Title and captions only.",
    dynamic: "Framing change every 2-2.5 s (1.0x, 1.10x, 1.14x), key-term emphasis and a key-phrase callout.",
    high_energy: "Framing change every 1.2-1.8 s (1.0x, 1.12x, 1.18x), two callouts and a progress bar.",
  };

  function paintPreview() {
    const o = currentOptions();
    const phone = $("phone");
    phone.style.setProperty("--c-base", "#" + o.colors.base);
    phone.style.setProperty("--c-active", "#" + o.colors.active);
    phone.style.setProperty("--c-stroke", "#" + o.colors.stroke);
    phone.style.setProperty("--c-accent", "#" + o.accent);
    phone.style.setProperty("--size", o.caption_size);
    phone.style.setProperty("--hook", Math.round(o.caption_size * 1.08));
    phone.style.setProperty("--sy", `${o.caption_y * 100}%`);
    phone.style.setProperty("--ar", { "9:16": "9 / 16", "1:1": "1 / 1", "16:9": "16 / 9" }[o.aspect]);
    phone.dataset.aspect = o.aspect;
    const preset = state.presets.find((p) => p.key === o.preset);
    const lower = Boolean(preset && preset.case === "lower");
    $("pv-hook").textContent = lower ? "the play nobody saw coming" : "The play nobody saw coming";
    $("pv-speech").replaceChildren(lower ? "this is how " : "This is how ", el("span", { className: "on", textContent: "it ends" }));
    $("size-out").textContent = o.caption_size;
    $("y-out").textContent = `${Math.round(o.caption_y * 100)}%`;
    $("intensity-hint").textContent = INTENSITY_HINTS[o.intensity];
    const { rights_confirmed, reels, ...rest } = o;
    save({ ...rest, targets: state.targets });
  }

  function applyPreset(key) {
    const p = state.presets.find((x) => x.key === key);
    if (!p) return;
    const c = p.colors || {};
    $("c-base").value = "#" + (c.base || "FFFFFF");
    $("c-active").value = "#" + (c.active || "FFD400");
    $("c-stroke").value = "#" + (c.stroke || "000000");
    $("size").value = p.speech_size;
    paintPreview();
  }

  /* ---------- reel count + per-reel durations ---------- */
  function renderCount() {
    const wrap = $("count");
    const n = state.targets.length;
    const custom = !COUNTS.includes(n);
    wrap.replaceChildren(
      ...COUNTS.map((c) => el("label", {},
        el("input", { type: "radio", name: "count", value: String(c), checked: c === n }),
        el("span", { textContent: String(c) }))),
      el("label", { className: "seg__custom" },
        el("input", { type: "radio", name: "count", value: "custom", checked: custom }),
        el("span", {}, "Custom ",
          Object.assign(el("input", { type: "number", id: "count-custom", min: 1, max: state.reels.max_reels,
            value: String(n), "aria-label": "Custom number of reels" }), {}))));
  }

  function setCount(n) {
    n = clamp(Math.round(n) || 1, 1, state.reels.max_reels);
    const t = state.targets.slice(0, n);
    while (t.length < n) t.push(t[t.length - 1] || DEFAULT_TARGET);
    state.targets = t;
    renderReels();
    paintPreview();
  }

  function renderReels() {
    const wrap = $("reels-config");
    wrap.replaceChildren(...state.targets.map((target, i) => {
      const chips = state.reels.presets.map((p) => el("label", {},
        el("input", { type: "radio", name: `dur-${i}`, value: String(p), checked: p === target }),
        el("span", { textContent: `${p}s` })));
      const input = el("input", { type: "number", min: state.reels.min, max: state.reels.max, value: String(target),
        "aria-label": `Reel ${i + 1} target duration in seconds` });
      input.addEventListener("change", () => {
        state.targets[i] = clamp(Number(input.value) || DEFAULT_TARGET, state.reels.min, state.reels.max);
        renderReels();
        paintPreview();
      });
      const group = el("div", { className: "seg seg--small", role: "radiogroup", "aria-label": `Reel ${i + 1} duration` }, chips);
      group.addEventListener("change", (e) => {
        state.targets[i] = Number(e.target.value);
        renderReels();
        paintPreview();
      });
      return el("div", { className: "reel-row" },
        el("span", { className: "reel-row__label", textContent: `Reel ${i + 1}` }), group,
        el("label", { className: "reel-row__custom" }, input, el("span", {}, "sec")));
    }));
    capacityHint();
  }

  function capacityHint() {
    const hint = $("capacity-hint");
    const a = state.analysis;
    if (!a) { hint.textContent = ""; hint.className = "hint"; return; }
    const n = state.targets.length;
    const cap = state.rulesCapacity != null ? state.rulesCapacity : a.capacity;
    const withRules = state.rulesCapacity != null ? " that follow your rules" : "";
    if (n > cap) {
      hint.className = "hint hint--warn";
      hint.textContent = `This video has ${cap} distinct complete moment${cap === 1 ? "" : "s"}${withRules}. ` +
        `You asked for ${n}: Quillcut will make ${a.capacity} and tell you why it stopped, instead of adding filler.`;
    } else {
      hint.className = "hint";
      hint.textContent = `This video has about ${a.capacity} distinct complete moments, so ${n} reel${n === 1 ? "" : "s"} is fine.`;
    }
  }

  /* ---------- choices ---------- */
  function renderChoices() {
    $("formats").replaceChildren(...state.formats.map((f) => el("label", { className: "choice" },
      el("input", { type: "radio", name: "format", value: f.key, checked: f.available, disabled: !f.available }),
      el("span", {}, el("b", { textContent: f.name }),
        el("small", { textContent: f.available ? f.description : (f.note || f.description) })))));

    const saved = load();
    const first = state.presets.some((p) => p.key === saved.preset) ? saved.preset : (state.presets[0] || {}).key;
    const presets = $("presets");
    presets.replaceChildren(...state.presets.map((p) => el("label", { className: "choice" },
      el("input", { type: "radio", name: "preset", value: p.key, checked: p.key === first }),
      el("span", {}, el("b", { textContent: p.name }),
        el("small", { textContent: p.case === "lower" ? "lowercase, bold" : "natural case" })))));
    presets.addEventListener("change", (e) => { if (e.target.name === "preset") applyPreset(e.target.value); });

    if (Array.isArray(saved.targets) && saved.targets.length) {
      state.targets = saved.targets.slice(0, state.reels.max_reels)
        .map((t) => clamp(Number(t) || DEFAULT_TARGET, state.reels.min, state.reels.max));
    }
    renderCount();
    renderReels();
    applyPreset(first);
    const set = (id, v) => { if (v !== undefined && v !== null) $(id).value = v; };
    if (saved.colors) {
      set("c-base", "#" + saved.colors.base); set("c-active", "#" + saved.colors.active); set("c-stroke", "#" + saved.colors.stroke);
    }
    if (saved.accent) set("c-accent", "#" + saved.accent);
    set("size", saved.caption_size);
    if (saved.caption_y) set("caption-y", clamp(Math.round(saved.caption_y * 100), 35, 65));
    set("content-style", saved.content_style);
    set("include-question", saved.include_question);
    set("rules", saved.rules);
    ["intensity", "aspect", "quality", "music"].forEach((name) => {
      const input = saved[name] && document.querySelector(`input[name="${name}"][value="${saved[name]}"]`);
      if (input) input.checked = true;
    });
    paintPreview();
  }

  /* ---------- upload ---------- */
  function showUploadError(msg) {
    const box = $("upload-error");
    box.textContent = msg || "";
    box.hidden = !msg;
  }
  function resetUpload() {
    state.upload = null;
    state.analysis = null;
    clearTimeout(state.analysisTimer);
    $("file-chip").hidden = true;
    $("analysis").hidden = true;
    $("drop").hidden = false;
    $("file").value = "";
    showUploadError("");
    capacityHint();
    refreshGo();
  }
  function startUpload(file) {
    showUploadError("");
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!state.extensions.includes(ext)) {
      return showUploadError(`That file type isn't supported. Use ${state.extensions.join(", ")}.`);
    }
    if (file.size > state.maxBytes) {
      return showUploadError(`That file is larger than ${state.maxGb} GB.`);
    }
    $("drop").hidden = true;
    $("file-chip").hidden = false;
    $("file-name").textContent = file.name;
    $("file-info").textContent = `${fmtSize(file.size)} · uploading`;
    $("file-warn").hidden = true;
    $("up-bar").hidden = false;
    $("up-fill").style.setProperty("--p", 0);
    $("file-cancel").hidden = false;
    $("file-remove").hidden = true;

    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("POST", "/api/upload?name=" + encodeURIComponent(file.name));
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) $("up-fill").style.setProperty("--p", e.loaded / e.total);
    };
    xhr.onload = () => {
      state.xhr = null;
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* handled below */ }
      if (xhr.status !== 201) {
        resetUpload();
        return showUploadError(data.error || "The upload failed. Please try again.");
      }
      state.upload = data;
      $("up-bar").hidden = true;
      $("file-cancel").hidden = true;
      $("file-remove").hidden = false;
      $("file-info").textContent = `${fmtSize(data.size)} · ${fmtTime(data.duration)} · ${data.width}×${data.height}`;
      if (data.warning) { $("file-warn").textContent = data.warning; $("file-warn").hidden = false; }
      refreshGo();
      startAnalysis();
    };
    xhr.onerror = () => { state.xhr = null; resetUpload(); showUploadError("The upload failed. Is the app still running?"); };
    xhr.onabort = () => { state.xhr = null; resetUpload(); };
    xhr.send(file);
  }

  async function uploadLogo(file) {
    const note = $("logo-state");
    if (!file) { state.logoId = null; note.textContent = ""; $("pv-logo").hidden = true; return; }
    note.textContent = "Uploading logo…";
    try {
      const res = await fetch("/api/upload?kind=logo&name=" + encodeURIComponent(file.name), { method: "POST", body: file });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Logo upload failed");
      state.logoId = data.upload_id;
      note.textContent = `${file.name} will appear on every reel.`;
      $("pv-logo").hidden = false;
    } catch (err) {
      state.logoId = null;
      note.textContent = err.message;
    }
  }

  /* ---------- rules (optional) ---------- */
  let rulesTimer;
  function renderRulesReport(res) {
    const box = $("rules-report");
    const rep = res.report || { applied: [], advisory: [] };
    const rows = [];
    rep.applied.forEach((a) => rows.push(el("div", { className: `rule rule--${a.kind}` },
      el("b", { textContent: a.kind === "filter" ? "Enforced" : "Setting" }), el("span", { textContent: a.meaning }))));
    rep.advisory.forEach((line) => rows.push(el("div", { className: "rule rule--advisory" },
      el("b", { textContent: "Not enforced" }),
      el("span", { textContent: `“${line}” can't be checked automatically. Quillcut will still ask the local model for an opinion on each reel.` }))));
    if (res.capacity && rep.filters) {
      state.rulesCapacity = res.capacity.capacity;
      rows.push(el("div", { className: "rule rule--count" },
        el("span", { textContent: `With these rules: ${res.capacity.capacity} complete moment${res.capacity.capacity === 1 ? "" : "s"} qualify.` })));
    } else {
      state.rulesCapacity = null;
    }
    box.replaceChildren(...rows);
    $("rules-clear").hidden = !$("rules").value.trim();
    capacityHint();
  }
  function previewRules() {
    clearTimeout(rulesTimer);
    rulesTimer = setTimeout(async () => {
      const text = $("rules").value;
      if (!text.trim()) { renderRulesReport({}); return; }
      try {
        renderRulesReport(await post("/api/rules/preview", { rules: text, analysis_job: state.analysisJob }));
      } catch (err) {
        $("rules-report").replaceChildren(el("div", { className: "rule rule--advisory", textContent: err.message }));
      }
    }, 500);
  }
  function loadRulesFile(file) {
    if (!file) return;
    if (file.size > 65536) return toast("That file is too large for rules (64 KB max).");
    const reader = new FileReader();
    reader.onload = () => {
      $("rules").value = String(reader.result).slice(0, 4000);
      if (String(reader.result).length > 4000) toast("Only the first 4000 characters were used.");
      paintPreview();
      previewRules();
    };
    reader.readAsText(file);
  }

  /* ---------- analysis ---------- */
  async function startAnalysis() {
    $("analysis").hidden = false;
    $("analysis-result").hidden = true;
    $("analysis-bar").hidden = false;
    $("analysis-state").textContent = "Reading the whole conversation…";
    $("analysis-fill").style.setProperty("--p", 0.1);
    try {
      const job = await post("/api/jobs", { kind: "analyze", upload_id: state.upload.upload_id });
      state.analysisJob = job.id;
      pollAnalysis(job.id);
    } catch (err) {
      $("analysis-state").textContent = err.message;
      $("analysis-bar").hidden = true;
    }
  }

  const ANALYSIS_STAGES = { queued: ["Waiting in line…", 0.1], starting: ["Getting started…", 0.15],
    waiting_for_gpu: ["Waiting for the graphics card…", 0.2], analysing: ["Listening to the whole video…", 0.5],
    transcribing: ["Transcribing every word…", 0.5], understanding: ["Understanding the conversation…", 0.85] };

  function pollAnalysis(id) {
    clearTimeout(state.analysisTimer);
    state.analysisTimer = setTimeout(async () => {
      try {
        const job = await api(`/api/jobs/${id}`);
        if (job.status === "queued" || job.status === "running") {
          const [label, p] = ANALYSIS_STAGES[job.stage] || ["Working…", 0.4];
          $("analysis-state").textContent = label;
          $("analysis-fill").style.setProperty("--p", p);
          return pollAnalysis(id);
        }
        if (job.status === "failed") {
          $("analysis-bar").hidden = true;
          $("analysis-state").textContent = job.error || "Couldn't analyse this video.";
          return;
        }
        showAnalysis(job.analysis);
      } catch (err) {
        $("analysis-state").textContent = err.message + " Retrying…";
        pollAnalysis(id);
      }
    }, 1500);
  }

  function showAnalysis(a) {
    state.analysis = a;
    $("analysis-bar").hidden = true;
    $("analysis-state").textContent = "Done reading.";
    $("analysis-result").hidden = false;
    const voices = a.speakers.length > 1 ? `${a.speakers.length} voices` : "one voice";
    $("analysis-headline").textContent =
      `${a.capacity} complete moment${a.capacity === 1 ? "" : "s"} found in ${fmtTime(a.duration)} · ` +
      `${a.questions} question${a.questions === 1 ? "" : "s"} · ${voices}`;
    $("moments").replaceChildren(...a.moments.slice(0, 6).map((m) => el("li", {},
      el("span", { className: "moment__time", textContent: `${fmtTime(m.start)}–${fmtTime(m.end)} · ${Math.round(m.duration)}s` }),
      el("span", { className: "moment__tag", textContent: m.kind === "qa" ? "Question + answer" : "Talk" }),
      el("q", { textContent: m.hook.length > 110 ? m.hook.slice(0, 107) + "…" : m.hook }))));
    if (a.capacity > 0 && !load().targets) setCount(Math.min(3, a.capacity));
    capacityHint();
    previewRules();
    refreshGo();
  }

  /* ---------- job ---------- */
  function refreshGo() {
    const ready = Boolean(state.upload) && $("rights-ok").checked && !state.activeJob;
    $("go").disabled = !ready;
    $("go-hint").textContent = state.activeJob ? "Your reels are being made."
      : !state.upload ? "Upload a video to start. Rules are optional, but a video is required."
      : !$("rights-ok").checked ? "Tick the box to confirm you can use this video." : "Ready when you are.";
  }

  const STAGES = [
    ["transcribing", "Transcribing"], ["understanding", "Understanding the conversation"],
    ["reviewing", "Checking each moment"], ["editing", "Planning the edit"], ["rendering", "Editing and rendering"],
    ["validating", "Quality checks"], ["writing_captions", "Writing captions"],
  ];
  function paintProgress(job) {
    const idx = Math.max(0, STAGES.findIndex(([k]) => k === job.stage));
    const done = job.stage === "finished";
    $("stages").replaceChildren(...STAGES.map(([k, label], i) =>
      el("li", { className: done || i < idx ? "is-done" : i === idx ? "is-now" : "", textContent: label })));
    let p = 0.03 + (idx / STAGES.length) * 0.9;
    let text = (STAGES[idx] || ["", "Starting"])[1] + "…";
    if (job.stage === "waiting_for_gpu") text = "Waiting for the graphics card to be free…";
    if (job.stage === "rendering" || job.stage === "editing" || job.stage === "validating") {
      const total = Math.max(job.total || 1, 1);
      const reel = Math.min((job.done || 0) + (job.stage === "validating" ? 0 : 1), total);
      text += ` reel ${reel} of ${total}`;
      p += (job.done / total) * (0.9 / STAGES.length) * 3;
    }
    $("progress-stage").textContent = text;
    $("job-fill").style.setProperty("--p", Math.min(p, 0.98));
  }

  async function startJob(overrides = {}) {
    $("job-error").hidden = true;
    const body = { ...currentOptions(), ...overrides, kind: "reels", upload_id: state.upload.upload_id };
    if (state.logoId) body.logo_upload = state.logoId;
    $("go").disabled = true;
    try {
      const job = await post("/api/jobs", body);
      state.activeJob = job.id;
      $("progress").hidden = false;
      paintProgress(job);
      $("progress").scrollIntoView({ behavior: "smooth", block: "center" });
      poll(job.id);
    } catch (err) {
      const box = $("job-error");
      box.textContent = err.message;
      box.hidden = false;
    }
    refreshGo();
  }

  function poll(id) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      try {
        const job = await api(`/api/jobs/${id}`);
        if (job.status === "queued" || job.status === "running") {
          paintProgress(job);
          return poll(id);
        }
        finishJob(job);
      } catch (err) {
        $("progress-stage").textContent = err.message + " Retrying…";
        poll(id);
      }
    }, 1500);
  }

  function finishJob(job) {
    state.activeJob = null;
    $("progress").hidden = true;
    refreshGo();
    if (job.status === "failed") {
      const box = $("job-error");
      box.textContent = job.error || "Something went wrong.";
      box.hidden = false;
      toast("That run didn't finish.");
    } else {
      showResults(job);
      toast("Your reels are ready.");
      $("results").scrollIntoView({ behavior: "smooth" });
    }
    loadHistory();
  }

  /* ---------- results ---------- */
  async function copy(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      toast(`${label} copied`);
    } catch {
      toast("Copy isn't allowed here. Select the text and press Ctrl+C.");
    }
  }

  const HOOK_LABEL = { cold_open: "Opens on the strongest line", question: "Opens on the question", opening: "Opens on the first line" };
  const ARCHETYPE = { loss_avoidance: "Loss avoidance", information_asymmetry: "Insider information", status_optimization: "Status and optimisation", contrarian: "Contrarian" };

  function checksList(items) {
    return el("ul", { className: "checks" }, items.map((v) =>
      el("li", { className: v.status === "PASS" ? "ok" : v.status === "WARN" ? "warn" : "bad",
        textContent: `${v.status === "PASS" ? "✓" : v.status === "WARN" ? "!" : "✕"} ${v.name.replace(/_/g, " ")}: ${v.message}` })));
  }

  function reelCard(clip, index, jobId) {
    if (!clip.file) {
      const why = clip.rejected || (clip.failures || []).join(" · ") || "This reel failed a quality check and was not kept.";
      return el("article", { className: "reel" },
        el("div", { className: "reel__head" }, el("h3", { textContent: `Reel ${clip.reel || index + 1}` }),
          el("span", { className: "badge badge--bad", textContent: clip.rejected ? "Not made" : "Didn't pass checks" })),
        el("p", { className: "hint", textContent: why }));
    }
    const caption = el("textarea", { id: `cap-${clip.clip_id}`, value: clip.caption || "", rows: 8 });
    const tags = clip.hashtags || [];
    const source = clip.source === "ollama" ? "Written by your local AI model" : "Template caption · start Ollama for AI-written captions";
    const review = clip.needs_review;
    const badges = [
      el("span", { className: "badge" + (review ? " badge--warn" : ""), textContent: review ? "Needs a look" : "Checks passed" }),
    ];
    if (clip.extended) badges.push(el("span", { className: "badge badge--ext", textContent: "Extended to finish the thought" }));
    if ((clip.rules_check || []).length && clip.rules_check.every((r) => r.ok)) {
      badges.push(el("span", { className: "badge", textContent: "Follows your rules" }));
    }
    const durations = el("p", { className: "reel__dur" },
      el("b", { textContent: `${fmtTime(clip.actual || clip.duration)}` }),
      ` actual · ${Math.round(clip.target || 0)}s target · source ${fmtTime(clip.source_start || 0)}–${fmtTime(clip.source_end || 0)}`);
    const regen = el("select", { "aria-label": "New duration for this reel" },
      state.reels.presets.map((p) => el("option", { value: String(p), textContent: `${p}s`, selected: p === Math.round(clip.target) })));
    return el("article", { className: "reel" },
      el("video", { src: clip.url + "#t=1", controls: true, preload: "metadata", playsInline: true, "aria-label": `Reel ${index + 1} preview` }),
      el("div", { className: "reel__head" },
        el("h3", { textContent: clip.topic ? `${index + 1}. ${clip.topic}` : `Reel ${index + 1}` }), badges),
      durations,
      clip.note ? el("p", { className: "reel__note", textContent: clip.note }) : "",
      el("p", { className: "reel__hook" },
        el("small", { textContent: (HOOK_LABEL[(clip.hook || {}).kind] || "Hook") + ((clip.hook || {}).archetype ? " · " + ARCHETYPE[clip.hook.archetype] : "") }), " ",
        el("q", { textContent: (clip.hook || {}).text || "" })),
      el("label", { htmlFor: caption.id, textContent: "Caption" }),
      caption,
      el("div", { className: "tags", "aria-label": "Hashtags" }, tags.map((t) => el("span", { className: "tag", textContent: t }))),
      el("p", { className: "src", textContent: source }),
      el("div", { className: "reel__row" },
        el("a", { className: "mini mini--solid", href: clip.download, textContent: "Download MP4" }),
        Object.assign(el("button", { type: "button", className: "mini", textContent: "Copy caption" }),
          { onclick: () => copy(caption.value, "Caption") }),
        Object.assign(el("button", { type: "button", className: "mini", textContent: "Copy hashtags" }),
          { onclick: () => copy(tags.join(" "), "Hashtags") })),
      el("details", { className: "reel__more" },
        el("summary", { textContent: "How this reel was made" }),
        (clip.rules_check || []).length ? el("div", {}, el("p", { className: "hint", textContent: "Your rules" }),
          checksList(clip.rules_check.map((r) => ({ status: r.ok ? "PASS" : "FAIL", name: r.rule, message: r.detail })))) : "",
        (clip.rules_opinion || []).length ? el("div", {},
          el("p", { className: "hint", textContent: "Local model's opinion on the rules it cannot check (not guaranteed)" }),
          el("ul", { className: "checks" }, clip.rules_opinion.map((o) => el("li", {
            className: o.verdict === "yes" ? "ok" : o.verdict === "no" ? "warn" : "na",
            textContent: `${o.verdict.toUpperCase()}: ${o.rule}${o.reason ? " — " + o.reason : ""}` })))) : "",
        checksList(clip.validation || []),
        el("p", { className: "hint", textContent: `Edit: ${clip.intensity || "dynamic"} · ${clip.aspect || "9:16"}` +
          `${clip.question_included ? " · question included" : ""}` }),
        (clip.reasons || []).length ? el("ul", { className: "reasons" }, clip.reasons.map((r) => el("li", { textContent: r }))) : ""),
      el("div", { className: "reel__row reel__row--regen" },
        regen,
        Object.assign(el("button", { type: "button", className: "mini", textContent: "New length" }), {
          onclick: () => regenerate({ reels: [{ target: Number(regen.value) }],
            prefer: [clip.source_start, clip.source_end] }) }),
        Object.assign(el("button", { type: "button", className: "mini", textContent: "Try another moment" }), {
          onclick: () => regenerate({ reels: [{ target: Number(regen.value) }],
            exclude: [[clip.source_start, clip.source_end]] }) })));
  }

  function regenerate(overrides) {
    if (!state.upload) {
      return toast("Upload the video again to regenerate a reel.");
    }
    startJob(overrides);
  }

  function showResults(job) {
    state.lastJob = job;
    const grid = $("grid");
    const clips = job.clips || [];
    const good = clips.filter((c) => c.file);
    grid.replaceChildren(...clips.map((c, i) => reelCard(c, i, job.id)));
    $("results-empty").hidden = clips.length > 0;
    const note = $("results-note");
    note.hidden = !job.note;
    note.textContent = job.note || "";
    $("results-messages").replaceChildren(...(job.messages || []).map((m) => el("li", { textContent: m })));
    const zip = $("zip");
    zip.hidden = good.length === 0;
    zip.href = `/api/jobs/${job.id}/zip`;
    zip.textContent = `Download all ${good.length} (.zip)`;
  }

  /* ---------- history ---------- */
  async function loadHistory() {
    let jobs = [];
    try { jobs = (await api("/api/jobs")).jobs.filter((j) => j.kind !== "analyze"); } catch { return; }
    $("history-empty").hidden = jobs.length > 0;
    $("runs").replaceChildren(...jobs.map((j) => {
      const count = (j.clips || []).filter((c) => c.file).length;
      const when = new Date(j.created * 1000).toLocaleString();
      const status = j.status === "done" ? `${count} reel${count === 1 ? "" : "s"}` : j.status;
      const open = Object.assign(el("button", { type: "button", className: "mini", textContent: "Open" }),
        { onclick: () => { showResults(j); $("results").scrollIntoView({ behavior: "smooth" }); } });
      const del = Object.assign(el("button", { type: "button", className: "mini", textContent: "Delete" }), {
        onclick: async () => {
          if (!confirm("Delete this run and its reels?")) return;
          try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); loadHistory(); } catch (e) { toast(e.message); }
        },
      });
      return el("li", {},
        el("div", {}, el("div", { className: "runs__name", textContent: j.filename || j.id }),
          el("div", { className: "runs__meta", textContent: `${when} · ${status}` })),
        el("div", { className: "reel__row" }, j.status === "done" && count ? open : "", del));
    }));
  }

  /* ---------- boot ---------- */
  async function boot() {
    let status;
    try {
      status = await api("/api/status");
    } catch (err) {
      $("status-text").textContent = "App not reachable";
      $("status").querySelector(".dot").dataset.state = "bad";
      return showUploadError(err.message);
    }
    state.presets = status.presets;
    state.formats = status.formats;
    state.extensions = status.extensions;
    state.maxGb = status.max_upload_gb;
    state.maxBytes = status.max_upload_gb * 1024 ** 3;
    if (status.reels) state.reels = { ...state.reels, ...status.reels };

    const o = status.ollama;
    const dot = $("status").querySelector(".dot");
    if (o.running && o.has_model) { dot.dataset.state = "ok"; $("status-text").textContent = "Ready · AI captions on"; }
    else if (o.running) { dot.dataset.state = "warn"; $("status-text").textContent = `Ready · install ${o.model} for AI captions`; }
    else { dot.dataset.state = "warn"; $("status-text").textContent = "Ready · template captions (Ollama is off)"; }
    $("music-hint").textContent = status.reels && status.reels.music_available ? ""
      : "No music track is installed, so reels render without music. Run tools/make_music.py to generate royalty-free beds.";

    $("rights").replaceChildren(...Object.entries(status.rights).map(([k, v]) => el("option", { value: k, textContent: v })));
    const saved = load();
    if (saved.rights && status.rights[saved.rights]) $("rights").value = saved.rights;
    renderChoices();

    // events
    $("pick").addEventListener("click", (e) => { e.stopPropagation(); $("file").click(); });
    const drop = $("drop");
    drop.addEventListener("click", () => $("file").click());
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) startUpload(f); });
    ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));
    $("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) startUpload(f); });
    $("file-cancel").addEventListener("click", () => state.xhr && state.xhr.abort());
    $("file-remove").addEventListener("click", resetUpload);
    $("logo").addEventListener("change", (e) => uploadLogo(e.target.files[0]));
    ["size", "caption-y", "c-base", "c-active", "c-stroke", "c-accent", "content-style", "include-question", "cta"]
      .forEach((id) => $(id).addEventListener("input", paintPreview));
    $("strict").addEventListener("change", paintPreview);
    document.addEventListener("change", (e) => {
      const n = e.target.name;
      if (["quality", "intensity", "aspect", "music"].includes(n)) paintPreview();
      if (n === "count" && e.target.value !== "custom") setCount(Number(e.target.value));
      if (n === "count" && e.target.value === "custom") {
        const box = $("count-custom");
        setCount(Number(box.value) || state.targets.length);
        box.focus();
      }
    });
    document.addEventListener("input", (e) => {
      if (e.target.id === "count-custom") {
        const radio = document.querySelector('input[name="count"][value="custom"]');
        if (radio) radio.checked = true;
        setCount(Number(e.target.value));
      }
    });
    $("rights-ok").addEventListener("change", refreshGo);
    $("rights").addEventListener("change", paintPreview);
    $("rules").addEventListener("input", () => { paintPreview(); previewRules(); });
    $("rules-file").addEventListener("change", (e) => loadRulesFile(e.target.files[0]));
    $("rules-clear").addEventListener("click", () => { $("rules").value = ""; paintPreview(); previewRules(); });
    previewRules();
    $("go").addEventListener("click", () => startJob());

    refreshGo();
    loadHistory();
    // pick up a run that was still going when the page was reloaded
    const jobs = (await api("/api/jobs").catch(() => ({ jobs: [] }))).jobs.filter((j) => j.kind !== "analyze");
    const live = jobs.find((j) => j.status === "queued" || j.status === "running");
    if (live) { state.activeJob = live.id; $("progress").hidden = false; paintProgress(live); poll(live.id); refreshGo(); }
    else { const last = jobs.find((j) => j.status === "done" && (j.clips || []).some((c) => c.file)); if (last) showResults(last); }
  }

  boot();
})();

