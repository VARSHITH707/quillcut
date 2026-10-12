/* Quillcut Studio frontend. Plain JavaScript, no build step. Talks to the local server only. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STORE_KEY = "quillcut.options.v1";

  const state = {
    upload: null,          // {upload_id, name, size, duration, width, height}
    presets: [],
    formats: [],
    xhr: null,
    pollTimer: null,
    activeJob: null,
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
  const save = (o) => { try { localStorage.setItem(STORE_KEY, JSON.stringify(o)); } catch { /* private mode */ } };
  const load = () => { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { return {}; } };

  /* ---------- options + live preview ---------- */
  const hex = (id) => $(id).value.replace("#", "").toUpperCase();
  const selected = (name) => (document.querySelector(`input[name="${name}"]:checked`) || {}).value;

  function currentOptions() {
    return {
      preset: selected("preset"),
      colors: { base: hex("c-base"), active: hex("c-active"), stroke: hex("c-stroke") },
      caption_size: Number($("size").value),
      max_clips: Number(selected("count") || 3),
      quality: selected("quality") || "draft",
      rights: $("rights").value,
      rights_confirmed: $("rights-ok").checked,
    };
  }

  function paintPreview() {
    const o = currentOptions();
    const phone = $("phone");
    phone.style.setProperty("--c-base", "#" + o.colors.base);
    phone.style.setProperty("--c-active", "#" + o.colors.active);
    phone.style.setProperty("--c-stroke", "#" + o.colors.stroke);
    phone.style.setProperty("--size", o.caption_size);
    phone.style.setProperty("--hook", Math.round(o.caption_size * 1.08));
    const preset = state.presets.find((p) => p.key === o.preset);
    const lower = Boolean(preset && preset.case === "lower");
    $("pv-hook").textContent = lower ? "the play nobody saw coming" : "The play nobody saw coming";
    $("pv-speech").replaceChildren(
      lower ? "this is how " : "This is how ",
      el("span", { className: "on", textContent: "it ends" }));
    $("size-out").textContent = o.caption_size;
    const { rights, rights_confirmed, ...rest } = o;
    save({ ...rest, rights });
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

  function renderChoices() {
    const formats = $("formats");
    formats.replaceChildren(...state.formats.map((f) => el("label", { className: "choice" },
      el("input", { type: "radio", name: "format", value: f.key, checked: f.available, disabled: !f.available }),
      el("span", {}, el("b", { textContent: f.name }),
        el("small", { textContent: f.available ? f.description : (f.note || f.description) })))));

    const presets = $("presets");
    const saved = load();
    const first = state.presets.some((p) => p.key === saved.preset) ? saved.preset : (state.presets[0] || {}).key;
    presets.replaceChildren(...state.presets.map((p) => el("label", { className: "choice" },
      el("input", { type: "radio", name: "preset", value: p.key, checked: p.key === first }),
      el("span", {}, el("b", { textContent: p.name }),
        el("small", { textContent: p.case === "lower" ? "lowercase, bold" : "natural case" })))));
    presets.addEventListener("change", (e) => { if (e.target.name === "preset") applyPreset(e.target.value); });

    const count = $("count");
    count.replaceChildren(...[1, 2, 3, 4, 5, 6, 7, 8].map((n) => el("label", {},
      el("input", { type: "radio", name: "count", value: String(n), checked: n === (saved.max_clips || 3) }),
      el("span", { textContent: String(n) }))));

    applyPreset(first);
    if (saved.colors) {
      $("c-base").value = "#" + saved.colors.base;
      $("c-active").value = "#" + saved.colors.active;
      $("c-stroke").value = "#" + saved.colors.stroke;
    }
    if (saved.caption_size) $("size").value = saved.caption_size;
    if (saved.quality) {
      const q = document.querySelector(`input[name="quality"][value="${saved.quality}"]`);
      if (q) q.checked = true;
    }
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
    $("file-chip").hidden = true;
    $("drop").hidden = false;
    $("file").value = "";
    showUploadError("");
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
    };
    xhr.onerror = () => { state.xhr = null; resetUpload(); showUploadError("The upload failed. Is the app still running?"); };
    xhr.onabort = () => { state.xhr = null; resetUpload(); };
    xhr.send(file);
  }

  /* ---------- job ---------- */
  function refreshGo() {
    const ready = Boolean(state.upload) && $("rights-ok").checked && !state.activeJob;
    $("go").disabled = !ready;
    $("go-hint").textContent = state.activeJob ? "Your reels are being made."
      : !state.upload ? "Upload a video and tick the box to continue."
      : !$("rights-ok").checked ? "Tick the box to confirm you can use this video." : "Ready when you are.";
  }

  const STAGES = {
    queued: ["Waiting in line…", 0.03],
    starting: ["Getting started…", 0.05],
    waiting_for_gpu: ["Waiting for the graphics card to be free…", 0.08],
    analysing: ["Listening to and watching your video…", 0.15],
    rendering: ["Cutting and captioning reel", 0.3],
    writing_captions: ["Writing captions and hashtags…", 0.92],
    finished: ["Done", 1],
  };
  function paintProgress(job) {
    const [label, base] = STAGES[job.stage] || [job.stage, 0.2];
    let p = base;
    let text = label;
    if (job.stage === "rendering") {
      p = 0.3 + 0.6 * (job.done / Math.max(job.total, 1));
      text = `${label} ${Math.min(job.done + 1, job.total)} of ${job.total}…`;
    }
    $("progress-stage").textContent = text;
    $("job-fill").style.setProperty("--p", p);
  }

  async function startJob() {
    $("job-error").hidden = true;
    const body = { ...currentOptions(), upload_id: state.upload.upload_id };
    $("go").disabled = true;
    try {
      const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      state.activeJob = job.id;
      $("progress").hidden = false;
      $("progress-title").textContent = "Making your reels…";
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

  function reelCard(clip, index) {
    if (!clip.file) {
      return el("article", { className: "reel" },
        el("div", { className: "reel__head" }, el("h3", { textContent: `Reel ${index + 1}` }),
          el("span", { className: "badge badge--bad", textContent: "Didn't pass checks" })),
        el("p", { className: "hint", textContent: (clip.failures || []).join(" · ") || "This reel failed a quality check and was not kept." }));
    }
    const caption = el("textarea", { id: `cap-${clip.clip_id}`, value: clip.caption || "", rows: 3 });
    const tags = (clip.hashtags || []);
    const source = clip.source === "ollama" ? "Written by your local AI model" : "Template caption · start Ollama for AI-written captions";
    return el("article", { className: "reel" },
      el("video", { src: clip.url + "#t=1", controls: true, preload: "metadata", playsInline: true, "aria-label": `Reel ${index + 1} preview` }),
      el("div", { className: "reel__head" },
        el("h3", { textContent: `Reel ${index + 1} · ${fmtTime(clip.duration)}` }),
        el("span", { className: "badge", textContent: "Checks passed" })),
      el("label", { htmlFor: caption.id, textContent: "Caption" }),
      caption,
      el("div", { className: "tags", "aria-label": "Hashtags" }, tags.map((t) => el("span", { className: "tag", textContent: t }))),
      el("p", { className: "src", textContent: source }),
      el("div", { className: "reel__row" },
        el("a", { className: "mini mini--solid", href: clip.download, textContent: "Download MP4" }),
        Object.assign(el("button", { type: "button", className: "mini", textContent: "Copy caption" }),
          { onclick: () => copy(caption.value, "Caption") }),
        Object.assign(el("button", { type: "button", className: "mini", textContent: "Copy hashtags" }),
          { onclick: () => copy(tags.join(" "), "Hashtags") })));
  }

  function showResults(job) {
    const grid = $("grid");
    const clips = job.clips || [];
    const good = clips.filter((c) => c.file);
    grid.replaceChildren(...clips.map(reelCard));
    $("results-empty").hidden = clips.length > 0;
    const note = $("results-note");
    note.hidden = !(job.note && good.length === 0);
    note.textContent = job.note || "";
    const zip = $("zip");
    zip.hidden = good.length === 0;
    zip.href = `/api/jobs/${job.id}/zip`;
    zip.textContent = `Download all ${good.length} (.zip)`;
  }

  /* ---------- history ---------- */
  async function loadHistory() {
    let jobs = [];
    try { jobs = (await api("/api/jobs")).jobs; } catch { return; }
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

    const o = status.ollama;
    const dot = $("status").querySelector(".dot");
    if (o.running && o.has_model) { dot.dataset.state = "ok"; $("status-text").textContent = "Ready · AI captions on"; }
    else if (o.running) { dot.dataset.state = "warn"; $("status-text").textContent = `Ready · install ${o.model} for AI captions`; }
    else { dot.dataset.state = "warn"; $("status-text").textContent = "Ready · template captions (Ollama is off)"; }

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
    // dropping anywhere else on the page must not navigate away from the app
    ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));
    $("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) startUpload(f); });
    $("file-cancel").addEventListener("click", () => state.xhr && state.xhr.abort());
    $("file-remove").addEventListener("click", resetUpload);
    ["size", "c-base", "c-active", "c-stroke"].forEach((id) => $(id).addEventListener("input", paintPreview));
    document.addEventListener("change", (e) => {
      if (["quality", "count"].includes(e.target.name)) paintPreview();
    });
    $("rights-ok").addEventListener("change", refreshGo);
    $("rights").addEventListener("change", paintPreview);
    $("go").addEventListener("click", startJob);

    refreshGo();
    loadHistory();
    // pick up a run that was still going when the page was reloaded
    const jobs = (await api("/api/jobs").catch(() => ({ jobs: [] }))).jobs;
    const live = jobs.find((j) => j.status === "queued" || j.status === "running");
    if (live) { state.activeJob = live.id; $("progress").hidden = false; paintProgress(live); poll(live.id); refreshGo(); }
    else { const last = jobs.find((j) => j.status === "done" && (j.clips || []).some((c) => c.file)); if (last) showResults(last); }
  }

  boot();
})();
