/* Pizarra IA — interfaz web. Todas las rutas son relativas (funciona bajo /pizarra/). */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const KEY_STORE = "pizarra.clave";
  const JOB_STORE = "pizarra.trabajo";
  const CHARS_PER_SEC = 13.5;
  let pollTimer = null;
  let currentJob = null;

  // ---------- almacenamiento local seguro ----------
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* sin almacenamiento */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* nada */ } },
  };

  async function api(path, body) {
    const opt = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    };
    const r = await fetch(path, opt);
    let data = null;
    try { data = await r.json(); } catch (e) { /* respuesta vacía */ }
    if (!r.ok) {
      let msg = (data && data.detail) || `Error ${r.status}`;
      if (Array.isArray(msg)) msg = "Revisa los datos del formulario.";
      throw new Error(msg);
    }
    return data;
  }

  function setStatus(el, text, kind) {
    el.textContent = text || "";
    el.className = "status" + (kind ? " " + kind : "");
  }

  function radio(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : null;
  }

  // ---------- configuración inicial ----------
  async function init() {
    const saved = store.get(KEY_STORE);
    if (saved) { $("#clave").value = saved; $("#recordar").checked = true; }
    try {
      const cfg = await api("api/config");
      const idi = $("#idioma");
      Object.entries(cfg.idiomas).forEach(([k, v]) => idi.add(new Option(v, k, k === "es", k === "es")));
      const voz = $("#voz");
      const def = cfg.voz_defecto || "Charon";
      const info = cfg.voces_info || {};
      cfg.voces.forEach((v) => voz.add(new Option(info[v] ? `${v} — ${info[v]}` : v, v, v === def, v === def)));
      $("#cupo").textContent = `Te quedan ${cfg.renders_restantes} de ${cfg.renders_por_dia} vídeos hoy en este servidor · máximo ${cfg.max_segundos} s por vídeo.`;
    } catch (e) { /* la página sigue funcionando */ }
    const last = store.get(JOB_STORE);
    if (last) resumeJob(last);
  }

  function rememberKey() {
    const v = $("#clave").value.trim();
    if ($("#recordar").checked && v) store.set(KEY_STORE, v); else store.del(KEY_STORE);
  }
  $("#recordar").addEventListener("change", rememberKey);
  $("#clave").addEventListener("change", rememberKey);

  function fillSelect(sel, items, def) {
    sel.length = 1;
    items.forEach((m) => sel.add(new Option(m + (m === def ? " (recomendado)" : ""), m)));
  }

  $("#btn-clave").addEventListener("click", async () => {
    const st = $("#clave-estado");
    const clave = $("#clave").value.trim();
    if (!clave) { setStatus(st, "Pega tu clave primero.", "err"); return; }
    rememberKey();
    setStatus(st, "Comprobando…");
    try {
      const r = await api("api/modelos", { clave });
      fillSelect($("#m-texto"), r.texto, r.defecto.texto);
      fillSelect($("#m-imagen"), r.imagen, r.defecto.imagen);
      fillSelect($("#m-tts"), r.tts, r.defecto.tts);
      const partes = [`texto: ${r.defecto.texto || "ninguno"}`, `imagen: ${r.defecto.imagen || "ninguno (dibujo vectorial)"}`, `voz: ${r.defecto.tts || "ninguna (voz local)"}`];
      setStatus(st, "Clave válida · " + partes.join(" · "), "ok");
    } catch (e) { setStatus(st, e.message, "err"); }
  });

  // ---------- voz ----------
  let audioUrl = null;
  function syncVoice() {
    const local = radio("tts") === "piper";
    $("#voz").disabled = local;
    $("#btn-probar").textContent = local ? "Probar voz local" : "Probar voz";
  }
  document.querySelectorAll('input[name="tts"]').forEach((el) => el.addEventListener("change", syncVoice));
  $("#voz").addEventListener("change", () => setStatus($("#voz-estado"), ""));

  $("#btn-probar").addEventListener("click", async () => {
    const st = $("#voz-estado");
    const motor = radio("tts") === "piper" ? "piper" : "gemini";
    const clave = $("#clave").value.trim();
    if (motor === "gemini" && !clave) {
      setStatus(st, "Para oír las voces de Gemini pega tu clave (paso 1). La voz local se puede probar sin clave.", "err");
      return;
    }
    const btn = $("#btn-probar");
    btn.disabled = true;
    setStatus(st, motor === "piper" ? "Generando la voz local…" : `Generando la voz ${$("#voz").value}…`);
    try {
      const r = await fetch("api/probar-voz", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clave: motor === "gemini" ? clave : null, voz: $("#voz").value, motor,
          idioma: $("#idioma").value, modelo: $("#m-tts").value || null }),
      });
      if (!r.ok) {
        let msg = `Error ${r.status}`;
        try { const d = await r.json(); if (typeof d.detail === "string") msg = d.detail; } catch (e) { /* nada */ }
        throw new Error(msg);
      }
      const blob = await r.blob();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      audioUrl = URL.createObjectURL(blob);
      const a = $("#audio-voz");
      a.src = audioUrl;
      a.classList.remove("hidden");
      setStatus(st, "");
      try { await a.play(); } catch (e) { /* el navegador bloquea la reproducción automática */ }
    } catch (e) { setStatus(st, e.message, "err"); }
    btn.disabled = false;
  });

  // ---------- guion ----------
  $("#btn-guion").addEventListener("click", async () => {
    const st = $("#guion-estado");
    const clave = $("#clave").value.trim();
    const tema = $("#tema").value.trim();
    if (!clave) { setStatus(st, "Falta tu clave de Gemini (paso 1).", "err"); $("#clave").focus(); return; }
    if (tema.length < 3) { setStatus(st, "Escribe un tema.", "err"); $("#tema").focus(); return; }
    rememberKey();
    const btn = $("#btn-guion");
    btn.disabled = true;
    setStatus(st, $("#investigar").checked ? "Investigando y escribiendo… (hasta 1 min)" : "Escribiendo el guion… (unos segundos)");
    try {
      const plan = await api("api/guion", {
        clave, tema, idioma: $("#idioma").value, duracion: Number($("#duracion").value),
        investigar: $("#investigar").checked, modelo_texto: $("#m-texto").value || null,
      });
      showPlan(plan);
      setStatus(st, "Guion listo. Revísalo abajo.", "ok");
      $("#paso-guion").scrollIntoView({ behavior: "smooth" });
    } catch (e) { setStatus(st, e.message, "err"); }
    btn.disabled = false;
  });

  function sceneEl(sc) {
    const li = $("#tpl-escena").content.firstElementChild.cloneNode(true);
    li.querySelectorAll("[data-k]").forEach((el) => { el.value = sc[el.dataset.k] || ""; });
    li.addEventListener("input", updateEstimate);
    li.querySelector(".scene-tools").addEventListener("click", (ev) => {
      const act = ev.target.dataset.act;
      if (!act) return;
      if (act === "del") {
        if ($("#escenas").children.length > 1) li.remove();
      } else if (act === "up" && li.previousElementSibling) {
        li.parentNode.insertBefore(li, li.previousElementSibling);
      } else if (act === "down" && li.nextElementSibling) {
        li.parentNode.insertBefore(li.nextElementSibling, li);
      }
      renumber();
    });
    return li;
  }

  function renumber() {
    [...$("#escenas").children].forEach((li, i) => { li.querySelector(".scene-n").textContent = `Escena ${i + 1}`; });
    updateEstimate();
  }

  function showPlan(plan) {
    $("#titulo").value = plan.titulo || "";
    const ol = $("#escenas");
    ol.innerHTML = "";
    (plan.escenas || []).forEach((sc) => ol.appendChild(sceneEl(sc)));
    $("#paso-guion").classList.remove("hidden");
    renumber();
  }

  function collectPlan() {
    return {
      titulo: $("#titulo").value.trim(),
      idioma: $("#idioma").value,
      escenas: [...$("#escenas").children].map((li) => {
        const sc = {};
        li.querySelectorAll("[data-k]").forEach((el) => { sc[el.dataset.k] = el.value.trim(); });
        return sc;
      }).filter((s) => s.narracion),
    };
  }

  function updateEstimate() {
    const p = collectPlan();
    const chars = p.escenas.reduce((a, s) => a + s.narracion.length, 0);
    const sec = Math.round(chars / CHARS_PER_SEC + p.escenas.length * 0.9);
    $("#estimacion").textContent = `${p.escenas.length} escenas · ≈ ${sec} s de vídeo`;
  }

  $("#btn-add").addEventListener("click", () => {
    $("#escenas").appendChild(sceneEl({ narracion: "", visual: "", etiqueta: "" }));
    renumber();
  });

  function options() {
    return {
      formato: radio("formato"), estilo: radio("estilo"), idioma: $("#idioma").value, voz: $("#voz").value,
      tts: radio("tts"), imagenes: $("#imagenes").value, color: $("#color").checked,
      subtitulos: $("#subtitulos").checked,
      modelo_texto: $("#m-texto").value || null, modelo_imagen: $("#m-imagen").value || null,
      modelo_tts: $("#m-tts").value || null,
    };
  }

  // ---------- render ----------
  async function startRender(body, st) {
    try {
      const r = await api("api/render", body);
      store.set(JOB_STORE, r.id);
      watchJob(r.id);
      setStatus(st, "");
    } catch (e) { setStatus(st, e.message, "err"); }
  }

  $("#btn-render").addEventListener("click", () => {
    const plan = collectPlan();
    const st = $("#render-estado");
    if (!plan.escenas.length) { setStatus(st, "El guion no tiene ninguna escena con narración.", "err"); return; }
    startRender({ clave: $("#clave").value.trim(), plan, opciones: options() }, st);
  });

  // «Mira un vídeo de ejemplo»: si la página tiene un vídeo de ejemplo (#ejemplo), baja hasta él y lo reproduce;
  // si no, el enlace lleva a los ejemplos del README en GitHub.
  $("#ver-ejemplo").addEventListener("click", (e) => {
    const ej = document.getElementById("ejemplo");
    if (!ej) return;
    e.preventDefault();
    ej.scrollIntoView({ behavior: "smooth", block: "center" });
    const v = ej.querySelector("video");
    if (v) { v.play().catch(() => {}); }
  });

  function resumeJob(id) {
    api(`api/trabajos/${id}`).then((j) => {
      if (j.estado === "cancelado") { store.del(JOB_STORE); return; }
      watchJob(id);
    }).catch(() => store.del(JOB_STORE));
  }

  function watchJob(id) {
    currentJob = id;
    $("#paso-video").classList.remove("hidden");
    $("#resultado").classList.add("hidden");
    $("#btn-cancel").classList.remove("hidden");
    $("#paso-video").scrollIntoView({ behavior: "smooth" });
    clearTimeout(pollTimer);
    poll();
  }

  async function poll() {
    if (!currentJob) return;
    let j;
    try { j = await api(`api/trabajos/${currentJob}`); } catch (e) {
      $("#job-msg").textContent = e.message;
      store.del(JOB_STORE);
      return;
    }
    const msg = $("#job-msg");
    const sub = $("#job-sub");
    $("#bar").style.width = Math.round((j.progreso || 0) * 100) + "%";
    if (j.estado === "en_cola") {
      msg.textContent = j.posicion > 1 ? `En cola: hay ${j.posicion - 1} vídeo(s) antes que el tuyo.` : "En cola: el siguiente eres tú.";
      sub.textContent = "Se crea un vídeo a la vez. Puedes dejar esta página abierta.";
    } else if (j.estado === "procesando") {
      msg.textContent = j.mensaje || "Creando el vídeo…";
      sub.textContent = `${Math.round((j.progreso || 0) * 100)} %`;
    } else if (j.estado === "listo") {
      msg.textContent = `Vídeo listo · ${Math.round(j.duracion || 0)} s` + (j.tiempo_render ? ` · creado en ${Math.round(j.tiempo_render)} s` : "");
      sub.textContent = j.motores ? Object.entries(j.motores).map(([k, v]) => `${k}: ${v}`).join(" · ") : "";
      showResult(j);
      return;
    } else if (j.estado === "error") {
      msg.textContent = "No se pudo crear el vídeo: " + (j.error || "error desconocido");
      sub.textContent = "";
      $("#btn-cancel").classList.add("hidden");
      store.del(JOB_STORE);
      return;
    } else if (j.estado === "cancelado") {
      msg.textContent = "Cancelado.";
      sub.textContent = "";
      $("#btn-cancel").classList.add("hidden");
      store.del(JOB_STORE);
      return;
    }
    pollTimer = setTimeout(poll, j.estado === "en_cola" ? 4000 : 2000);
  }

  function showResult(j) {
    const base = `api/trabajos/${j.id}/`;
    $("#btn-cancel").classList.add("hidden");
    $("#player").src = base + "video.mp4";
    $("#dl-mp4").href = base + "video.mp4";
    $("#dl-srt").href = base + "subtitulos.srt";
    $("#dl-json").href = base + "guion.json";
    const ul = $("#avisos");
    ul.innerHTML = "";
    (j.avisos || []).forEach((a) => { const li = document.createElement("li"); li.textContent = a; ul.appendChild(li); });
    $("#resultado").classList.remove("hidden");
  }

  $("#btn-cancel").addEventListener("click", async () => {
    if (!currentJob) return;
    try { await api(`api/trabajos/${currentJob}/cancelar`, {}); } catch (e) { /* nada */ }
    store.del(JOB_STORE);
    poll();
  });

  init();
})();
