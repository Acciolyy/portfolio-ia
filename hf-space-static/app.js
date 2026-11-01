(function () {
  const MODEL_URL = "https://huggingface.co/Acciolyyy/lenet5-mnist-unb/resolve/main/model.onnx";
  const LOW_CONFIDENCE = 0.7;
  const pad = document.getElementById("pad");
  const ctx = pad.getContext("2d", { willReadFrequently: true });
  const seen = document.getElementById("seen");
  const statusEl = document.getElementById("status");
  const warning = document.getElementById("warning");
  const classifyBtn = document.getElementById("classify");
  const bars = document.getElementById("bars");
  let session = null;
  let drawn = false;

  function clearPad() {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, pad.width, pad.height);
    drawn = false;
    warning.hidden = true;
    seen.getContext("2d").clearRect(0, 0, 28, 28);
    renderBars(null);
  }

  // ---- desenho (mouse, toque e caneta via pointer events) ----
  let last = null;
  function pos(e) {
    const r = pad.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * pad.width) / r.width, y: ((e.clientY - r.top) * pad.height) / r.height };
  }
  function stroke(a, b) {
    ctx.strokeStyle = "#000";
    ctx.fillStyle = "#000";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    drawn = true;
  }
  pad.addEventListener("pointerdown", (e) => {
    pad.setPointerCapture(e.pointerId);
    last = pos(e);
    stroke(last, last);
  });
  pad.addEventListener("pointermove", (e) => {
    if (!last) return;
    const p = pos(e);
    stroke(last, p);
    last = p;
  });
  const end = () => { last = null; if (drawn && session) classify(); };
  pad.addEventListener("pointerup", end);
  pad.addEventListener("pointercancel", end);

  // ---- resultado ----
  function renderBars(probs) {
    bars.innerHTML = "";
    const top = probs ? probs.indexOf(Math.max(...probs)) : -1;
    for (let i = 0; i < 10; i++) {
      const p = probs ? probs[i] : 0;
      const li = document.createElement("li");
      if (i === top) li.className = "top";
      li.innerHTML = `<span>${i}</span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span>${(p * 100).toFixed(1)}%</span>`;
      bars.appendChild(li);
    }
  }

  function showSeen(view) {
    const c = seen.getContext("2d");
    const img = c.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(view[i] * 255);
      img.data.set([v, v, v, 255], i * 4);
    }
    c.putImageData(img, 0, 0);
  }

  async function classify() {
    if (!session || !drawn) return;
    const { data } = ctx.getImageData(0, 0, pad.width, pad.height);
    const gray = new Float32Array(pad.width * pad.height);
    for (let i = 0; i < gray.length; i++) {
      // mesma conversao para cinza do Pillow (modo "L")
      gray[i] = (data[i * 4] * 19595 + data[i * 4 + 1] * 38470 + data[i * 4 + 2] * 7471 + 0x8000) >> 16;
    }
    const out = LenetPreprocess.preprocess(gray, pad.height, pad.width);
    if (!out) return;
    showSeen(out.canvas);
    const result = await session.run({ input: new ort.Tensor("float32", out.tensor, [1, 1, 28, 28]) });
    const logits = Array.from(result.logits.data);
    const m = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sum);
    renderBars(probs);
    warning.hidden = Math.max(...probs) >= LOW_CONFIDENCE;
  }

  document.getElementById("clear").addEventListener("click", clearPad);
  classifyBtn.addEventListener("click", classify);

  // ---- modelo ----
  async function loadModel() {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    ort.env.wasm.numThreads = 1;
    try {
      const resp = await fetch(MODEL_URL);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      session = await ort.InferenceSession.create(new Uint8Array(await resp.arrayBuffer()), { executionProviders: ["wasm"] });
      statusEl.textContent = "Modelo carregado. Desenhe um dígito.";
      classifyBtn.disabled = false;
      window.__modelReady = true;
    } catch (err) {
      statusEl.textContent = "Não foi possível carregar o modelo: " + err.message;
    }
  }

  clearPad();
  loadModel();
})();
