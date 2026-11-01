// Pre-processamento identico ao do app Gradio (hf-space/app.py): cinza -> inversao se o
// fundo for branco -> recorte do traco -> 20 px dentro de 28x28 -> centro de massa -> normalizacao.
// Funciona no navegador (global LenetPreprocess) e no Node (module.exports).
(function (root) {
  const MEAN = 0.1307;
  const STD = 0.3081;
  const f32 = Math.fround;

  // round() do Python: arredonda .5 para o par mais proximo.
  function pyRound(x) {
    const f = Math.floor(x);
    const d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }

  function lanczos(x) {
    if (x <= -3 || x >= 3) return 0;
    if (x === 0) return 1;
    const px = Math.PI * x;
    return (3 * Math.sin(px) * Math.sin(px / 3)) / (px * px);
  }

  // Coeficientes de reamostragem no mesmo esquema do Pillow (Image.resize com LANCZOS).
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = 3 * filterScale;
    const out = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
      const w = [];
      let sum = 0;
      for (let x = xmin; x < xmax; x++) {
        const v = lanczos((x - center + 0.5) / filterScale);
        w.push(v);
        sum += v;
      }
      out.push({ xmin, w: w.map((v) => (sum !== 0 ? v / sum : v)) });
    }
    return out;
  }

  const clip8 = (v) => Math.min(255, Math.max(0, Math.round(v)));

  // Redimensiona uma imagem uint8 (h x w) para (newH x newW): passada horizontal e depois vertical,
  // com arredondamento para uint8 entre as passadas, como o Pillow.
  function resizeLanczos(src, h, w, newH, newW) {
    const cx = coeffs(w, newW);
    const tmp = new Float64Array(h * newW);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < newW; x++) {
        let s = 0;
        const c = cx[x];
        for (let k = 0; k < c.w.length; k++) s += src[y * w + c.xmin + k] * c.w[k];
        tmp[y * newW + x] = clip8(s);
      }
    }
    const cy = coeffs(h, newH);
    const out = new Float64Array(newH * newW);
    for (let y = 0; y < newH; y++) {
      const c = cy[y];
      for (let x = 0; x < newW; x++) {
        let s = 0;
        for (let k = 0; k < c.w.length; k++) s += tmp[(c.xmin + k) * newW + x] * c.w[k];
        out[y * newW + x] = clip8(s);
      }
    }
    return out;
  }

  // gray: array de tamanho h*w com valores em [0, 255]. Devolve { tensor, canvas } ou null.
  // tensor: Float32Array(784) normalizado; canvas: Float32Array(784) em [0, 1] (o que o modelo "ve").
  function preprocess(gray, h, w) {
    let g = Float32Array.from(gray);
    let borderSum = 0;
    for (let x = 0; x < w; x++) borderSum += g[x] + g[(h - 1) * w + x];
    for (let y = 0; y < h; y++) borderSum += g[y * w] + g[y * w + w - 1];
    if (borderSum / (2 * w + 2 * h) > 127) for (let i = 0; i < g.length; i++) g[i] = f32(255 - g[i]);

    let mn = Infinity;
    let mx = -Infinity;
    for (const v of g) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    for (let i = 0; i < g.length; i++) g[i] = f32(g[i] - mn);
    mx = f32(mx - mn);
    if (mx < 1e-3) return null;
    for (let i = 0; i < g.length; i++) g[i] = f32(f32(g[i] / mx) * 255);

    let y0 = h, y1 = -1, x0 = w, x1 = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (g[y * w + x] > 30) {
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
        }
      }
    }
    if (y1 < 0) return null;
    const ch = y1 - y0 + 1;
    const cw = x1 - x0 + 1;
    const crop = new Float64Array(ch * cw);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = Math.trunc(g[(y0 + y) * w + x0 + x]);

    const scale = 20 / Math.max(ch, cw);
    const newW = Math.max(1, pyRound(cw * scale));
    const newH = Math.max(1, pyRound(ch * scale));
    const small = resizeLanczos(crop, ch, cw, newH, newW);

    const canvas = new Float64Array(784);
    const top = Math.floor((28 - newH) / 2);
    const left = Math.floor((28 - newW) / 2);
    for (let y = 0; y < newH; y++) for (let x = 0; x < newW; x++) canvas[(top + y) * 28 + left + x] = small[y * newW + x];

    let total = 0, cy = 0, cx = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = canvas[y * 28 + x];
        total += v;
        cy += y * v;
        cx += x * v;
      }
    }
    cy /= total;
    cx /= total;
    const dy = 13.5 - cy;
    const dx = 13.5 - cx;

    // Deslocamento com interpolacao linear (equivale a scipy.ndimage.shift(order=1), fora = 0).
    const shifted = new Float64Array(784);
    const at = (y, x) => (y < 0 || y > 27 || x < 0 || x > 27 ? 0 : canvas[y * 28 + x]);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const sy = y - dy;
        const sx = x - dx;
        const y0i = Math.floor(sy);
        const x0i = Math.floor(sx);
        const fy = sy - y0i;
        const fx = sx - x0i;
        shifted[y * 28 + x] =
          at(y0i, x0i) * (1 - fy) * (1 - fx) + at(y0i, x0i + 1) * (1 - fy) * fx +
          at(y0i + 1, x0i) * fy * (1 - fx) + at(y0i + 1, x0i + 1) * fy * fx;
      }
    }

    const view = new Float32Array(784);
    const tensor = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      const v = f32(Math.min(255, Math.max(0, shifted[i])) / 255);
      view[i] = v;
      tensor[i] = f32(f32(v - MEAN) / STD);
    }
    return { tensor, canvas: view };
  }

  const api = { preprocess, MEAN, STD };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LenetPreprocess = api;
})(typeof self !== "undefined" ? self : this);
