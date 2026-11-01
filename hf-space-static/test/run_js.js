// Le um JSON {w,h,gray} por imagem e escreve os tensores do pre-processamento em JS.
const fs = require("fs");
const { preprocess } = require("../preprocess.js");
const items = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = items.map((it) => {
  const r = preprocess(it.gray, it.h, it.w);
  return r ? Array.from(r.tensor) : null;
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
