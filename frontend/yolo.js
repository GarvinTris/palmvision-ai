// Fungsi murni untuk pra/pasca-proses YOLO26 (ONNX). Tidak bergantung pada DOM,
// jadi bisa diuji di Node dan dipakai di browser.

/** Geometri letterbox ala Ultralytics (resize proporsional + padding di tengah). */
export function letterboxGeometry(w, h, size) {
  const s = Math.min(size / w, size / h);
  const nw = Math.round(w * s), nh = Math.round(h * s);
  return { s, nw, nh, px: Math.floor((size - nw) / 2), py: Math.floor((size - nh) / 2) };
}

export function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** NMS lintas-kelas: satu tandan fisik tidak boleh punya dua label sekaligus. */
export function nms(boxes, thr = 0.6) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of sorted) if (keep.every(k => iou(k, b) < thr)) keep.push(b);
  return keep;
}

/**
 * Decode keluaran model → kotak ternormalisasi (0..1) pada gambar asli.
 * Mendukung dua format:
 *  - end-to-end tanpa NMS (YOLO26 default): dims [1, N, 6] = x1,y1,x2,y2,score,cls
 *  - klasik: dims [1, 4+nc, N] = cx,cy,w,h,skor-per-kelas
 */
export function decode(data, dims, geo, W, H, nc, minScore = 0.05) {
  if (dims.length !== 3) throw new Error(`Bentuk output tidak dikenal: [${dims}]`);
  const raw = [];
  if (dims[2] === 6) {
    for (let i = 0; i < dims[1]; i++) {
      const o = i * 6, score = data[o + 4], cls = Math.round(data[o + 5]);
      if (!(score >= minScore) || cls < 0 || cls >= nc) continue;
      raw.push({ x1: data[o], y1: data[o + 1], x2: data[o + 2], y2: data[o + 3], score, cls });
    }
  } else {
    if (dims[1] !== 4 + nc) throw new Error(`Jumlah kelas model (${dims[1] - 4}) ≠ model_meta.json (${nc}).`);
    const N = dims[2];
    for (let i = 0; i < N; i++) {
      let best = 0, bc = 0;
      for (let c = 0; c < nc; c++) { const v = data[(4 + c) * N + i]; if (v > best) { best = v; bc = c; } }
      if (best < minScore) continue;
      const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
      raw.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bc });
    }
  }
  const clamp = (v, max) => Math.min(Math.max(v, 0), max);
  const mapped = raw.map(b => ({
    x1: clamp((b.x1 - geo.px) / geo.s, W) / W, y1: clamp((b.y1 - geo.py) / geo.s, H) / H,
    x2: clamp((b.x2 - geo.px) / geo.s, W) / W, y2: clamp((b.y2 - geo.py) / geo.s, H) / H,
    score: b.score, cls: b.cls,
  })).filter(b => b.x2 - b.x1 > 0.002 && b.y2 - b.y1 > 0.002);
  return nms(mapped, 0.6);
}

/** Terapkan threshold per kelas + tandai deteksi di zona ragu-ragu. */
export function classify(boxes, meta) {
  const margin = meta.review_margin ?? 0.15;
  const out = [];
  for (const b of boxes) {
    const c = meta.classes[b.cls];
    if (!c || b.score < c.conf) continue;
    out.push({ ...b, className: c.name, needsReview: b.score < c.conf + margin });
  }
  return out.sort((a, b) => a.y1 - b.y1 || a.x1 - b.x1);
}
