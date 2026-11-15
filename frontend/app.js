import { letterboxGeometry, decode, classify } from './yolo.js';

const cfg = window.PALM_CONFIG || {};
const BUCKET = cfg.STORAGE_BUCKET || 'inspections';
const configured = !!(cfg.SUPABASE_URL && cfg.SUPABASE_ANON_KEY && !cfg.SUPABASE_URL.includes('YOUR-'));
// Tanpa login: semua akses memakai role "anon" Supabase (diatur lewat RLS di supabase/schema.sql).
const sb = configured
  ? window.supabase.createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY, { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } })
  : null;

const $ = id => document.getElementById(id);
const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); else console.warn(`[PalmVision] elemen #${id} tidak ada di index.html`); };
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let meta = null;          // isi model_meta.json
let session = null;       // sesi ONNX Runtime
let current = null;       // { bmp, dets:[{...,finalClass,rejected}] }
let chart = null;

const classColor = name => meta?.classes.find(c => c.name === name)?.color || '#555';
const prettyName = n => n.replace(/_/g, ' ');

/* ------------------------------ MODEL ------------------------------ */

async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} belum dipasang (HTTP ${r.status})`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(), chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function loadModel() {
  const status = $('modelStatus'), t0 = performance.now();
  const setS = (t, warn = false) => { status.textContent = t; status.classList.toggle('warn', warn); };
  try {
    setS('Memuat model… 0%');
    // meta dan model diunduh BERSAMAAN (sebelumnya berurutan)
    const metaP = fetch('model/model_meta.json', { cache: 'no-cache' }).then(r => {
      if (!r.ok) throw new Error('model_meta.json tidak ditemukan'); return r.json();
    });
    const bytesP = fetchWithProgress('model/best.onnx', (got, total) => {
      const mb = (got / 1048576).toFixed(1);
      setS(total ? `Mengunduh model… ${Math.min(99, Math.round(100 * got / total))}% (${mb} MB)` : `Mengunduh model… ${mb} MB`);
    });
    bytesP.catch(() => {});                       // cegah peringatan jika meta gagal lebih dulu
    meta = await metaP;
    renderModelInfo();
    loadHistory();                                // grafik butuh daftar kelas dari meta

    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    const bytes = await bytesP;
    const tDl = performance.now();
    setS('Menyiapkan model…');
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const tReady = performance.now();
    console.info(`[PalmVision] unduh ${((tDl - t0) / 1000).toFixed(1)} dtk, siapkan ${((tReady - tDl) / 1000).toFixed(1)} dtk`);

    const placeholder = String(meta.version).startsWith('placeholder');
    const preview = meta.preset === 'preview';
    setS(placeholder ? 'Model contoh (belum dilatih)'
      : preview ? 'Model preview · akurasi rendah, jangan dipakai untuk keputusan'
      : `Model aktif · ${meta.version}`, placeholder || preview);
    $('detectBtn').disabled = !$('imageInput').files.length;
    if ($('cameraBtn')) $('cameraBtn').disabled = false;
    if ($('imageInput').files.length && !current) analyzeFile($('imageInput').files[0]);

    // pemanasan: jalankan sekali dengan gambar kosong agar deteksi pertama tidak lambat
    const size = meta.imgsz, dummy = new ort.Tensor('float32', new Float32Array(3 * size * size).fill(0.45), [1, 3, size, size]);
    inferChain = inferChain.then(() => session.run({ [session.inputNames[0]]: dummy })).catch(() => {});
  } catch (e) {
    console.error(e);
    setS(`Model tidak siap: ${e.message}`, true);
  }
}

function renderModelInfo() {
  const rows = Object.entries(meta.metrics_test || {});
  const table = rows.length
    ? `<div class="table-wrap"><table><thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr></thead><tbody>` +
      rows.map(([k, v]) => `<tr><td>${esc(prettyName(k))}</td><td>${v.precision}</td><td>${v.recall}</td><td>${v.mAP50}</td><td>${v['mAP50-95']}</td></tr>`).join('') +
      `</tbody></table></div>`
    : `<p><strong>Belum ada metrik.</strong> Latih model dengan notebook di <code>training/</code> lalu timpa <code>model/model_meta.json</code>.</p>`;
  const thr = meta.classes.map(c => `${esc(prettyName(c.name))}: ${c.conf}`).join(' · ');
  $('modelInfo').innerHTML = `
    <p><strong>Versi:</strong> ${esc(meta.version)} · <strong>Resolusi:</strong> ${esc(meta.imgsz)} px</p>
    <p><strong>Kelas:</strong> ${esc(meta.classes.map(c => prettyName(c.name)).join(', '))}</p>
    <p><strong>Threshold per kelas:</strong> ${thr}</p>
    <p><strong>Metrik di data test (bukan jaminan di kebunmu):</strong></p>${table}
    <p class="small">${esc(meta.class_map || '')}</p>
    <p class="small">${esc(meta.dataset || '')}</p>
    <p class="small">Akurasi di foto baru bisa lebih rendah: buah tertutup pelepah, cahaya, dan sudut kamera sangat berpengaruh. Selalu verifikasi manual sebelum keputusan panen.</p>`;
}

const preCanvas = document.createElement('canvas');
let preCtx = null;
function preprocess(src, w, h, size) {
  const g = letterboxGeometry(w, h, size);
  if (preCanvas.width !== size) { preCanvas.width = preCanvas.height = size; preCtx = null; }
  preCtx ??= preCanvas.getContext('2d', { willReadFrequently: true });
  preCtx.fillStyle = 'rgb(114,114,114)'; preCtx.fillRect(0, 0, size, size);
  preCtx.drawImage(src, g.px, g.py, g.nw, g.nh);
  const d = preCtx.getImageData(0, 0, size, size).data, n = size * size, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { t[i] = d[4 * i] / 255; t[n + i] = d[4 * i + 1] / 255; t[2 * n + i] = d[4 * i + 2] / 255; }
  return { tensor: new ort.Tensor('float32', t, [1, 3, size, size]), geo: g };
}

// Semua inferensi lewat satu antrian: session.run tidak boleh dipanggil bersamaan.
let inferChain = Promise.resolve();
function inferSource(src, w, h) {
  const job = inferChain.then(async () => {
    const { tensor, geo } = preprocess(src, w, h, meta.imgsz);
    const out = await session.run({ [session.inputNames[0]]: tensor });
    const o = out[session.outputNames[0]];
    const boxes = decode(o.data, o.dims, geo, w, h, meta.classes.length);
    return classify(boxes, meta).map(d => ({ ...d, finalClass: d.className, rejected: false }));
  });
  inferChain = job.catch(() => {});
  return job;
}

async function runDetection(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  return { bmp, dets: await inferSource(bmp, bmp.width, bmp.height) };
}

/* ------------------------------ UI HASIL ------------------------------ */

function drawResult() {
  const { bmp, dets } = current;
  const cv = $('resultCanvas'); cv.width = bmp.width; cv.height = bmp.height;
  const ctx = cv.getContext('2d'); ctx.drawImage(bmp, 0, 0);
  const disp = Math.min(1, (cv.clientWidth || bmp.width) / bmp.width);      // skala tampil di layar
  const lw = Math.max(2, 2.5 / disp), fs = Math.max(12 / disp, bmp.width / 70);
  ctx.font = `600 ${fs}px sans-serif`; ctx.textBaseline = 'top';
  for (const d of dets) {
    if (d.rejected) continue;
    const x = d.x1 * bmp.width, y = d.y1 * bmp.height, w = (d.x2 - d.x1) * bmp.width, h = (d.y2 - d.y1) * bmp.height;
    const col = classColor(d.finalClass);
    ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.setLineDash(d.needsReview ? [lw * 3, lw * 2] : []);
    ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    const label = `${prettyName(d.finalClass)} ${Math.round(d.score * 100)}%${d.needsReview ? ' ?' : ''}`;
    const tw = ctx.measureText(label).width + 8, ty = Math.max(0, y - fs - 6);
    ctx.fillStyle = col; ctx.fillRect(x, ty, tw, fs + 6);
    ctx.fillStyle = '#fff'; ctx.fillText(label, x + 4, ty + 3);
  }
}

function renderResult() {
  const { dets } = current;
  const active = dets.filter(d => !d.rejected);
  $('totalPill').textContent = `${active.length} tandan`;

  const counts = {}; meta.classes.forEach(c => counts[c.name] = 0);
  active.forEach(d => counts[d.finalClass]++);
  const review = active.filter(d => d.needsReview).length;
  $('countChips').innerHTML = meta.classes.map(c =>
    `<span class="chip" style="--c:${c.color}"><i></i>${esc(prettyName(c.name))}: <b>${counts[c.name]}</b></span>`).join('') +
    (review ? `<span class="chip warn">Perlu dicek: <b>${review}</b></span>` : '');

  const opts = meta.classes.map(c => `<option value="${esc(c.name)}">${esc(prettyName(c.name))}</option>`).join('');
  const list = $('detectionList'); list.innerHTML = '';
  if (!dets.length) { list.innerHTML = '<span class="muted">Tidak ada tandan yang lolos threshold. Coba foto yang lebih dekat dan terang.</span>'; }
  dets.forEach((d, i) => {
    const row = document.createElement('div'); row.className = 'det-row' + (d.rejected ? ' rejected' : '');
    row.innerHTML = `<i style="background:${classColor(d.finalClass)}"></i>
      <select aria-label="Kelas deteksi ${i + 1}">${opts}</select>
      <span class="conf">${Math.round(d.score * 100)}%${d.needsReview ? ' · <em>cek</em>' : ''}</span>
      <button type="button" class="btn-mini">${d.rejected ? 'Pulihkan' : 'Tolak'}</button>`;
    const sel = row.querySelector('select'); sel.value = d.finalClass; sel.disabled = d.rejected;
    sel.addEventListener('change', () => { d.finalClass = sel.value; renderResult(); });
    row.querySelector('button').addEventListener('click', () => { d.rejected = !d.rejected; renderResult(); });
    list.appendChild(row);
  });
  drawResult();
}

/* ------------------------------ INFO KATEGORI (sidebar kamera) ------------------------------ */

// Panduan umum per kategori. Bukan hasil pengukuran; kriteria panen tiap kebun bisa berbeda.
const CLASS_INFO = {
  mentah:           { ciri: 'Buah kecil, hitam atau hijau gelap, belum ada rona merah.', saran: 'Belum waktunya dipanen. Biarkan menua dan periksa lagi nanti.' },
  menjelang_matang: { ciri: 'Warna hitam dengan transisi merah mulai tampak.', saran: 'Mendekati matang. Jadwalkan pemeriksaan ulang secara berkala.' },
  matang:           { ciri: 'Warna merah dominan, buah besar dan bulat.', saran: 'Kandidat panen. Verifikasi manual (warna, brondolan) sebelum dipanen.' },
  terlalu_matang:   { ciri: 'Warna kusam atau pudar dan banyak brondolan (buah lepas).', saran: 'Segera panen dan olah; keterlambatan dapat menurunkan kualitas minyak.' },
};
const infoBlock = name => CLASS_INFO[name]
  ? `<p><b>Ciri:</b> ${esc(CLASS_INFO[name].ciri)}</p><p><b>Saran:</b> ${esc(CLASS_INFO[name].saran)}</p>` : '';

function renderCamGuide() {
  if (!meta) return;
  $('camGuide').innerHTML = [...meta.classes].reverse().map(c =>
    `<div class="cam-card" style="--c:${esc(c.color)}"><div class="cam-card-head"><strong>${esc(prettyName(c.name))}</strong></div>${infoBlock(c.name)}</div>`).join('');
}

function camAdviceText(counts, review) {
  const n = k => counts[k] || 0, total = Object.values(counts).reduce((s, v) => s + v, 0);
  if (!total) return 'Belum ada tandan terdeteksi. Dekatkan kamera, pastikan cahaya cukup dan tandan tidak tertutup pelepah.';
  let t;
  if (n('terlalu_matang')) t = `${n('terlalu_matang')} tandan terlalu matang terdeteksi. Pertimbangkan segera dipanen setelah verifikasi manual.`;
  else if (n('matang')) t = `${n('matang')} tandan matang terdeteksi. Kandidat panen, verifikasi manual dulu.`;
  else if (n('menjelang_matang')) t = `Belum ada yang matang. ${n('menjelang_matang')} tandan menjelang matang, periksa lagi nanti.`;
  else t = 'Semua tandan yang terdeteksi masih mentah. Belum waktunya panen.';
  return review ? `${t} (${review} deteksi perlu dicek.)` : t;
}

let camSideAt = 0;
function updateCamSide(dets, force = false) {
  const now = performance.now();
  if (!force && now - camSideAt < 350) return;      // batasi pembaruan DOM ~3x/detik
  camSideAt = now;
  const counts = {}, sums = {};
  dets.forEach(d => { counts[d.className] = (counts[d.className] || 0) + 1; sums[d.className] = (sums[d.className] || 0) + d.score; });
  $('camTotal').textContent = `${dets.length} tandan`;
  $('camAdvice').textContent = camAdviceText(counts, dets.filter(d => d.needsReview).length);
  $('camCards').innerHTML = [...meta.classes].reverse().filter(c => counts[c.name]).map(c => {
    const avg = Math.round(100 * sums[c.name] / counts[c.name]);
    return `<div class="cam-card" style="--c:${esc(c.color)}"><div class="cam-card-head"><strong>${esc(prettyName(c.name))}</strong><span class="cam-count">${counts[c.name]}</span></div><p class="muted small">Keyakinan rata-rata ${avg}%</p>${infoBlock(c.name)}</div>`;
  }).join('');
  const rows = [...dets].sort((a, b) => b.score - a.score).slice(0, 12);
  $('camList').innerHTML = rows.length
    ? rows.map(d => `<div class="cam-row" style="--c:${esc(classColor(d.className))}"><i></i><span>${esc(prettyName(d.className))}</span><span class="bar"><b style="width:${Math.round(d.score * 100)}%"></b></span><span>${Math.round(d.score * 100)}%${d.needsReview ? ' <em>cek</em>' : ''}</span></div>`).join('')
    : '<span class="muted small">Belum ada deteksi.</span>';
}

/* ------------------------------ KAMERA LIVE ------------------------------ */

let camStream = null, camActive = false, camFacing = 'environment';

function stopStream() { camStream?.getTracks().forEach(t => t.stop()); camStream = null; }
function stopCamera() {
  camActive = false; stopStream(); $('camVideo').srcObject = null;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}
function tryFullscreen() {
  const d = $('cameraDialog');
  try { if (d.requestFullscreen && !document.fullscreenElement) d.requestFullscreen().catch(() => {}); } catch { /* opsional */ }
}

async function startCamera() {
  if (!session) return setMsg('Model belum siap.', true);
  if (!navigator.mediaDevices?.getUserMedia) return setMsg('Kamera butuh HTTPS (atau localhost) dan browser yang mendukung.', true);
  try {
    stopStream();
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: camFacing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    return setMsg(e.name === 'NotAllowedError' ? 'Izin kamera ditolak. Izinkan kamera di pengaturan browser.' : `Kamera gagal: ${e.message}`, true);
  }
  const v = $('camVideo'); v.srcObject = camStream; await v.play().catch(() => {});
  if (!$('cameraDialog').open) { $('cameraDialog').showModal(); renderCamGuide(); updateCamSide([], true); tryFullscreen(); }
  if (!camActive) { camActive = true; camLoop(); }
}

function fitOverlay(v, ov) {
  const W = v.clientWidth, H = v.clientHeight, vw = v.videoWidth, vh = v.videoHeight;
  if (!W || !H || !vw) return;
  const s = Math.min(W / vw, H / vh), w = vw * s, h = vh * s;
  ov.style.width = `${w}px`; ov.style.height = `${h}px`;
  ov.style.left = `${(W - w) / 2}px`; ov.style.top = `${(H - h) / 2}px`;
}

function drawLive(ov, v, dets) {
  fitOverlay(v, ov);
  ov.width = v.videoWidth; ov.height = v.videoHeight;
  const ctx = ov.getContext('2d'); ctx.clearRect(0, 0, ov.width, ov.height);
  const disp = (parseFloat(ov.style.width) || ov.width) / ov.width;          // skala tampil di layar
  const lw = Math.max(2, 2.5 / disp), fs = Math.max(12 / disp, ov.width / 60);
  ctx.font = `600 ${fs}px sans-serif`; ctx.textBaseline = 'top';
  for (const d of dets) {
    const x = d.x1 * ov.width, y = d.y1 * ov.height, w = (d.x2 - d.x1) * ov.width, h = (d.y2 - d.y1) * ov.height;
    const col = classColor(d.className);
    ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.setLineDash(d.needsReview ? [lw * 3, lw * 2] : []);
    ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    const label = `${prettyName(d.className)} ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(label).width + 8, ty = Math.max(0, y - fs - 6);
    ctx.fillStyle = col; ctx.fillRect(x, ty, tw, fs + 6); ctx.fillStyle = '#fff'; ctx.fillText(label, x + 4, ty + 3);
  }
}

async function camLoop() {
  const v = $('camVideo'), ov = $('camOverlay'); let last = performance.now(), fps = 0;
  while (camActive) {
    if (v.readyState >= 2 && v.videoWidth) {
      try {
        const dets = await inferSource(v, v.videoWidth, v.videoHeight);
        if (!camActive) break;
        drawLive(ov, v, dets);
        updateCamSide(dets);
        const now = performance.now(); fps = fps ? 0.8 * fps + 0.2 * (1000 / (now - last)) : 1000 / (now - last); last = now;
        $('camInfo').textContent = `Live · ${fps.toFixed(1)} fps`;
      } catch (e) { console.error(e); $('camInfo').textContent = `Deteksi gagal: ${e.message}`; camActive = false; break; }
    }
    await new Promise(r => requestAnimationFrame(r));
  }
}

async function captureFrame() {
  const v = $('camVideo'); if (!v.videoWidth) return;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const bmp = await createImageBitmap(c);
  stopCamera(); $('cameraDialog').close();
  setMsg('Menganalisis foto…');
  try {
    current = { bmp, dets: await inferSource(bmp, bmp.width, bmp.height) };
    $('result').classList.remove('hidden'); renderResult();
    $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    setMsg('Foto diambil dan dianalisis. Periksa hasilnya sebelum menyimpan.');
  } catch (e) { console.error(e); setMsg(`Deteksi gagal: ${e.message}`, true); }
}

/* ------------------------------ SIMPAN ------------------------------ */

function toJpegBlob(bmp, maxSide = 1600, q = 0.85) {
  const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas'); c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  return new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error('Gagal membuat JPEG')), 'image/jpeg', q));
}

async function saveInspection() {
  if (!sb) return setMsg('Supabase belum dikonfigurasi (config.js).', true);
  if (!current) return setMsg('Belum ada hasil deteksi untuk disimpan.', true);
  const btn = $('saveBtn'); btn.disabled = true; setMsg('Menyimpan…');
  let path = null, inspectionId = null;
  try {
    const { bmp, dets } = current;
    const blob = await toJpegBlob(bmp);
    path = `riwayat/${crypto.randomUUID()}.jpg`;
    let r = await sb.storage.from(BUCKET).upload(path, blob, { contentType: 'image/jpeg', upsert: false });
    if (r.error) throw r.error;

    const active = dets.filter(d => !d.rejected);
    const counts = {}; meta.classes.forEach(c => counts[c.name] = 0);
    active.forEach(d => counts[d.finalClass]++);

    const ins = await sb.from('inspections').insert({
      inspector: $('inspector').value.trim(), location: $('location').value.trim(),
      image_path: path, image_width: bmp.width, image_height: bmp.height,
      model_version: meta.version, total_bunches: active.length, counts,
    }).select('id').single();
    if (ins.error) throw ins.error;
    inspectionId = ins.data.id;

    if (dets.length) {
      const rows = dets.map(d => ({
        inspection_id: inspectionId, predicted_class: d.className,
        final_class: d.rejected ? null : d.finalClass,
        status: d.rejected ? 'rejected' : (d.finalClass !== d.className ? 'corrected' : 'auto'),
        confidence: Math.min(1, Math.max(0, d.score)), needs_review: d.needsReview,
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
      }));
      const dr = await sb.from('detections').insert(rows);
      if (dr.error) throw dr.error;
    }
    const chk = await sb.from('inspections').select('id', { count: 'exact', head: true }).eq('id', inspectionId);
    if (chk.error || !chk.count) throw new Error('Data terkirim tetapi tidak terbaca kembali (cek izin SELECT / schema.sql).');
    setMsg('Tersimpan ke Supabase ✔ (tabel inspections & detections)');
    await loadHistory();
  } catch (e) {
    console.error(e);
    // rollback agar tidak ada data setengah jadi
    if (inspectionId) await sb.from('inspections').delete().eq('id', inspectionId);
    if (path) await sb.storage.from(BUCKET).remove([path]);
    setMsg(saveErrorText(e), true);
  } finally { btn.disabled = false; }
}

function saveErrorText(e) {
  const m = String(e?.message || e);
  if (/row-level security|violates|permission denied|not authorized/i.test(m))
    return 'Ditolak database (izin akses). Jalankan ulang supabase/schema.sql di Supabase SQL Editor, lalu coba lagi.';
  if (/bucket not found/i.test(m)) return 'Bucket "inspections" belum ada. Jalankan supabase/schema.sql di Supabase SQL Editor.';
  if (/relation .* does not exist|schema cache/i.test(m)) return 'Tabel belum dibuat. Jalankan supabase/schema.sql di Supabase SQL Editor.';
  if (/failed to fetch|networkerror/i.test(m)) return 'Tidak bisa terhubung ke Supabase. Periksa koneksi internet.';
  return `Gagal menyimpan: ${m}`;
}

/* ------------------------------ RIWAYAT ------------------------------ */

async function loadHistory() {
  const body = $('historyBody');
  if (!sb) {
    body.innerHTML = `<tr><td colspan="7">Supabase belum dikonfigurasi.</td></tr>`;
    ['totalChecks', 'totalBunches', 'ripeShare', 'acceptRate'].forEach(id => $(id).textContent = '–');
    $('acceptNote').textContent = '';
    if (chart) { chart.destroy(); chart = null; }
    return;
  }
  const { data: rows, error } = await sb.from('inspections')
    .select('id,inspector,location,image_path,total_bunches,counts,created_at')
    .order('created_at', { ascending: false }).limit(100);
  if (error) { body.innerHTML = `<tr><td colspan="7">${esc(error.message)}</td></tr>`; return; }

  body.innerHTML = rows.length ? '' : '<tr><td colspan="7">Belum ada data.</td></tr>';
  for (const r of rows) {
    const url = sb.storage.from(BUCKET).getPublicUrl(r.image_path).data.publicUrl;
    const comp = Object.entries(r.counts || {}).map(([k, v]) => `${esc(prettyName(k))}: ${v}`).join(' · ');
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="${esc(url)}" target="_blank" rel="noopener"><img class="thumb" src="${esc(url)}" alt="" loading="lazy"></a></td>
      <td>${esc(r.inspector || '-')}</td><td>${esc(r.location || '-')}</td><td>${r.total_bunches}</td><td>${comp}</td>
      <td>${new Date(r.created_at).toLocaleString('id-ID')}</td>
      <td><button type="button" class="btn-mini">Hapus</button></td>`;
    tr.querySelector('button').addEventListener('click', async () => {
      if (!confirm('Hapus pemeriksaan ini?')) return;
      await sb.from('inspections').delete().eq('id', r.id);
      await sb.storage.from(BUCKET).remove([r.image_path]);
      loadHistory();
    });
    body.appendChild(tr);
  }

  const total = rows.reduce((s, r) => s + r.total_bunches, 0);
  const ripe = rows.reduce((s, r) => s + (r.counts?.matang || 0), 0);
  $('totalChecks').textContent = rows.length;
  $('totalBunches').textContent = total;
  $('ripeShare').textContent = total ? `${Math.round(100 * ripe / total)}%` : '–';

  drawChart(rows);
  loadFeedback();
}

function drawChart(rows) {
  if (!meta) return;
  const recent = [...rows].slice(0, 12).reverse();
  const cfgc = {
    type: 'bar',
    data: {
      labels: recent.map(r => new Date(r.created_at).toLocaleDateString('id-ID', { day: '2-digit', month: 'short' }) + ' #' + r.id.slice(0, 4)),
      datasets: meta.classes.map(c => ({ label: prettyName(c.name), backgroundColor: c.color, data: recent.map(r => r.counts?.[c.name] || 0) })),
    },
    options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } } },
  };
  if (chart) chart.destroy();
  chart = new Chart($('historyChart'), cfgc);
}

async function loadFeedback() {
  const { data, error } = await sb.from('model_feedback_summary').select('*');
  if (error || !data?.length) { $('acceptRate').textContent = '–'; $('acceptNote').textContent = ''; return; }
  const total = data.reduce((s, r) => s + r.total, 0), ok = data.reduce((s, r) => s + r.accepted, 0);
  $('acceptRate').textContent = total ? `${Math.round(100 * ok / total)}%` : '–';
  $('acceptNote').textContent = total < 50 ? ` n=${total} (masih sedikit)` : ` n=${total}`;
}

/* ------------------------------ PESAN ------------------------------ */

function setMsg(t, err = false) { const m = $('message'); m.textContent = t; m.classList.toggle('error', err); }

/* ------------------------------ INIT ------------------------------ */

async function analyzeFile(f) {
  if (!f) return;
  if (!session) return setMsg('Model masih dimuat. Tunggu sebentar; analisis jalan otomatis setelah siap.', false);
  if (f.size > 20 * 1024 * 1024) return setMsg('Ukuran gambar maksimal 20 MB.', true);
  $('detectBtn').disabled = true; $('result').classList.add('hidden'); setMsg('Menjalankan deteksi…');
  try {
    const t0 = performance.now();
    current = await runDetection(f);
    $('result').classList.remove('hidden'); renderResult();
    setMsg(`Selesai dalam ${((performance.now() - t0) / 1000).toFixed(1)} dtk. Periksa hasilnya sebelum menyimpan.`);
  } catch (e) { console.error(e); setMsg(`Deteksi gagal: ${e.message}`, true); }
  finally { $('detectBtn').disabled = false; }
}

on('imageInput', 'change', () => {
  const f = $('imageInput').files[0];
  $('fileName').textContent = f ? f.name : 'Belum ada file dipilih';
  $('detectBtn').disabled = !f || !session;
  analyzeFile(f);                      // otomatis begitu gambar dipilih
});
on('detectBtn', 'click', () => analyzeFile($('imageInput').files[0]));

on('cameraBtn', 'click', startCamera);
on('camShot', 'click', captureFrame);
on('camSwitch', 'click', () => { camFacing = camFacing === 'environment' ? 'user' : 'environment'; startCamera(); });
on('camClose', 'click', () => $('cameraDialog').close());
on('cameraDialog', 'close', stopCamera);

on('saveBtn', 'click', saveInspection);

if (!configured) $('configBanner').classList.remove('hidden');
loadHistory();
if (sb) sb.from('inspections').select('id', { count: 'exact', head: true }).then(r => {
  if (r.error) { console.error('[PalmVision] Supabase:', r.error); setMsg(`Supabase error: ${r.error.message}. ${saveErrorText(r.error)}`, true); }
  else console.info(`[PalmVision] Supabase tersambung. Baris di inspections: ${r.count}`);
});
loadModel();