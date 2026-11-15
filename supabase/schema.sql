-- PalmVision AI — skema Supabase (TANPA LOGIN)
-- Jalankan sekali di: Supabase Dashboard → SQL Editor → New query → Run.
-- Aman dijalankan ulang (idempotent) dan aman untuk database yang SUDAH berisi data versi lama
-- (versi lama mewajibkan login; bagian "MIGRASI" di bawah melonggarkannya).

-- 1. TABEL ------------------------------------------------------------------

create table if not exists public.inspections (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid,                                -- tidak dipakai lagi (tanpa login); dibiarkan agar data lama aman
  inspector      text not null default '',
  location       text not null default '',
  image_path     text not null,                       -- path di Storage: riwayat/<uuid>.jpg
  image_width    int,
  image_height   int,
  model_version  text,
  total_bunches  int  not null default 0,             -- hanya deteksi yang tidak ditolak
  counts         jsonb not null default '{}'::jsonb,  -- {"mentah":3,"matang":2,...}
  created_at     timestamptz not null default now()
);

create table if not exists public.detections (
  id             bigint generated always as identity primary key,
  inspection_id  uuid not null references public.inspections(id) on delete cascade,
  user_id        uuid,
  predicted_class text not null,                      -- keluaran model
  final_class     text,                               -- kelas setelah diperiksa manusia (null jika ditolak)
  status          text not null default 'auto' check (status in ('auto','corrected','rejected')),
  confidence      real not null check (confidence >= 0 and confidence <= 1),
  needs_review    boolean not null default false,     -- confidence di zona ragu-ragu
  x1 real, y1 real, x2 real, y2 real,                 -- koordinat ternormalisasi 0..1
  created_at      timestamptz not null default now()
);

-- MIGRASI dari versi lama (yang mewajibkan login): lepas kewajiban user_id & relasinya ke auth.users.
alter table public.inspections alter column user_id drop not null;
alter table public.inspections alter column user_id drop default;
alter table public.inspections drop constraint if exists inspections_user_id_fkey;
alter table public.detections  alter column user_id drop not null;
alter table public.detections  alter column user_id drop default;
alter table public.detections  drop constraint if exists detections_user_id_fkey;

create index if not exists inspections_created_idx   on public.inspections (created_at desc);
create index if not exists detections_inspection_idx on public.detections (inspection_id);
create index if not exists detections_status_idx     on public.detections (status);

-- 2. ROW LEVEL SECURITY -----------------------------------------------------
-- Tanpa login: role "anon" (kunci publishable di frontend) boleh membaca, menambah, dan menghapus riwayat.
-- Catatan: siapa pun yang punya URL aplikasi bisa melihat/menghapus riwayat. Cocok untuk satu tim/kebun.
-- Jika nanti butuh pembatasan per pengguna, aktifkan kembali login dan kebijakan berbasis auth.uid().

alter table public.inspections enable row level security;
alter table public.detections  enable row level security;

-- bersihkan kebijakan lama (versi login) dan versi ini
do $$
declare p record;
begin
  for p in
    select policyname, tablename from pg_policies
    where schemaname = 'public' and tablename in ('inspections', 'detections')
  loop
    execute format('drop policy if exists %I on public.%I', p.policyname, p.tablename);
  end loop;
end $$;

create policy "inspections_select_all" on public.inspections for select to anon, authenticated using (true);
create policy "inspections_insert_all" on public.inspections for insert to anon, authenticated with check (true);
create policy "inspections_delete_all" on public.inspections for delete to anon, authenticated using (true);

create policy "detections_select_all" on public.detections for select to anon, authenticated using (true);
create policy "detections_insert_all" on public.detections for insert to anon, authenticated with check (true);
create policy "detections_delete_all" on public.detections for delete to anon, authenticated using (true);

-- Izin tingkat tabel (RLS di atas yang membatasi baris)
grant usage on schema public to anon, authenticated;
grant select, insert, delete on public.inspections to anon, authenticated;
grant select, insert, delete on public.detections  to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated;

-- 3. RINGKASAN KOREKSI (berguna untuk mengukur akurasi di lapangan) ----------

create or replace view public.model_feedback_summary
with (security_invoker = true) as
select
  predicted_class,
  count(*)                                         as total,
  count(*) filter (where status = 'auto')          as accepted,
  count(*) filter (where status = 'corrected')     as corrected,
  count(*) filter (where status = 'rejected')      as rejected,
  round(100.0 * count(*) filter (where status = 'auto') / nullif(count(*), 0), 1) as accepted_pct
from public.detections
group by predicted_class;

grant select on public.model_feedback_summary to anon, authenticated;

-- 4. STORAGE ----------------------------------------------------------------
-- Bucket publik-baca (URL gambar bisa ditampilkan langsung). Hanya JPEG, maks 5 MB.
-- Upload/hapus dibuka untuk anon, tetapi hanya ke bucket "inspections".

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('inspections', 'inspections', true, 5242880, array['image/jpeg'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "inspections_upload_own_folder" on storage.objects;
drop policy if exists "inspections_delete_own_folder" on storage.objects;
drop policy if exists "inspections_upload_public"     on storage.objects;
drop policy if exists "inspections_delete_public"     on storage.objects;
create policy "inspections_upload_public" on storage.objects for insert to anon, authenticated
  with check (bucket_id = 'inspections');
create policy "inspections_delete_public" on storage.objects for delete to anon, authenticated
  using (bucket_id = 'inspections');