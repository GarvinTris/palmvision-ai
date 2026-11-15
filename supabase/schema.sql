-- PalmVision AI — skema Supabase
-- Jalankan sekali di: Supabase Dashboard → SQL Editor → New query → Run.
-- Aman dijalankan ulang (idempotent).

-- 1. TABEL ------------------------------------------------------------------

create table if not exists public.inspections (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null default auth.uid() references auth.users(id) on delete cascade,
  inspector      text not null default '',
  location       text not null default '',
  image_path     text not null,                       -- path di Storage: <user_id>/<uuid>.jpg
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
  user_id        uuid not null default auth.uid() references auth.users(id) on delete cascade,
  predicted_class text not null,                      -- keluaran model
  final_class     text,                               -- kelas setelah diperiksa manusia (null jika ditolak)
  status          text not null default 'auto' check (status in ('auto','corrected','rejected')),
  confidence      real not null check (confidence >= 0 and confidence <= 1),
  needs_review    boolean not null default false,     -- confidence di zona ragu-ragu
  x1 real, y1 real, x2 real, y2 real,                 -- koordinat ternormalisasi 0..1
  created_at      timestamptz not null default now()
);

create index if not exists inspections_user_created_idx on public.inspections (user_id, created_at desc);
create index if not exists detections_inspection_idx    on public.detections (inspection_id);
create index if not exists detections_status_idx        on public.detections (user_id, status);

-- 2. ROW LEVEL SECURITY -----------------------------------------------------
-- Tiap pengguna hanya melihat dan mengubah datanya sendiri.

alter table public.inspections enable row level security;
alter table public.detections  enable row level security;

drop policy if exists "inspections_select_own" on public.inspections;
drop policy if exists "inspections_insert_own" on public.inspections;
drop policy if exists "inspections_update_own" on public.inspections;
drop policy if exists "inspections_delete_own" on public.inspections;
create policy "inspections_select_own" on public.inspections for select to authenticated using (user_id = auth.uid());
create policy "inspections_insert_own" on public.inspections for insert to authenticated with check (user_id = auth.uid());
create policy "inspections_update_own" on public.inspections for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy "inspections_delete_own" on public.inspections for delete to authenticated using (user_id = auth.uid());

drop policy if exists "detections_select_own" on public.detections;
drop policy if exists "detections_insert_own" on public.detections;
drop policy if exists "detections_update_own" on public.detections;
drop policy if exists "detections_delete_own" on public.detections;
create policy "detections_select_own" on public.detections for select to authenticated using (user_id = auth.uid());
create policy "detections_insert_own" on public.detections for insert to authenticated
  with check (
    user_id = auth.uid()
    and exists (select 1 from public.inspections i where i.id = inspection_id and i.user_id = auth.uid())
  );
create policy "detections_update_own" on public.detections for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy "detections_delete_own" on public.detections for delete to authenticated using (user_id = auth.uid());

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

-- 4. STORAGE ----------------------------------------------------------------
-- Bucket publik-baca (URL gambar bisa ditampilkan langsung); hanya pemilik yang boleh upload/hapus
-- ke folder bernama user_id miliknya. Hanya JPEG, maks 5 MB.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('inspections', 'inspections', true, 5242880, array['image/jpeg'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "inspections_upload_own_folder" on storage.objects;
drop policy if exists "inspections_delete_own_folder" on storage.objects;
create policy "inspections_upload_own_folder" on storage.objects for insert to authenticated
  with check (bucket_id = 'inspections' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "inspections_delete_own_folder" on storage.objects for delete to authenticated
  using (bucket_id = 'inspections' and (storage.foldername(name))[1] = auth.uid()::text);
