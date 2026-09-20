-- ==============================================================================
-- ArtSpeak Supabase Database Schema v2
-- Welcome -> Caregiver/Therapist Signup+Login -> Child Profile ->
-- Session (face video + scribble stream) -> ML Predictions -> LLM Report ->
-- Routed to Caregiver, and additionally to Therapist when needed
-- ==============================================================================
-- This version replaces the earlier single polymorphic `profiles` table with
-- three explicit tables (caregivers, therapists, children) so each one maps
-- 1:1 onto its own signup form instead of sharing a table full of nullable
-- columns. Everything downstream (sessions, predictions, reports, RLS)
-- has been re-pointed at the new tables.
-- ==============================================================================

create extension if not exists pgcrypto;

-- ==============================================================================
-- 1. CAREGIVERS
-- One row per auth.users entry. Login = email + password (native Supabase
-- Auth, no extra table needed for that).
-- Signup fields: name, age, gender, relation_with_child, email, password.
-- ==============================================================================
create table if not exists public.caregivers (
  id uuid primary key references auth.users(id) on delete cascade,
  name text not null,
  age integer check (age between 16 and 100),
  gender text check (gender in ('Male', 'Female', 'Non-binary', 'Prefer not to say')),
  relation_with_child text not null
    check (relation_with_child in ('Parent', 'Guardian', 'Grandparent', 'Sibling', 'Other')),
  email text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Populated automatically from auth signup metadata, see handle_new_caregiver() below.
-- Client call:
--   supabase.auth.signUp(
--     email: email, password: password,
--     data: {role:'caregiver', name, age, gender, relation_with_child}
--   )

-- ==============================================================================
-- 2. THERAPISTS
-- Signup fields: name, age, gender, qualification, license_number, email, password.
-- Login: a system-generated username (derived from name) + the password they
-- set at signup. Supabase Auth always needs an email internally, so we sign
-- them up under a synthetic "login_email" and keep their real email only for
-- verification / sending them their username.
--
-- Flow:
--   1. Client calls RPC generate_unique_therapist_username(name) -> "sarah4821"
--   2. login_email := "sarah4821@artspeak.internal"
--   3. supabase.auth.signUp(email: login_email, password: <chosen by therapist>,
--        data: {role:'therapist', name, age, gender, qualification,
--               license_number, contact_email, username})
--   4. An admin reviews license_number and sets verified = true
--   5. A Supabase Edge Function (trigger: verified false->true) emails the
--      username to contact_email and sets credentials_emailed = true.
--      NOTE: only the username is ever emailed -- never the password, since
--      the therapist already chose it at signup.
--   6. Therapist login screen collects {name/username, password}; client
--      calls RPC get_therapist_login_email(username) to resolve login_email,
--      then supabase.auth.signInWithPassword(email: login_email, password).
-- ==============================================================================
create table if not exists public.therapists (
  id uuid primary key references auth.users(id) on delete cascade,
  name text not null,
  age integer check (age between 21 and 100),
  gender text check (gender in ('Male', 'Female', 'Non-binary', 'Prefer not to say')),
  qualification text not null,             -- e.g. "M.Phil Clinical Psychology"
  license_number text not null unique,
  contact_email text not null unique,      -- real email, used for verification + credential delivery
  username text not null unique,           -- system-generated login identifier (from name)
  login_email text not null unique,        -- synthetic: {username}@artspeak.internal, used by Supabase Auth
  verified boolean not null default false, -- admin approves license before the account is usable
  credentials_emailed boolean not null default false,
  created_at timestamptz not null default now()
);

create or replace function public.generate_unique_therapist_username(p_full_name text)
returns text
language plpgsql
as $$
declare
  base_slug text;
  candidate text;
begin
  base_slug := lower(regexp_replace(trim(p_full_name), '[^a-zA-Z]+', '', 'g'));
  if base_slug = '' then
    base_slug := 'therapist';
  end if;
  loop
    candidate := base_slug || floor(random() * 9000 + 1000)::int::text;
    exit when not exists (select 1 from public.therapists where username = candidate);
  end loop;
  return candidate;
end;
$$;

create or replace function public.get_therapist_login_email(p_username text)
returns text
language sql
security definer
stable
as $$
  select login_email from public.therapists where username = p_username and verified = true;
$$;

revoke all on function public.get_therapist_login_email(text) from public;
grant execute on function public.get_therapist_login_email(text) to anon, authenticated;
grant execute on function public.generate_unique_therapist_username(text) to anon, authenticated;

-- ==============================================================================
-- 3. CHILDREN
-- Child rows are caregiver-owned records (no auth.users row of their own).
-- Signup/profile fields: name, age, gender, conditions, hobbies,
-- known_triggers, calming_activities, therapist_needed.
-- Arrays (not free text) so the ML/LLM layer and any future filtering can
-- treat these as structured tags rather than parsing prose.
-- ==============================================================================
create table if not exists public.children (
  id uuid primary key default gen_random_uuid(),
  caregiver_id uuid not null references public.caregivers(id) on delete cascade,
  name text not null,
  age integer not null check (age between 1 and 21),
  gender text check (gender in ('Male', 'Female', 'Non-binary', 'Prefer not to say')),
  conditions text[] not null default '{}',           -- e.g. {'Autism Spectrum Disorder','ADHD'}
  hobbies text[] not null default '{}',               -- e.g. {'trains','drawing animals'}
  known_triggers text[] not null default '{}',        -- e.g. {'loud noises','sudden changes'}
  calming_activities text[] not null default '{}',    -- e.g. {'deep pressure','music'}
  needs_therapist boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_children_caregiver on public.children(caregiver_id);
create index if not exists idx_children_needs_therapist on public.children(needs_therapist) where needs_therapist;

-- ==============================================================================
-- 4. CHILD <-> THERAPIST ASSIGNMENTS
-- One active assignment per child at a time. Auto-assigned the moment a
-- child's needs_therapist flips to true (round-robin over verified,
-- least-loaded therapists) -- see trigger below -- and can also be
-- reassigned manually by an admin/caregiver by inserting a new active row.
-- ==============================================================================
create table if not exists public.child_therapist_assignments (
  id uuid primary key default gen_random_uuid(),
  child_id uuid not null references public.children(id) on delete cascade,
  therapist_id uuid not null references public.therapists(id) on delete cascade,
  assigned_by uuid references public.caregivers(id),
  active boolean not null default true,
  assigned_at timestamptz not null default now(),
  ended_at timestamptz
);

create unique index if not exists idx_one_active_therapist_per_child
  on public.child_therapist_assignments(child_id) where (active);

create index if not exists idx_assignments_therapist
  on public.child_therapist_assignments(therapist_id) where (active);

create or replace function public.auto_assign_therapist()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  chosen_therapist uuid;
begin
  if new.needs_therapist = true and
     not exists (select 1 from public.child_therapist_assignments where child_id = new.id and active) then

    select t.id into chosen_therapist
    from public.therapists t
    where t.verified = true
    order by (
      select count(*) from public.child_therapist_assignments a
      where a.therapist_id = t.id and a.active
    ) asc
    limit 1;

    if chosen_therapist is not null then
      insert into public.child_therapist_assignments (child_id, therapist_id, assigned_by, active)
      values (new.id, chosen_therapist, new.caregiver_id, true);
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_auto_assign_therapist on public.children;
create trigger trg_auto_assign_therapist
  after insert or update of needs_therapist on public.children
  for each row execute function public.auto_assign_therapist();

-- ==============================================================================
-- 5. SESSIONS
-- One row per "Record Session" -> "Art Canvas" -> "End Session" flow.
-- ==============================================================================
create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  caregiver_id uuid not null references public.caregivers(id) on delete cascade,
  child_id uuid not null references public.children(id) on delete cascade,

  video_duration_seconds integer not null default 10,
  art_duration_seconds integer not null default 15,
  status text not null default 'recording'
    check (status in ('recording', 'processing', 'completed', 'failed')),

  started_at timestamptz not null default now(),
  ended_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists idx_sessions_caregiver on public.sessions(caregiver_id);
create index if not exists idx_sessions_child on public.sessions(child_id);
create index if not exists idx_sessions_status on public.sessions(status);

-- ==============================================================================
-- 6. FACE VIDEO RECORDING (captured while the child draws)
-- storage_path convention: {caregiver_id}/{session_id}.mp4 in the
-- 'session-videos' bucket (created near the bottom of this file).
-- ==============================================================================
create table if not exists public.session_video_recordings (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null unique references public.sessions(id) on delete cascade,
  storage_path text not null,
  duration_seconds numeric(5, 2),
  fps numeric(4, 2) default 30.0,
  resolution text default '640x480',
  processed boolean not null default false,
  uploaded_at timestamptz not null default now()
);

-- ==============================================================================
-- 7. SCRIBBLE / DRAWING EVENT STREAM
-- Insert in a single batch from the art canvas at "End Session", not one
-- row per pointer event.
-- ==============================================================================
create table if not exists public.session_touch_events (
  id bigint generated by default as identity primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  timestamp_ms bigint not null,
  stroke_id integer not null default 0,
  event_type text not null check (event_type in ('down', 'move', 'up')),
  x numeric(7, 2) not null,
  y numeric(7, 2) not null,
  pressure numeric(4, 3) default 0.5,
  color text,
  brush_size numeric(4, 2),
  recorded_at timestamptz not null default now()
);

create index if not exists idx_touch_session_time on public.session_touch_events(session_id, timestamp_ms);

-- Rendered PNG snapshot of the finished artwork (for display + as a GAN /
-- vision-model input alongside the raw stroke stream).
create table if not exists public.session_artwork (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null unique references public.sessions(id) on delete cascade,
  storage_path text not null,   -- 'session-artwork' bucket
  stroke_count integer not null default 0,
  created_at timestamptz not null default now()
);

-- ==============================================================================
-- 8. MODEL PREDICTIONS
-- Two input modalities in this version (face video, scribble stream) -> two
-- prediction tables. Written by the backend inference service using the
-- service-role key (bypasses RLS). Additional modalities (gaze, wearables)
-- can be added later by following this same "predictions_<model>" pattern.
-- ==============================================================================

-- 8a. Facial emotion + VAD, sampled every ~0.5s from the 10s face video.
create table if not exists public.predictions_facial_emotion (
  id bigint generated by default as identity primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  timestamp_sec numeric(6, 2) not null,
  dominant_emotion text not null
    check (dominant_emotion in ('Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral')),
  confidence numeric(4, 3) not null,
  valence numeric(4, 3) not null,
  arousal numeric(4, 3) not null,
  dominance numeric(4, 3) not null,
  emotion_probabilities jsonb not null,
  predicted_at timestamptz not null default now()
);

create index if not exists idx_pred_face_session on public.predictions_facial_emotion(session_id, timestamp_sec);

-- 8b. Touch/scribble rhythm + VAD, windowed over the 15s art canvas capture.
create table if not exists public.predictions_touch_rhythm (
  id bigint generated by default as identity primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  window_start_sec numeric(6, 2) not null,
  window_end_sec numeric(6, 2) not null,
  behavioral_state text not null check (behavioral_state in (
    'Calm_Regulated', 'Engaged_Focused', 'Excited_Stimming',
    'Anxious_Overstimulated', 'Distressed_PreMeltdown', 'Withdrawn_FlatAffect'
  )),
  state_probabilities jsonb not null,
  touch_valence numeric(4, 3) not null,
  touch_arousal numeric(4, 3) not null,
  touch_dominance numeric(4, 3) not null,
  stroke_speed_avg numeric(6, 2),
  pressure_avg numeric(4, 3),
  stereotypy_detected boolean not null default false,
  predicted_at timestamptz not null default now()
);

create index if not exists idx_pred_touch_session on public.predictions_touch_rhythm(session_id, window_start_sec);

-- ==============================================================================
-- 9. SESSION REPORTS (LLM-fused output)
-- routed_to is set automatically at insert time from children.needs_therapist,
-- so RLS can decide who is allowed to read the row without re-deriving it
-- from a join every time.
-- ==============================================================================
create table if not exists public.session_reports (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null unique references public.sessions(id) on delete cascade,
  caregiver_id uuid not null references public.caregivers(id) on delete cascade,
  child_id uuid not null references public.children(id) on delete cascade,

  overall_state text not null,
  regulation_score_pct integer check (regulation_score_pct between 0 and 100),
  valence numeric(4, 3) not null,
  arousal numeric(4, 3) not null,
  dominance numeric(4, 3) not null,

  insights text[] not null default '{}',
  recommendations text[] not null default '{}',
  distress_alerts text[] not null default '{}',
  report_markdown text not null,
  raw_report_json jsonb not null,

  routed_to text not null check (routed_to in ('caregiver', 'therapist')),

  reviewed_by_therapist_id uuid references public.therapists(id),
  therapist_notes text,
  reviewed_at timestamptz,

  created_at timestamptz not null default now()
);

create index if not exists idx_reports_caregiver on public.session_reports(caregiver_id);
create index if not exists idx_reports_child on public.session_reports(child_id);
create index if not exists idx_reports_routed_to on public.session_reports(routed_to);

create or replace function public.set_report_routing()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  select case when needs_therapist then 'therapist' else 'caregiver' end
  into new.routed_to
  from public.children where id = new.child_id;
  return new;
end;
$$;

drop trigger if exists trg_set_report_routing on public.session_reports;
create trigger trg_set_report_routing
  before insert on public.session_reports
  for each row execute function public.set_report_routing();

-- ==============================================================================
-- ROW LEVEL SECURITY
-- ==============================================================================

alter table public.caregivers enable row level security;
alter table public.therapists enable row level security;
alter table public.children enable row level security;
alter table public.child_therapist_assignments enable row level security;
alter table public.sessions enable row level security;
alter table public.session_video_recordings enable row level security;
alter table public.session_touch_events enable row level security;
alter table public.session_artwork enable row level security;
alter table public.predictions_facial_emotion enable row level security;
alter table public.predictions_touch_rhythm enable row level security;
alter table public.session_reports enable row level security;

drop policy if exists "Caregivers manage their own row" on public.caregivers;
drop policy if exists "Therapists manage their own row" on public.therapists;
drop policy if exists "Caregivers manage their children" on public.children;
drop policy if exists "Therapists view assigned children needing therapy" on public.children;
drop policy if exists "Caregivers manage assignments for their children" on public.child_therapist_assignments;
drop policy if exists "Therapists view their own assignments" on public.child_therapist_assignments;
drop policy if exists "Caregivers manage their sessions" on public.sessions;
drop policy if exists "Therapists view sessions for assigned children needing therapy" on public.sessions;

create policy "Caregivers manage their own row"
  on public.caregivers for all to authenticated
  using (auth.uid() = id) with check (auth.uid() = id);

create policy "Therapists manage their own row"
  on public.therapists for all to authenticated
  using (auth.uid() = id) with check (auth.uid() = id);

create policy "Caregivers manage their children"
  on public.children for all to authenticated
  using (auth.uid() = caregiver_id) with check (auth.uid() = caregiver_id);

create policy "Therapists view assigned children needing therapy"
  on public.children for select to authenticated
  using (
    needs_therapist and id in (
      select child_id from public.child_therapist_assignments
      where therapist_id = auth.uid() and active
    )
  );

create policy "Caregivers manage assignments for their children"
  on public.child_therapist_assignments for all to authenticated
  using (assigned_by = auth.uid())
  with check (assigned_by = auth.uid());

create policy "Therapists view their own assignments"
  on public.child_therapist_assignments for select to authenticated
  using (auth.uid() = therapist_id);

create policy "Caregivers manage their sessions"
  on public.sessions for all to authenticated
  using (auth.uid() = caregiver_id) with check (auth.uid() = caregiver_id);

create policy "Therapists view sessions for assigned children needing therapy"
  on public.sessions for select to authenticated
  using (
    child_id in (
      select child_id from public.child_therapist_assignments
      where therapist_id = auth.uid() and active
    )
  );

-- Raw inputs: scoped through the owning session's caregiver only. Therapists
-- see the fused report, not the raw video/stroke stream, keeping the least
-- amount of sensitive raw footage in the therapist's view.
create policy "Caregiver access to video recordings via session"
  on public.session_video_recordings for all to authenticated
  using (session_id in (select id from public.sessions where caregiver_id = auth.uid()))
  with check (session_id in (select id from public.sessions where caregiver_id = auth.uid()));

create policy "Caregiver access to touch events via session"
  on public.session_touch_events for all to authenticated
  using (session_id in (select id from public.sessions where caregiver_id = auth.uid()))
  with check (session_id in (select id from public.sessions where caregiver_id = auth.uid()));

create policy "Caregiver access to artwork via session"
  on public.session_artwork for all to authenticated
  using (session_id in (select id from public.sessions where caregiver_id = auth.uid()))
  with check (session_id in (select id from public.sessions where caregiver_id = auth.uid()));

-- Predictions: read-only for the owning caregiver, or the assigned therapist
-- when the child needs therapy. Writes come from the backend inference
-- service using the service-role key, which bypasses RLS entirely.
create policy "Caregiver read predictions_facial_emotion"
  on public.predictions_facial_emotion for select to authenticated
  using (session_id in (select id from public.sessions where caregiver_id = auth.uid()));

create policy "Therapist read predictions_facial_emotion"
  on public.predictions_facial_emotion for select to authenticated
  using (session_id in (
    select s.id from public.sessions s
    join public.child_therapist_assignments cta on cta.child_id = s.child_id
    join public.children c on c.id = s.child_id
    where cta.therapist_id = auth.uid() and cta.active and c.needs_therapist
  ));

create policy "Caregiver read predictions_touch_rhythm"
  on public.predictions_touch_rhythm for select to authenticated
  using (session_id in (select id from public.sessions where caregiver_id = auth.uid()));

create policy "Therapist read predictions_touch_rhythm"
  on public.predictions_touch_rhythm for select to authenticated
  using (session_id in (
    select s.id from public.sessions s
    join public.child_therapist_assignments cta on cta.child_id = s.child_id
    join public.children c on c.id = s.child_id
    where cta.therapist_id = auth.uid() and cta.active and c.needs_therapist
  ));

-- Reports: caregiver always sees their own; therapist only sees the ones
-- routed to them (routed_to = 'therapist'), which is exactly the set for
-- children with needs_therapist = true and an active assignment.
create policy "Caregivers view their session reports"
  on public.session_reports for select to authenticated
  using (auth.uid() = caregiver_id);

create policy "Therapists view reports routed to them"
  on public.session_reports for select to authenticated
  using (
    routed_to = 'therapist'
    and child_id in (
      select cta.child_id from public.child_therapist_assignments cta
      where cta.therapist_id = auth.uid() and cta.active
    )
  );

create policy "Therapists can add their review notes"
  on public.session_reports for update to authenticated
  using (
    routed_to = 'therapist'
    and child_id in (
      select cta.child_id from public.child_therapist_assignments cta
      where cta.therapist_id = auth.uid() and cta.active
    )
  )
  with check (auth.uid() = reviewed_by_therapist_id);

-- ==============================================================================
-- AUTO-CREATE CAREGIVER / THERAPIST ROWS ON SIGNUP
-- ==============================================================================

create or replace function public.handle_new_caregiver()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if coalesce(new.raw_user_meta_data->>'role', 'caregiver') = 'caregiver' then
    insert into public.caregivers (id, name, age, gender, relation_with_child, email)
    values (
      new.id,
      coalesce(new.raw_user_meta_data->>'name', new.email),
      nullif(new.raw_user_meta_data->>'age', '')::integer,
      new.raw_user_meta_data->>'gender',
      coalesce(new.raw_user_meta_data->>'relation_with_child', 'Parent'),
      new.email
    );
  end if;
  return new;
end;
$$;

drop trigger if exists on_auth_caregiver_created on auth.users;
create trigger on_auth_caregiver_created
  after insert on auth.users
  for each row execute function public.handle_new_caregiver();

create or replace function public.handle_new_therapist()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if new.raw_user_meta_data->>'role' = 'therapist' then
    insert into public.therapists (
      id, name, age, gender, qualification, license_number,
      contact_email, username, login_email
    )
    values (
      new.id,
      new.raw_user_meta_data->>'name',
      nullif(new.raw_user_meta_data->>'age', '')::integer,
      new.raw_user_meta_data->>'gender',
      new.raw_user_meta_data->>'qualification',
      new.raw_user_meta_data->>'license_number',
      new.raw_user_meta_data->>'contact_email',
      new.raw_user_meta_data->>'username',
      new.email
    );
  end if;
  return new;
end;
$$;

drop trigger if exists on_auth_therapist_created on auth.users;
create trigger on_auth_therapist_created
  after insert on auth.users
  for each row execute function public.handle_new_therapist();

-- ==============================================================================
-- STORAGE BUCKETS
-- Upload path convention: {caregiver_id}/{session_id}.mp4 / .png
-- ==============================================================================
insert into storage.buckets (id, name, public)
values ('session-videos', 'session-videos', false)
on conflict (id) do nothing;

insert into storage.buckets (id, name, public)
values ('session-artwork', 'session-artwork', false)
on conflict (id) do nothing;

create policy "Caregivers upload their own session videos"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'session-videos' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Caregivers read their own session videos"
  on storage.objects for select to authenticated
  using (bucket_id = 'session-videos' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Caregivers upload their own artwork"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'session-artwork' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Caregivers read their own artwork"
  on storage.objects for select to authenticated
  using (bucket_id = 'session-artwork' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Therapists read artwork for assigned children needing therapy"
  on storage.objects for select to authenticated
  using (
    bucket_id = 'session-artwork'
    and (storage.foldername(name))[1] in (
      select s.caregiver_id::text from public.sessions s
      join public.child_therapist_assignments cta on cta.child_id = s.child_id
      join public.children c on c.id = s.child_id
      where cta.therapist_id = auth.uid() and cta.active and c.needs_therapist
    )
  );