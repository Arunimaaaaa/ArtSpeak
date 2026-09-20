-- ==============================================================================
-- ArtSpeak: Hardened fix for "infinite recursion detected in policy for
-- relation children" (42P17)
--
-- The crash happened because children's therapist-select policy queries
-- child_therapist_assignments, and child_therapist_assignments' caregiver
-- policy queried children -- a circular reference. This version breaks the
-- cycle the same way as the quick fix (child_therapist_assignments no
-- longer directly subqueries children under RLS), but restores the
-- ownership check via a SECURITY DEFINER helper function, which runs with
-- elevated privileges and bypasses children's RLS instead of re-entering
-- it -- so the check still happens, just outside the policy-evaluation loop.
--
-- Safe to run any time after the quick fix has already been applied.
-- ==============================================================================

-- 1. Helper function: checks child ownership without going through RLS.
create or replace function public.caregiver_owns_child(p_child_id uuid)
returns boolean
language sql
security definer
set search_path = public
stable
as $$
  select exists (
    select 1 from public.children
    where id = p_child_id and caregiver_id = auth.uid()
  );
$$;

revoke all on function public.caregiver_owns_child(uuid) from public;
grant execute on function public.caregiver_owns_child(uuid) to authenticated;

-- 2. Re-point the assignments policy at the helper instead of assigned_by alone.
drop policy if exists "Caregivers manage assignments for their children" on public.child_therapist_assignments;

create policy "Caregivers manage assignments for their children"
on public.child_therapist_assignments for all to authenticated
using (assigned_by = auth.uid() and public.caregiver_owns_child(child_id))
with check (assigned_by = auth.uid() and public.caregiver_owns_child(child_id));

-- 3. Also re-point auto_assign_therapist(), which inserts assignment rows
-- on the caregiver's behalf via a trigger -- it needs assigned_by set so
-- the ownership check above passes for those rows too. (No-op if your
-- version already sets assigned_by; included here for safety.)
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

-- 4. Optional but recommended: restore the needs_therapist safety check on
-- sessions now that it's safe to do so (children -> child_therapist_assignments
-- is now one-directional, since step 2 no longer reads children back through
-- RLS). This protects against a stale *active* assignment row outliving a
-- caregiver later switching needs_therapist back to false.
drop policy if exists "Therapists view sessions for assigned children needing therapy" on public.sessions;

create policy "Therapists view sessions for assigned children needing therapy"
on public.sessions for select to authenticated
using (
  child_id in (
    select cta.child_id from public.child_therapist_assignments cta
    join public.children c on c.id = cta.child_id
    where cta.therapist_id = auth.uid() and cta.active and c.needs_therapist
  )
);