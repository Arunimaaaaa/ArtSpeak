## Connected Supabase and ML flow

1. Run `supabase_schema.sql` in the Supabase SQL editor or with `supabase db push`.
2. Install the worker dependencies from the repository root with `pip install -r requirements.txt`.
3. Start the worker from the repository root:

  `SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... uvicorn services.inference_api:app --host 0.0.0.0 --port 8000`

4. Start Flutter with the public client key and worker URL:

  `flutter run --dart-define=SUPABASE_URL=... --dart-define=SUPABASE_ANON_KEY=... --dart-define=INFERENCE_API_URL=http://localhost:8000`

The service-role key belongs only on the worker host. Never pass it to Flutter or commit it.

### VS Code setup

Copy `config/frontend.env.example` values into your shell environment, and copy
`config/inference.env.example` to `.env.inference` at the repository root.
Install the Python dependencies with `pip install -r requirements.txt`, then
run the `ArtSpeak Inference API` and `ArtSpeak Flutter (Supabase)` launch
profiles from VS Code. The Flutter launch profile uses Chrome on port 8080.

In Supabase Dashboard > Authentication > Providers, enable Email. For local
testing, either disable email confirmation or confirm the signup email before
logging in.

### Existing-project migration

After updating the schema file, run it again in the Supabase SQL Editor. The
`profiles_id_fkey` migration is required for child onboarding: children are
stored as caregiver-owned profiles and do not have an `auth.users` account.
# ArtSpeak — Flutter Frontend Prototype

Frontend-only Flutter implementation matching the reference screens you shared,
built against the ArtSpeak synopsis (4-layer pipeline: Non-Verbal Input →
Behaviour Analysis → Sensory-Calibrated Generation → LLM Translation).

## Setup

```bash
flutter pub get
flutter run
```

For the connected Supabase app, keep local credentials in the repository-root
`.env.frontend` file and start the app with:

```bash
./frontend/run_with_env.sh
```

The credentials file is ignored by Git. Never put the Supabase service-role key
in it; Flutter must use only the publishable/anon key.

Add the platform permission snippets before running on a device:
- `android/app/src/main/AndroidManifest.snippet.xml` → merge into your generated
  `AndroidManifest.xml` (camera permission).
- `ios/Runner/Info.plist.snippet.xml` → merge into `Info.plist` (camera usage string).

If `flutter create .` hasn't been run yet in this folder, run it first so the
platform folders (`android/`, `ios/`) get fully scaffolded — the two snippet
files above are provided as a checklist of what to add to those generated files.

## Screen flow

```
WelcomeScreen
 ├─ Caregiver Login → Child Details (+ Gender) → Caregiver Dashboard
 │                                                    └─ Record Session
 │                                                        └─ SessionRecordingScreen (LIVE CAMERA, 10s)
 │                                                            └─ auto-advances → ArtCanvasScreen (15s, then "End Session")
 │                                                                └─ pops a SessionRecord back to the Dashboard
 └─ Therapist Login → Therapist Dashboard (patient roster)
                          └─ Patient Report (VAD trend chart, engagement chart,
                                              mood distribution, LLM-style session notes)
```

## Connected workflow

- Caregiver signup/login and child onboarding use Supabase Auth and the
  `create_child_profile` RPC.
- Session metadata, timestamped drawing events, and private video uploads are
  stored in Supabase.
- `services/inference_api.py` verifies the caller, runs the four existing ML
  models, writes prediction rows and `session_reports`, and updates session
  status.
- Caregiver and therapist dashboards read persisted reports. A report is shown
  as processing until the worker completes it; no random mood values are used
  for new sessions.

## Key files

- `lib/models/child_profile.dart` — child profile incl. **gender** (dropdown:
  Female / Male / Non-binary / Prefer not to say), age, condition, notes.
  Deliberately **no photo field** — matches the synopsis's no-external-PII
  design and your instruction to drop photo upload.
- `lib/screens/session_recording_screen.dart` — live front-camera preview,
  starts a real `CameraController.startVideoRecording()`, counts down 10s,
  stops recording, then pushes the art canvas.
- `lib/screens/art_canvas_screen.dart` — drawing canvas, 15s timer, reveals
  "End Session" button afterwards; on tap, builds a `SessionRecord` and pops
  it back through the recording screen to the dashboard.
- `lib/screens/therapist_dashboard_screen.dart` +
  `lib/screens/patient_report_screen.dart` — roster with distress badges,
  and a per-patient report screen with VAD trend / engagement / mood charts.

## Notes / things to decide before production

- Recorded video clips are currently discarded after `stopVideoRecording()`
  completes (the file path is returned by the plugin but not persisted) —
  wire in `path_provider` + your upload/inference endpoint where noted in
  `_finishRecording()`.
- Sensory-safety: palette in `app_theme.dart` avoids high-saturation reds
  except for alert states, and there are no flashing/animated elements,
  per the synopsis's UI requirements (Ch. 6, Ch. 10).
