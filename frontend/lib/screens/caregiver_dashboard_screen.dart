import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../models/child_profile.dart';
import '../models/session_record.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'session_canvas_screen.dart';
import '../services/art_speak_repository.dart';

class CaregiverDashboardScreen extends StatefulWidget {
  final ChildProfile initialProfile;
  const CaregiverDashboardScreen({super.key, required this.initialProfile});

  @override
  State<CaregiverDashboardScreen> createState() => _CaregiverDashboardScreenState();
}

class _CaregiverDashboardScreenState extends State<CaregiverDashboardScreen> {
  late List<ChildProfile> _profiles;
  int _activeIndex = 0;

  ChildProfile get _active => _profiles[_activeIndex];

  @override
  void initState() {
    super.initState();
    _profiles = [widget.initialProfile];
  }

  Future<void> _startSessionFlow() async {
    final childId = _active.id;
    if (childId == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('This child is not synced yet.')));
      return;
    }
    try {
      final sessionId = await artSpeakRepository.createSession(childId: childId, durationSeconds: kSessionSeconds);
      final result = await Navigator.of(context).push<SessionRecord>(
        MaterialPageRoute(builder: (_) => SessionCanvasScreen(childName: _active.name, sessionId: sessionId)),
      );
      if (result != null && mounted) {
        setState(() => _active.sessions.insert(0, result));
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Session uploaded and sent for analysis.')));
      }
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(error.toString())));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgTop,
      appBar: const AppHeader(subtitle: 'Caregiver Dashboard'),
      bottomNavigationBar: AppFooter(onLogout: () => Navigator.of(context).popUntil((r) => r.isFirst)),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(gradient: AppGradients.primaryCard, borderRadius: BorderRadius.circular(22)),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('ACTIVE CHILD PROFILE',
                            style: GoogleFonts.nunito(color: Colors.white70, fontSize: 12, letterSpacing: 1)),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(_active.name,
                                style: GoogleFonts.fredoka(color: Colors.white, fontSize: 30, fontWeight: FontWeight.w700)),
                            Container(
                              width: 52,
                              height: 52,
                              decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
                              child: const Icon(Icons.palette_rounded, color: Colors.white, size: 26),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Age ${_active.age} · ${_active.gender.label} · ${_active.condition ?? 'Not specified'}',
                          style: GoogleFonts.nunito(color: Colors.white.withOpacity(0.85), fontSize: 14),
                        ),
                        if (_active.hobbies != null || _active.knownTriggers != null || _active.calmingActivities != null) ...[
                          const SizedBox(height: 12),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              if (_active.hobbies != null) _InfoPill(icon: Icons.celebration_rounded, text: _active.hobbies!),
                              if (_active.knownTriggers != null) _InfoPill(icon: Icons.warning_amber_rounded, text: _active.knownTriggers!),
                              if (_active.calmingActivities != null) _InfoPill(icon: Icons.spa_rounded, text: _active.calmingActivities!),
                            ],
                          ),
                        ],
                        const SizedBox(height: 20),
                        Row(
                          children: [
                            StatChip(value: '${_active.sessionCount}', label: 'Sessions'),
                            StatChip(value: '${_active.sessionsThisWeek}', label: 'This Week'),
                            StatChip(value: '${_active.streakDays} days', label: 'Streak'),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  _StartSessionCard(childName: _active.name, onTap: _startSessionFlow),
                  const SizedBox(height: 28),
                  Row(
                    children: [
                      Text('Past Sessions', style: GoogleFonts.fredoka(fontSize: 20, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                      const SizedBox(width: 10),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                        decoration: BoxDecoration(color: AppColors.accentSoft, borderRadius: BorderRadius.circular(20)),
                        child: Text('${_active.sessionCount}', style: GoogleFonts.fredoka(color: AppColors.primary, fontWeight: FontWeight.w700)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  if (_active.sessions.isEmpty)
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)),
                      child: Text('No sessions yet — tap "Start Session" to begin.',
                          style: GoogleFonts.nunito(color: AppColors.textMuted)),
                    )
            else
              ..._active.sessions.map((s) => _SessionTile(session: s, therapistNeeded: _active.therapistNeeded)),
          ],
        ),
      ),
    );
  }
}

class _InfoPill extends StatelessWidget {
  final IconData icon;
  final String text;
  const _InfoPill({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(color: Colors.white.withOpacity(0.16), borderRadius: BorderRadius.circular(14)),
      constraints: const BoxConstraints(maxWidth: 260),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: Colors.white, size: 14),
          const SizedBox(width: 6),
          Flexible(
            child: Text(text,
                overflow: TextOverflow.ellipsis,
                maxLines: 1,
                style: GoogleFonts.nunito(color: Colors.white, fontSize: 12)),
          ),
        ],
      ),
    );
  }
}

class _StartSessionCard extends StatelessWidget {
  final String childName;
  final VoidCallback onTap;
  const _StartSessionCard({required this.childName, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.primary, width: 1.6),
          ),
          child: Row(
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(color: AppColors.accentSoft, borderRadius: BorderRadius.circular(14)),
                child: const Icon(Icons.play_arrow_rounded, color: AppColors.primary, size: 30),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Start Session', style: GoogleFonts.fredoka(fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                    const SizedBox(height: 4),
                    Text('A 20-second guided drawing session for $childName',
                        style: GoogleFonts.nunito(fontSize: 12.5, color: AppColors.textMuted, height: 1.3)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded, color: AppColors.primary),
            ],
          ),
        ),
      ),
    );
  }
}

class _SessionTile extends StatelessWidget {
  final SessionRecord session;
  final bool therapistNeeded;
  const _SessionTile({required this.session, required this.therapistNeeded});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(DateFormat('MMM d, yyyy').format(session.date),
                    style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, color: AppColors.primaryDark, fontSize: 15)),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Text('${session.recordingSeconds + session.canvasSeconds}s total',
                        style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 13)),
                    if (!therapistNeeded) ...[
                      const SizedBox(width: 8),
                      Text('· ${session.mood.label}',
                          style: GoogleFonts.nunito(color: session.mood.color, fontSize: 13, fontWeight: FontWeight.w700)),
                    ],
                  ],
                ),
              ],
            ),
          ),
          if (session.distressAlert && !therapistNeeded)
            const Padding(
              padding: EdgeInsets.only(right: 8),
              child: Icon(Icons.warning_amber_rounded, color: AppColors.alertRed),
            ),
          if (therapistNeeded)
            const Padding(
              padding: EdgeInsets.only(right: 8),
              child: Icon(Icons.lock_outline_rounded, color: AppColors.textMuted),
            ),
          TextButton(
            style: TextButton.styleFrom(backgroundColor: AppColors.accentSoft, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10))),
            onPressed: () {
              showModalBottomSheet(
                context: context,
                shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
                builder: (_) => Padding(
                  padding: const EdgeInsets.all(24),
                  child: therapistNeeded ? _lockedSummary() : _visibleSummary(),
                ),
              );
            },
            child: Text('View', style: GoogleFonts.fredoka(color: AppColors.primary, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }

  Widget _visibleSummary() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Session Summary', style: GoogleFonts.fredoka(fontSize: 18, fontWeight: FontWeight.w700)),
        const SizedBox(height: 10),
        Text(session.caregiverSummary, style: GoogleFonts.nunito(height: 1.4)),
        const SizedBox(height: 16),
        Text('Valence ${session.valence}  ·  Arousal ${session.arousal}  ·  Dominance ${session.dominance}',
            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 12)),
      ],
    );
  }

  Widget _lockedSummary() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.lock_outline_rounded, color: AppColors.textMuted),
            const SizedBox(width: 8),
            Text('Summary Reserved for Therapist', style: GoogleFonts.fredoka(fontSize: 17, fontWeight: FontWeight.w700)),
          ],
        ),
        const SizedBox(height: 10),
        Text(
          "You indicated this child's profile needs a therapist, so this session's detailed summary is shared "
          'directly with the assigned therapist rather than shown here. Your therapist will follow up with you '
          'with any guidance.',
          style: GoogleFonts.nunito(height: 1.4, color: AppColors.textMuted),
        ),
      ],
    );
  }
}
