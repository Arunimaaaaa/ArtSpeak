import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/child_profile.dart';
import '../models/session_record.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'patient_report_screen.dart';
import '../services/art_speak_repository.dart';

class TherapistDashboardScreen extends StatefulWidget {
  const TherapistDashboardScreen({super.key});

  @override
  State<TherapistDashboardScreen> createState() => _TherapistDashboardScreenState();
}

class _TherapistDashboardScreenState extends State<TherapistDashboardScreen> {
  List<ChildProfile> roster = [];
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _loadRoster();
  }

  Future<void> _loadRoster() async {
    try {
      final loaded = await artSpeakRepository.loadTherapistRoster();
      if (mounted) setState(() { roster = loaded; loading = false; });
    } catch (error) {
      if (mounted) {
        setState(() => loading = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(error.toString())));
      }
    }
  }

  int get _totalSessions => roster.fold(0, (s, c) => s + c.sessionCount);
  int get _alertCount => roster.fold(0, (s, c) => s + c.sessions.where((r) => r.distressAlert).length);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgTop,
      appBar: const AppHeader(subtitle: 'Therapist Dashboard'),
      bottomNavigationBar: AppFooter(onLogout: () => Navigator.of(context).popUntil((r) => r.isFirst)),
      body: loading ? const Center(child: CircularProgressIndicator()) : ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Row(
            children: [
              Expanded(child: _SummaryCard(icon: Icons.groups_rounded, value: '${roster.length}', label: 'Active Patients')),
              const SizedBox(width: 14),
              Expanded(child: _SummaryCard(icon: Icons.insert_chart_rounded, value: '$_totalSessions', label: 'Sessions Logged')),
              const SizedBox(width: 14),
              Expanded(
                child: _SummaryCard(
                  icon: Icons.warning_amber_rounded,
                  value: '$_alertCount',
                  label: 'Distress Alerts',
                  alert: _alertCount > 0,
                ),
              ),
            ],
          ),
          const SizedBox(height: 28),
          Text('Patient Reports', style: GoogleFonts.fredoka(fontSize: 20, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
          const SizedBox(height: 6),
          Text('Generated automatically by the Behaviour Analysis and LLM Translation layers after each session.',
              style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 13)),
          const SizedBox(height: 16),
          ...roster.map((c) => _PatientTile(
                profile: c,
                onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => PatientReportScreen(profile: c))),
              )),
        ],
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  final IconData icon;
  final String value;
  final String label;
  final bool alert;
  const _SummaryCard({required this.icon, required this.value, required this.label, this.alert = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 14),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        children: [
          Icon(icon, color: alert ? AppColors.alertRed : AppColors.primary, size: 26),
          const SizedBox(height: 8),
          Text(value, style: GoogleFonts.fredoka(fontSize: 22, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
          const SizedBox(height: 2),
          Text(label, textAlign: TextAlign.center, style: GoogleFonts.nunito(fontSize: 11.5, color: AppColors.textMuted)),
        ],
      ),
    );
  }
}

class _PatientTile extends StatelessWidget {
  final ChildProfile profile;
  final VoidCallback onTap;
  const _PatientTile({required this.profile, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final hasAlert = profile.sessions.any((s) => s.distressAlert);
    final latest = profile.sessions.isNotEmpty ? profile.sessions.last : null;
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              CircleAvatar(
                radius: 24,
                backgroundColor: AppColors.accentSoft,
                child: Text(profile.name[0], style: GoogleFonts.fredoka(color: AppColors.primary, fontWeight: FontWeight.w700, fontSize: 18)),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(profile.name, style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, fontSize: 16, color: AppColors.primaryDark)),
                    const SizedBox(height: 2),
                    Text('Age ${profile.age} · ${profile.gender.label} · ${profile.condition ?? '—'}',
                        style: GoogleFonts.nunito(fontSize: 12.5, color: AppColors.textMuted)),
                    if (latest != null) ...[
                      const SizedBox(height: 4),
                      Text('Last session: ${latest.mood.label}',
                          style: GoogleFonts.nunito(fontSize: 12.5, color: latest.mood.color, fontWeight: FontWeight.w700)),
                    ],
                  ],
                ),
              ),
              if (hasAlert)
                Container(
                  margin: const EdgeInsets.only(right: 10),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(color: AppColors.alertRed.withOpacity(0.1), borderRadius: BorderRadius.circular(20)),
                  child: Text('Alert', style: GoogleFonts.fredoka(color: AppColors.alertRed, fontWeight: FontWeight.w700, fontSize: 12)),
                ),
              Text('${profile.sessionCount} sessions', style: GoogleFonts.nunito(fontSize: 12, color: AppColors.textMuted)),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right_rounded, color: AppColors.textMuted),
            ],
          ),
        ),
      ),
    );
  }
}
