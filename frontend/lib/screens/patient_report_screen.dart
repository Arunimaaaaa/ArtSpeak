import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../models/child_profile.dart';
import '../models/session_record.dart';
import '../theme/app_theme.dart';

class PatientReportScreen extends StatelessWidget {
  final ChildProfile profile;
  const PatientReportScreen({super.key, required this.profile});

  @override
  Widget build(BuildContext context) {
    final sessions = profile.sessions;
    return Scaffold(
      backgroundColor: AppColors.bgTop,
      appBar: AppBar(
        title: Text('${profile.name}\'s Report'),
      ),
      body: sessions.isEmpty
          ? Center(child: Text('No sessions recorded yet.', style: GoogleFonts.nunito(color: AppColors.textMuted)))
          : ListView(
              padding: const EdgeInsets.all(20),
              children: [
                _ProfileHeader(profile: profile),
                const SizedBox(height: 24),
                Text('Session Engagement',
                    style: GoogleFonts.fredoka(fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                const SizedBox(height: 14),
                _EngagementChart(sessions: sessions),
                const SizedBox(height: 28),
                Text('Mood Distribution',
                    style: GoogleFonts.fredoka(fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                const SizedBox(height: 14),
                _MoodDistribution(sessions: sessions),
                const SizedBox(height: 28),
                Text('Generated Caregiver / Clinical Notes',
                    style: GoogleFonts.fredoka(fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                const SizedBox(height: 4),
                Text('Auto-written by the LLM Translation Layer after each session.',
                    style: GoogleFonts.nunito(fontSize: 12.5, color: AppColors.textMuted)),
                const SizedBox(height: 14),
                ...sessions.reversed.map((s) => _ReportCard(session: s)),
              ],
            ),
    );
  }
}

class _ProfileHeader extends StatelessWidget {
  final ChildProfile profile;
  const _ProfileHeader({required this.profile});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(gradient: AppGradients.primaryCard, borderRadius: BorderRadius.circular(20)),
      child: Row(
        children: [
          CircleAvatar(
            radius: 26,
            backgroundColor: Colors.white.withValues(alpha: 0.2),
            child: Text(profile.name[0], style: GoogleFonts.fredoka(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 20)),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(profile.name, style: GoogleFonts.fredoka(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w700)),
                const SizedBox(height: 2),
                Text('Age ${profile.age} · ${profile.gender.label} · ${profile.condition ?? '—'}',
                    style: GoogleFonts.nunito(color: Colors.white70, fontSize: 12.5)),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text('${profile.sessionCount}', style: GoogleFonts.fredoka(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w700)),
              Text('sessions', style: GoogleFonts.nunito(color: Colors.white70, fontSize: 11)),
            ],
          ),
        ],
      ),
    );
  }
}

class _EngagementChart extends StatelessWidget {
  final List<SessionRecord> sessions;
  const _EngagementChart({required this.sessions});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      padding: const EdgeInsets.fromLTRB(12, 20, 20, 12),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)),
      child: BarChart(
        BarChartData(
          maxY: 40,
          gridData: const FlGridData(show: true, drawVerticalLine: false),
          borderData: FlBorderData(show: false),
          titlesData: FlTitlesData(
            topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            leftTitles: AxisTitles(
                sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              getTitlesWidget: (v, meta) => Text('${v.toInt()}s', style: GoogleFonts.nunito(fontSize: 10, color: AppColors.textMuted)),
            )),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (v, meta) {
                  final i = v.toInt();
                  if (i < 0 || i >= sessions.length) return const SizedBox.shrink();
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text('S${i + 1}', style: GoogleFonts.nunito(fontSize: 10, color: AppColors.textMuted)),
                  );
                },
              ),
            ),
          ),
          barGroups: List.generate(sessions.length, (i) {
            final total = (sessions[i].recordingSeconds + sessions[i].canvasSeconds).toDouble();
            return BarChartGroupData(x: i, barRods: [
              BarChartRodData(toY: total, color: AppColors.primaryLight, width: 14, borderRadius: BorderRadius.circular(4)),
            ]);
          }),
        ),
      ),
    );
  }
}

class _MoodDistribution extends StatelessWidget {
  final List<SessionRecord> sessions;
  const _MoodDistribution({required this.sessions});

  @override
  Widget build(BuildContext context) {
    final counts = <SessionMood, int>{};
    for (final s in sessions) {
      counts[s.mood] = (counts[s.mood] ?? 0) + 1;
    }
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(18)),
      child: Column(
        children: counts.entries.map((e) {
          final ratio = e.value / sessions.length;
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Row(
              children: [
                SizedBox(width: 100, child: Text(e.key.label, style: GoogleFonts.nunito(fontSize: 13))),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: ratio,
                      minHeight: 10,
                      backgroundColor: AppColors.accentSoft,
                      valueColor: AlwaysStoppedAnimation(e.key.color),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Text('${e.value}', style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }
}

class _ReportCard extends StatelessWidget {
  final SessionRecord session;
  const _ReportCard({required this.session});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: session.distressAlert ? Border.all(color: AppColors.alertRed, width: 1.4) : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(DateFormat('MMM d, yyyy').format(session.date),
                  style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(color: session.mood.color.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(20)),
                child: Text(session.mood.label, style: GoogleFonts.fredoka(color: session.mood.color, fontWeight: FontWeight.w700, fontSize: 12)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(session.caregiverSummary, style: GoogleFonts.nunito(height: 1.4, fontSize: 13.5)),
          const SizedBox(height: 10),
          Text(
            'V ${session.valence}  ·  A ${session.arousal}  ·  D ${session.dominance}  ·  ${session.strokeCount} strokes',
            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 11.5),
          ),
          if (session.distressAlert) ...[
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.warning_amber_rounded, color: AppColors.alertRed, size: 16),
                const SizedBox(width: 6),
                Text('Distress alert sent to caregiver', style: GoogleFonts.fredoka(color: AppColors.alertRed, fontSize: 12, fontWeight: FontWeight.w600)),
              ],
            ),
          ],
        ],
      ),
    );
  }
}
