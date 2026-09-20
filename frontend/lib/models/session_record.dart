import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Coarse mood label surfaced to caregivers — derived, in the real system,
/// from the Behaviour Analysis Layer's VAD (Valence/Arousal/Dominance) vector.
enum SessionMood { calm, energetic, overloaded, neutral, distressed }

extension SessionMoodMeta on SessionMood {
  String get label {
    switch (this) {
      case SessionMood.calm:
        return 'Calm';
      case SessionMood.energetic:
        return 'Energetic';
      case SessionMood.overloaded:
        return 'Overload risk';
      case SessionMood.neutral:
        return 'Neutral';
      case SessionMood.distressed:
        return 'Distress flagged';
    }
  }

  Color get color {
    switch (this) {
      case SessionMood.calm:
        return AppColors.primary;
      case SessionMood.energetic:
        return AppColors.warnAmber;
      case SessionMood.overloaded:
        return AppColors.alertRed;
      case SessionMood.neutral:
        return AppColors.primaryLight;
      case SessionMood.distressed:
        return AppColors.alertRed;
    }
  }
}

/// One completed ArtSpeak session: the behavioural recording + the art
/// canvas output + the (mock) VAD vector and LLM-generated caregiver blurb
/// that the therapist dashboard visualises.
class SessionRecord {
  final String id;
  final DateTime date;
  final int recordingSeconds;
  final int canvasSeconds;
  final SessionMood mood;
  final double valence; // -1..1
  final double arousal; // 0..1
  final double dominance; // 0..1
  final int strokeCount;
  final String caregiverSummary;
  final bool distressAlert;
  final String? artworkBase64;

  SessionRecord({
    required this.id,
    required this.date,
    required this.recordingSeconds,
    required this.canvasSeconds,
    required this.mood,
    required this.valence,
    required this.arousal,
    required this.dominance,
    required this.strokeCount,
    required this.caregiverSummary,
    this.distressAlert = false,
    this.artworkBase64,
  });

  factory SessionRecord.processing({required String id, required int seconds, required int strokeCount}) {
    return SessionRecord(
      id: id,
      date: DateTime.now(),
      recordingSeconds: seconds,
      canvasSeconds: seconds,
      mood: SessionMood.neutral,
      valence: 0,
      arousal: 0,
      dominance: 0,
      strokeCount: strokeCount,
      caregiverSummary: 'Your session is being analysed across facial, gaze, touch, and physiological signals.',
      artworkBase64: null,
    );
  }

  factory SessionRecord.fromReport(Map<String, dynamic> report) {
    final state = report['overall_state'] as String? ?? 'Calm_Regulated';
    final mood = state.contains('Distressed')
        ? SessionMood.distressed
        : state.contains('Anxious')
            ? SessionMood.overloaded
            : state.contains('Excited')
                ? SessionMood.energetic
                : state.contains('Calm')
                    ? SessionMood.calm
                    : SessionMood.neutral;
    final affect = (report['raw_report_json']?['affect_vector'] as Map?)?.cast<String, dynamic>() ?? const {};
    return SessionRecord(
      id: report['session_id'] as String,
      date: DateTime.tryParse(report['created_at'] as String? ?? '') ?? DateTime.now(),
      recordingSeconds: 0,
      canvasSeconds: 0,
      mood: mood,
      valence: (affect['valence'] as num?)?.toDouble() ?? 0,
      arousal: (affect['arousal'] as num?)?.toDouble() ?? 0,
      dominance: (affect['dominance'] as num?)?.toDouble() ?? 0,
      strokeCount: 0,
      caregiverSummary: (report['insights'] as List?)?.cast<String>().firstOrNull ?? 'Report available.',
      distressAlert: mood == SessionMood.distressed || mood == SessionMood.overloaded,
      artworkBase64: report['artwork_base64'] as String?,
    );
  }

  /// Builds a plausible mock record for a just-completed session, standing
  /// in for the real Behaviour Analysis + LLM Translation layers.
  factory SessionRecord.mockFromSession({
    required int recordingSeconds,
    required int canvasSeconds,
    required int strokeCount,
  }) {
    const mood = SessionMood.neutral;
    const valence = 0.0;
    const arousal = 0.0;
    const dominance = 0.0;
    final summaries = {
      SessionMood.calm: 'Session showed settled, rhythmic strokes with steady pressure — consistent with a regulated, calm state.',
      SessionMood.energetic: 'Fast, broad strokes with rising tempo. Positive engagement, slightly elevated arousal.',
      SessionMood.overloaded: 'Stroke pressure and tap rate spiked mid-session. Recommend a shorter session and a sensory break next time.',
      SessionMood.neutral: 'Interaction pattern close to baseline, no notable deviation this session.',
      SessionMood.distressed: 'Distress markers detected. Please check in with your child and consider contacting their therapist.',
    };
    return SessionRecord(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      date: DateTime.now(),
      recordingSeconds: recordingSeconds,
      canvasSeconds: canvasSeconds,
      mood: mood,
      valence: valence,
      arousal: arousal,
      dominance: dominance,
      strokeCount: strokeCount,
      caregiverSummary: summaries[mood]!,
      distressAlert: false,
      artworkBase64: null,
    );
  }
}
