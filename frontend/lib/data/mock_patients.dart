import 'dart:math';
import '../models/child_profile.dart';
import '../models/session_record.dart';

/// A handful of plausible patients with session history, standing in for
/// what the backend's Behaviour Analysis + LLM Translation layers would
/// push into the therapist dashboard once wired up.
List<ChildProfile> buildMockRoster() {
  final rnd = Random(7);
  final names = ['Aarav', 'Meher', 'Zayn', 'Ishaan'];
  final genders = [ChildGender.male, ChildGender.female, ChildGender.male, ChildGender.others];
  final ages = [6, 9, 5, 8];
  final conditions = [
    'Autism Spectrum Disorder',
    'Autism Spectrum Disorder',
    'Sensory Processing Disorder',
    'ADHD',
  ];

  return List.generate(names.length, (i) {
    final sessionCount = 4 + rnd.nextInt(5);
    final sessions = List.generate(sessionCount, (j) {
      final daysAgo = (sessionCount - j) * 2;
      final rec = SessionRecord.mockFromSession(
        recordingSeconds: 10,
        canvasSeconds: 15,
        strokeCount: 20 + rnd.nextInt(120),
      );
      return SessionRecord(
        id: '${names[i]}-$j',
        date: DateTime.now().subtract(Duration(days: daysAgo)),
        recordingSeconds: rec.recordingSeconds,
        canvasSeconds: rec.canvasSeconds,
        mood: rec.mood,
        valence: rec.valence,
        arousal: rec.arousal,
        dominance: rec.dominance,
        strokeCount: rec.strokeCount,
        caregiverSummary: rec.caregiverSummary,
        distressAlert: rec.distressAlert,
      );
    })
      ..sort((a, b) => a.date.compareTo(b.date));

    return ChildProfile(
      name: names[i],
      age: ages[i],
      gender: genders[i],
      condition: conditions[i],
      sessions: sessions,
    );
  });
}
