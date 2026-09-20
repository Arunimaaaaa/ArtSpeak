import 'session_record.dart';

enum ChildGender { male, female, others }

extension ChildGenderLabel on ChildGender {
  String get label {
    switch (this) {
      case ChildGender.male:
        return 'Male';
      case ChildGender.female:
        return 'Female';
      case ChildGender.others:
        return 'Prefer not to say';
    }
  }

  String get databaseValue => label;

}

ChildGender childGenderFromDatabase(String? value) {
  return ChildGender.values.firstWhere(
    (gender) => gender.databaseValue == value,
    orElse: () => ChildGender.others,
  );
}

/// Represents the child a caregiver is supporting.
/// No photo upload is stored/collected — only text/structured fields,
/// matching the synopsis's "no personally identifiable data transmitted
/// externally" constraint.
class ChildProfile {
  final String? id;
  final String name;
  final int age;
  final ChildGender gender;
  final String? condition;
  final String? hobbies;
  final String? knownTriggers;
  final String? calmingActivities;
  final bool therapistNeeded;
  final List<SessionRecord> sessions;

  ChildProfile({
    this.id,
    required this.name,
    required this.age,
    required this.gender,
    this.condition,
    this.hobbies,
    this.knownTriggers,
    this.calmingActivities,
    this.therapistNeeded = false,
    List<SessionRecord>? sessions,
  }) : sessions = sessions ?? [];

  int get sessionCount => sessions.length;

  int get sessionsThisWeek {
    final now = DateTime.now();
    final weekAgo = now.subtract(const Duration(days: 7));
    return sessions.where((s) => s.date.isAfter(weekAgo)).length;
  }

  int get streakDays {
    if (sessions.isEmpty) return 0;
    final dates = sessions.map((s) => DateTime(s.date.year, s.date.month, s.date.day)).toSet().toList()
      ..sort((a, b) => b.compareTo(a));
    int streak = 1;
    for (int i = 0; i < dates.length - 1; i++) {
      if (dates[i].difference(dates[i + 1]).inDays == 1) {
        streak++;
      } else {
        break;
      }
    }
    return streak;
  }
}
