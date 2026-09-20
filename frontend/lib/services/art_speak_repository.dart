import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../models/child_profile.dart';
import '../models/session_record.dart';

class ArtSpeakRepository {
  SupabaseClient get _client => Supabase.instance.client;

  User get currentUser {
    final user = _client.auth.currentUser;
    if (user == null) throw StateError('Please sign in first.');
    return user;
  }

  String authErrorMessage(Object error) {
    if (error is http.ClientException) {
      return 'Unable to reach Supabase. Check the Supabase URL, internet connection, and browser network access.';
    }
    if (error is AuthException) {
      final message = error.message.toLowerCase();
      if (message.contains('invalid login credentials')) return 'Email or password is incorrect.';
      if (message.contains('email not confirmed')) return 'Confirm your email address before signing in.';
      if (message.contains('password')) return error.message;
      return error.message;
    }
    if (error is PostgrestException) {
      return error.message;
    }
    return error.toString();
  }

  Future<void> signUpCaregiver({
    required String email,
    required String password,
    required String name,
    required int age,
    required String gender,
    required String relationWithChild,
  }) async {
    await _client.auth.signUp(
      email: email,
      password: password,
      data: {
        'role': 'caregiver',
        'name': name,
        'age': age.toString(),
        'gender': gender,
        'relation_with_child': relationWithChild,
      },
    );
  }

  Future<String> signUpTherapist({
    required String name,
    required int age,
    required String gender,
    required String qualification,
    required String licenseNumber,
    required String contactEmail,
    required String password,
  }) async {
    final username = await _client.rpc('generate_unique_therapist_username', params: {'p_full_name': name}) as String;
    final loginEmail = '$username@artspeak.internal';
    await _client.auth.signUp(
      email: loginEmail,
      password: password,
      data: {
        'role': 'therapist',
        'name': name,
        'age': age.toString(),
        'gender': gender,
        'qualification': qualification,
        'license_number': licenseNumber,
        'contact_email': contactEmail,
        'username': username,
      },
    );
    return username;
  }

  Future<void> signIn({required String email, required String password}) async {
    await _client.auth.signInWithPassword(email: email, password: password);
  }

  Future<void> signInTherapist({required String username, required String password}) async {
    final loginEmail = await _client.rpc('get_therapist_login_email', params: {'p_username': username});
    if (loginEmail == null) throw StateError('This therapist account is not verified yet.');
    await _client.auth.signInWithPassword(email: loginEmail as String, password: password);
  }

  Future<String> createChild(ChildProfile profile) async {
    final row = await _client.from('children').insert({
      'caregiver_id': currentUser.id,
      'name': profile.name,
      'age': profile.age,
      'gender': profile.gender.databaseValue,
      'conditions': _asTags(profile.condition),
      'hobbies': _asTags(profile.hobbies),
      'known_triggers': _asTags(profile.knownTriggers),
      'calming_activities': _asTags(profile.calmingActivities),
      'needs_therapist': profile.therapistNeeded,
    }).select('id').single();
    return row['id'] as String;
  }

  List<String> _asTags(String? value) => value == null || value.trim().isEmpty
      ? <String>[]
      : value.split(',').map((tag) => tag.trim()).where((tag) => tag.isNotEmpty).toList();

  Future<String> createSession({required String childId, required int durationSeconds}) async {
    final row = await _client.from('sessions').insert({
      'caregiver_id': currentUser.id,
      'child_id': childId,
      'video_duration_seconds': durationSeconds,
      'art_duration_seconds': durationSeconds,
    }).select('id').single();
    return row['id'] as String;
  }

  Future<void> saveTouchEvents(String sessionId, List<Map<String, dynamic>> events) async {
    if (events.isEmpty) return;
    await _client.from('session_touch_events').insert(
      events.map((event) => {...event, 'session_id': sessionId}).toList(),
    );
  }

  Future<void> saveVideo(String sessionId, Uint8List videoBytes, int durationSeconds) async {
    final path = '${currentUser.id}/$sessionId.mp4';
    await _client.storage.from('session-videos').uploadBinary(
          path,
          videoBytes,
          fileOptions: const FileOptions(upsert: false, contentType: 'video/mp4'),
        );
    await _client.from('session_video_recordings').insert({
      'session_id': sessionId,
      'storage_path': path,
      'duration_seconds': durationSeconds,
    });
  }

  Future<void> markProcessing(String sessionId) async {
    await _client.from('sessions').update({'status': 'processing', 'ended_at': DateTime.now().toIso8601String()}).eq('id', sessionId);
  }

  Future<void> requestInference(String sessionId) async {
    const baseUrl = String.fromEnvironment('INFERENCE_API_URL');
    if (baseUrl.isEmpty) throw StateError('INFERENCE_API_URL is not configured.');
    final response = await http.post(
      Uri.parse('$baseUrl/process/$sessionId'),
      headers: {
        'Authorization': 'Bearer ${_client.auth.currentSession?.accessToken ?? ''}',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'session_id': sessionId}),
    );
    if (response.statusCode >= 300) throw StateError('Inference request failed: ${response.body}');
  }

  Future<List<ChildProfile>> loadTherapistRoster() async {
    final assignments = await _client.from('child_therapist_assignments').select('child_id').eq('therapist_id', currentUser.id).eq('active', true);
    final roster = <ChildProfile>[];
    for (final assignment in assignments) {
      final child = await _client.from('children').select().eq('id', assignment['child_id']).single();
      final reports = await _client.from('session_reports').select().eq('child_id', child['id']).order('created_at', ascending: false);
      roster.add(ChildProfile(
        id: child['id'],
        name: child['name'],
        age: child['age'],
        gender: childGenderFromDatabase(child['gender']),
        condition: (child['conditions'] as List?)?.cast<String>().join(', '),
        hobbies: (child['hobbies'] as List?)?.cast<String>().join(', '),
        knownTriggers: (child['known_triggers'] as List?)?.cast<String>().join(', '),
        calmingActivities: (child['calming_activities'] as List?)?.cast<String>().join(', '),
        therapistNeeded: child['needs_therapist'] as bool? ?? false,
        sessions: reports.map<SessionRecord>((report) => SessionRecord.fromReport(report)).toList(),
      ));
    }
    return roster;
  }
}

final artSpeakRepository = ArtSpeakRepository();