import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'screens/welcome_screen.dart';
import 'theme/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // ArtSpeak is designed as a portrait-only mobile app.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  const supabaseUrl = String.fromEnvironment('SUPABASE_URL');
  const supabasePublishableKey = String.fromEnvironment('SUPABASE_ANON_KEY');
  if (supabaseUrl.isNotEmpty && supabasePublishableKey.isNotEmpty) {
    await Supabase.initialize(url: supabaseUrl, publishableKey: supabasePublishableKey);
  }
  runApp(ArtSpeakApp(supabaseConfigured: supabaseUrl.isNotEmpty && supabasePublishableKey.isNotEmpty));
}

class ArtSpeakApp extends StatelessWidget {
  final bool supabaseConfigured;

  const ArtSpeakApp({super.key, this.supabaseConfigured = false});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArtSpeak',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      home: supabaseConfigured ? const WelcomeScreen() : const SupabaseConfigurationScreen(),
    );
  }
}

class SupabaseConfigurationScreen extends StatelessWidget {
  const SupabaseConfigurationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'Supabase is not configured. Start Flutter with SUPABASE_URL and SUPABASE_ANON_KEY.',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}

