import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'therapist_dashboard_screen.dart';
import 'therapist_signup_screen.dart';
import 'welcome_screen.dart';
import '../services/art_speak_repository.dart';

class TherapistLoginScreen extends StatefulWidget {
  const TherapistLoginScreen({super.key});

  @override
  State<TherapistLoginScreen> createState() => _TherapistLoginScreenState();
}

class _TherapistLoginScreenState extends State<TherapistLoginScreen> {
  final _usernameCtrl = TextEditingController();
  final _passCtrl = TextEditingController();

  bool get _canSubmit => _usernameCtrl.text.trim().isNotEmpty && _passCtrl.text.trim().isNotEmpty;
  bool _busy = false;

  Future<void> _login() async {
    setState(() => _busy = true);
    try {
      await artSpeakRepository.signInTherapist(username: _usernameCtrl.text.trim(), password: _passCtrl.text);
      if (mounted) Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const TherapistDashboardScreen()));
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(error.toString())));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GradientBackdrop(
        child: SafeArea(
          child: Stack(
            children: [
              Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 90),
                  child: Container(
                    padding: const EdgeInsets.all(28),
                    constraints: const BoxConstraints(maxWidth: 460),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(26),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 24, offset: const Offset(0, 12))],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const ArtSpeakLogo(size: 38),
                        const SizedBox(height: 22),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                          decoration: BoxDecoration(
                              color: AppColors.accentSoft, borderRadius: BorderRadius.circular(20)),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(width: 8, height: 8, decoration: const BoxDecoration(color: AppColors.primary, shape: BoxShape.circle)),
                              const SizedBox(width: 8),
                              Text('THERAPIST PORTAL',
                                  style: GoogleFonts.fredoka(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.primary, letterSpacing: 0.5)),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text('Therapist Login',
                            style: GoogleFonts.fredoka(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                        const SizedBox(height: 6),
                        Text('Access your patient sessions and progress reports.',
                            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 14.5)),
                        const SizedBox(height: 24),
                        Text('Username', style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, fontSize: 14)),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _usernameCtrl,
                          onChanged: (_) => setState(() {}),
                          decoration: const InputDecoration(hintText: 'e.g. sarah4821'),
                        ),
                        const SizedBox(height: 20),
                        Text('Password', style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, fontSize: 14)),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _passCtrl,
                          onChanged: (_) => setState(() {}),
                          obscureText: true,
                          decoration: const InputDecoration(hintText: '••••••••'),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Use the credentials provided by your administrator after sign up.',
                          style: GoogleFonts.nunito(fontSize: 12, color: AppColors.textMuted, fontStyle: FontStyle.italic),
                        ),
                        const SizedBox(height: 26),
                        PrimaryButton(
                          label: 'Sign In',
                          trailingIcon: null,
                          enabled: _canSubmit,
                            onPressed: _busy ? null : _login,
                        ),
                        const SizedBox(height: 18),
                        Center(
                          child: GestureDetector(
                            onTap: () => Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => const TherapistSignupScreen())),
                            child: Wrap(
                              children: [
                                Text("Don't have an account? ", style: GoogleFonts.nunito(color: AppColors.textMuted)),
                                Text('Sign up',
                                    style: GoogleFonts.fredoka(color: AppColors.primary, fontWeight: FontWeight.w700)),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              BackPill(
                onTap: () => Navigator.of(context).pushAndRemoveUntil(
                  MaterialPageRoute(builder: (_) => const WelcomeScreen()),
                  (_) => false,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
