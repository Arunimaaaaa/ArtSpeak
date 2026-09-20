import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import '../services/art_speak_repository.dart';
import 'caregiver_signup_screen.dart';
import 'child_details_screen.dart';
import 'welcome_screen.dart';

class CaregiverLoginScreen extends StatefulWidget {
  const CaregiverLoginScreen({super.key});

  @override
  State<CaregiverLoginScreen> createState() => _CaregiverLoginScreenState();
}

class _CaregiverLoginScreenState extends State<CaregiverLoginScreen> {
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  bool get _validEmail => RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(_emailCtrl.text.trim());
  bool get _canContinue => _validEmail && _passCtrl.text.length >= 6;
  bool _busy = false;

  Future<void> _login() async {
    setState(() => _busy = true);
    try {
      await artSpeakRepository.signIn(email: _emailCtrl.text.trim(), password: _passCtrl.text);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const ChildDetailsScreen()));
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(artSpeakRepository.authErrorMessage(error))));
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
                        const SizedBox(height: 26),
                        const StepHeader(activeStep: 1),
                        const SizedBox(height: 24),
                        Text('Caregiver Login',
                            style: GoogleFonts.fredoka(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                        const SizedBox(height: 6),
                        Text('Sign in to manage sessions for your child.',
                            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 14.5)),
                        const SizedBox(height: 24),
                        Text('Email address', style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, fontSize: 14)),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _emailCtrl,
                          onChanged: (_) => setState(() {}),
                          keyboardType: TextInputType.emailAddress,
                          decoration: InputDecoration(
                            hintText: 'you@example.com',
                            errorText: _emailCtrl.text.isNotEmpty && !_validEmail
                                ? 'Enter a valid email address'
                                : null,
                          ),
                        ),
                        const SizedBox(height: 20),
                        Text('Password', style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, fontSize: 14)),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _passCtrl,
                          onChanged: (_) => setState(() {}),
                          obscureText: true,
                          decoration: InputDecoration(
                            hintText: '••••••••',
                            errorText: _passCtrl.text.isNotEmpty && _passCtrl.text.length < 6 ? 'Use at least 6 characters' : null,
                          ),
                        ),
                        const SizedBox(height: 26),
                        PrimaryButton(
                          label: 'Continue',
                          enabled: _canContinue,
                            onPressed: _busy ? null : _login,
                        ),
                        const SizedBox(height: 18),
                        Center(
                          child: GestureDetector(
                            onTap: () => Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => const CaregiverSignupScreen())),
                            child: Wrap(
                              children: [
                                Text("Don't have an account? ", style: GoogleFonts.nunito(color: AppColors.textMuted)),
                                Text('Sign up free',
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
