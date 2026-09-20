import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/child_profile.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import '../services/art_speak_repository.dart';

class TherapistSignupScreen extends StatefulWidget {
  const TherapistSignupScreen({super.key});

  @override
  State<TherapistSignupScreen> createState() => _TherapistSignupScreenState();
}

class _TherapistSignupScreenState extends State<TherapistSignupScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _qualificationCtrl = TextEditingController();
  final _licenseCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  ChildGender? _gender;
  bool _busy = false;

  bool get _canSubmit =>
      _nameCtrl.text.trim().isNotEmpty &&
      _ageCtrl.text.trim().isNotEmpty &&
      _gender != null &&
      _qualificationCtrl.text.trim().isNotEmpty &&
      _licenseCtrl.text.trim().isNotEmpty &&
      _emailCtrl.text.trim().isNotEmpty &&
      _passCtrl.text.trim().isNotEmpty;

  Future<void> _submit() async {
    setState(() => _busy = true);
    try {
      final username = await artSpeakRepository.signUpTherapist(
        name: _nameCtrl.text.trim(),
        age: int.parse(_ageCtrl.text.trim()),
        gender: _gender!.label,
        qualification: _qualificationCtrl.text.trim(),
        licenseNumber: _licenseCtrl.text.trim(),
        contactEmail: _emailCtrl.text.trim(),
        password: _passCtrl.text,
      );
      if (!mounted) return;
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Application submitted. Your username is $username.')));
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
              BackPill(onTap: () => Navigator.of(context).pop()),
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
                        Text('Therapist Sign Up',
                            style: GoogleFonts.fredoka(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                        const SizedBox(height: 6),
                        Text('Apply for access to review patient sessions and progress reports.',
                            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 14.5)),
                        const SizedBox(height: 24),

                        _field(
                          label: 'Full Name *',
                          child: TextField(
                            controller: _nameCtrl,
                            onChanged: (_) => setState(() {}),
                            decoration: const InputDecoration(hintText: 'e.g. Dr. Anya Kapoor'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(
                              child: _field(
                                label: 'Age *',
                                child: TextField(
                                  controller: _ageCtrl,
                                  onChanged: (_) => setState(() {}),
                                  keyboardType: TextInputType.number,
                                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                                  decoration: const InputDecoration(hintText: 'e.g. 41'),
                                ),
                              ),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: _field(
                                label: 'Gender *',
                                child: DropdownButtonFormField<ChildGender>(
                                  value: _gender,
                                  decoration: const InputDecoration(hintText: 'Select'),
                                  items: ChildGender.values
                                      .map((g) => DropdownMenuItem(value: g, child: Text(g.label)))
                                      .toList(),
                                  onChanged: (v) => setState(() => _gender = v),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Qualification *',
                          child: TextField(
                            controller: _qualificationCtrl,
                            onChanged: (_) => setState(() {}),
                            decoration: const InputDecoration(hintText: 'e.g. M.Sc. Occupational Therapy'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'License Number *',
                          child: TextField(
                            controller: _licenseCtrl,
                            onChanged: (_) => setState(() {}),
                            decoration: const InputDecoration(hintText: 'e.g. LPC-2024-00123'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Email address *',
                          child: TextField(
                            controller: _emailCtrl,
                            onChanged: (_) => setState(() {}),
                            keyboardType: TextInputType.emailAddress,
                            inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'[a-zA-Z0-9@._%+\-]'))],
                            decoration: const InputDecoration(hintText: 'therapist@clinic.com'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Password *',
                          child: TextField(
                            controller: _passCtrl,
                            onChanged: (_) => setState(() {}),
                            obscureText: true,
                            decoration: const InputDecoration(hintText: '••••••••'),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'After review, your administrator will provide your login credentials.',
                          style: GoogleFonts.nunito(fontSize: 12, color: AppColors.textMuted, fontStyle: FontStyle.italic),
                        ),
                        const SizedBox(height: 22),
                        PrimaryButton(
                          label: 'Sign Up',
                          enabled: _canSubmit,
                          onPressed: _busy ? null : _submit,
                        ),
                        const SizedBox(height: 18),
                        Center(
                          child: GestureDetector(
                            onTap: () => Navigator.of(context).pop(),
                            child: Wrap(
                              children: [
                                Text('Already have credentials? ', style: GoogleFonts.nunito(color: AppColors.textMuted)),
                                Text('Log in',
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
            ],
          ),
        ),
      ),
    );
  }

  Widget _field({required String label, required Widget child}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, fontSize: 14)),
        const SizedBox(height: 8),
        child,
      ],
    );
  }
}
