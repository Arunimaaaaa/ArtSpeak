import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/child_profile.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import '../services/art_speak_repository.dart';

const _relations = [
  'Parent',
  'Guardian',
  'Grandparent',
  'Sibling',
  'Other',
];

class CaregiverSignupScreen extends StatefulWidget {
  const CaregiverSignupScreen({super.key});

  @override
  State<CaregiverSignupScreen> createState() => _CaregiverSignupScreenState();
}

class _CaregiverSignupScreenState extends State<CaregiverSignupScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  ChildGender? _gender;
  String? _relation;
  bool _busy = false;

  bool get _canSubmit =>
      _nameCtrl.text.trim().isNotEmpty &&
      _ageCtrl.text.trim().isNotEmpty &&
      _gender != null &&
      _relation != null &&
      _emailCtrl.text.trim().isNotEmpty &&
      _passCtrl.text.trim().isNotEmpty;

  Future<void> _submit() async {
    setState(() => _busy = true);
    try {
      await artSpeakRepository.signUpCaregiver(
        email: _emailCtrl.text.trim(),
        password: _passCtrl.text,
        name: _nameCtrl.text.trim(),
        age: int.parse(_ageCtrl.text.trim()),
        gender: _gender!.label,
        relationWithChild: _relation!,
      );
      if (!mounted) return;
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Account created. Check your email, then log in.')));
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
                        const SizedBox(height: 26),
                        Text('Caregiver Sign Up',
                            style: GoogleFonts.fredoka(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                        const SizedBox(height: 6),
                        Text('Create an account to start supporting your child.',
                            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 14.5)),
                        const SizedBox(height: 24),

                        _field(
                          label: 'Full Name *',
                          child: TextField(
                            controller: _nameCtrl,
                            onChanged: (_) => setState(() {}),
                            decoration: const InputDecoration(hintText: 'e.g. Priya Sharma'),
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
                                  decoration: const InputDecoration(hintText: 'e.g. 34'),
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
                          label: 'Relation with Child *',
                          child: DropdownButtonFormField<String>(
                            value: _relation,
                            decoration: const InputDecoration(hintText: 'Select relation'),
                            items: _relations.map((r) => DropdownMenuItem(value: r, child: Text(r))).toList(),
                            onChanged: (v) => setState(() => _relation = v),
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
                            decoration: const InputDecoration(hintText: 'you@example.com'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Password *',
                          child: TextField(
                            controller: _passCtrl,
                            onChanged: (_) => setState(() {}),
                            obscureText: true,
                              decoration: InputDecoration(
                                hintText: '••••••••',
                                errorText: _passCtrl.text.isNotEmpty && _passCtrl.text.length < 6 ? 'Use at least 6 characters' : null,
                              ),
                          ),
                        ),
                        const SizedBox(height: 26),
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
                                Text('Already have an account? ', style: GoogleFonts.nunito(color: AppColors.textMuted)),
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
