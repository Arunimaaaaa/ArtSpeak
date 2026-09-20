import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/child_profile.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import '../services/art_speak_repository.dart';
import 'caregiver_dashboard_screen.dart';

const _conditions = [
  'Autism Spectrum Disorder',
  'ADHD',
  'Sensory Processing Disorder',
  'Speech / Language Delay',
  'Other / Not specified',
];

class ChildDetailsScreen extends StatefulWidget {
  /// When true (default) this is the very first child added right after
  /// caregiver login, so submitting takes the caregiver straight into the
  /// dashboard. When false, this screen was opened from the dashboard
  /// sidebar's "Add Child" action, so submitting just pops the new
  /// [ChildProfile] back to the dashboard to be appended to the list.
  final bool isFirstProfile;
  const ChildDetailsScreen({super.key, this.isFirstProfile = true});

  @override
  State<ChildDetailsScreen> createState() => _ChildDetailsScreenState();
}

class _ChildDetailsScreenState extends State<ChildDetailsScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _hobbiesCtrl = TextEditingController();
  final _triggersCtrl = TextEditingController();
  final _calmingCtrl = TextEditingController();
  String? _condition;
  ChildGender? _gender;
  bool? _therapistNeeded;
  bool _busy = false;

  bool get _canSubmit =>
      _nameCtrl.text.trim().isNotEmpty &&
      _ageCtrl.text.trim().isNotEmpty &&
      _gender != null &&
      _therapistNeeded != null;

  Future<void> _submit() async {
    final profile = ChildProfile(
      name: _nameCtrl.text.trim(),
      age: int.tryParse(_ageCtrl.text.trim()) ?? 0,
      gender: _gender!,
      condition: _condition,
      hobbies: _hobbiesCtrl.text.trim().isEmpty ? null : _hobbiesCtrl.text.trim(),
      knownTriggers: _triggersCtrl.text.trim().isEmpty ? null : _triggersCtrl.text.trim(),
      calmingActivities: _calmingCtrl.text.trim().isEmpty ? null : _calmingCtrl.text.trim(),
      therapistNeeded: _therapistNeeded!,
    );
    setState(() => _busy = true);
    try {
      final childId = await artSpeakRepository.createChild(profile);
      final savedProfile = ChildProfile(
        id: childId,
        name: profile.name,
        age: profile.age,
        gender: profile.gender,
        condition: profile.condition,
        hobbies: profile.hobbies,
        knownTriggers: profile.knownTriggers,
        calmingActivities: profile.calmingActivities,
        therapistNeeded: profile.therapistNeeded,
      );
      if (!mounted) return;
      if (widget.isFirstProfile) {
        Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => CaregiverDashboardScreen(initialProfile: savedProfile)));
      } else {
        Navigator.of(context).pop(savedProfile);
      }
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
              BackPill(onTap: () => Navigator.of(context).pop()),
              Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 90),
                  child: Container(
                    padding: const EdgeInsets.all(28),
                    constraints: const BoxConstraints(maxWidth: 480),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(26),
                      boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 24, offset: const Offset(0, 12))],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const ArtSpeakLogo(size: 38),
                        const SizedBox(height: 26),
                        if (widget.isFirstProfile) ...[
                          const StepHeader(activeStep: 2),
                          const SizedBox(height: 24),
                        ],
                        Text(widget.isFirstProfile ? 'Child Details' : 'Add a Child',
                            style: GoogleFonts.fredoka(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                        const SizedBox(height: 6),
                        Text('Tell us about the child you are supporting.',
                            style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 14.5)),
                        const SizedBox(height: 24),

                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(
                              child: _field(
                                label: "Child's Name *",
                                child: TextField(
                                  controller: _nameCtrl,
                                  onChanged: (_) => setState(() {}),
                                  decoration: const InputDecoration(hintText: 'e.g. Lily'),
                                ),
                              ),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: _field(
                                label: 'Age *',
                                child: TextField(
                                  controller: _ageCtrl,
                                  onChanged: (_) => setState(() {}),
                                  keyboardType: TextInputType.number,
                                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                                  decoration: const InputDecoration(hintText: 'e.g. 7'),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Gender *',
                          child: DropdownButtonFormField<ChildGender>(
                            initialValue: _gender,
                            decoration: const InputDecoration(hintText: 'Select gender'),
                            items: ChildGender.values
                                .map((g) => DropdownMenuItem(value: g, child: Text(g.label)))
                                .toList(),
                            onChanged: (v) => setState(() => _gender = v),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Condition / Diagnosis',
                          child: DropdownButtonFormField<String>(
                            initialValue: _condition,
                            decoration: const InputDecoration(hintText: 'Select condition'),
                            items: _conditions
                                .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                                .toList(),
                            onChanged: (v) => setState(() => _condition = v),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Hobbies / Favorite Activities',
                          child: TextField(
                            controller: _hobbiesCtrl,
                            maxLines: 2,
                            decoration: const InputDecoration(hintText: 'e.g. Puzzles, drawing, music'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Known Triggers',
                          child: TextField(
                            controller: _triggersCtrl,
                            maxLines: 2,
                            decoration: const InputDecoration(hintText: 'e.g. Loud noises, sudden changes in routine'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Calming Activities',
                          child: TextField(
                            controller: _calmingCtrl,
                            maxLines: 2,
                            decoration: const InputDecoration(hintText: 'e.g. Deep breathing, weighted blanket, quiet corner'),
                          ),
                        ),
                        const SizedBox(height: 18),

                        _field(
                          label: 'Is a Therapist Needed? *',
                          child: DropdownButtonFormField<bool>(
                            initialValue: _therapistNeeded,
                            decoration: const InputDecoration(hintText: 'Select an option'),
                            items: const [
                              DropdownMenuItem(value: true, child: Text('Yes')),
                              DropdownMenuItem(value: false, child: Text('No')),
                            ],
                            onChanged: (v) => setState(() => _therapistNeeded = v),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'No photo is collected — ArtSpeak keeps session data local to this device.',
                          style: GoogleFonts.nunito(fontSize: 12, color: AppColors.textMuted, fontStyle: FontStyle.italic),
                        ),
                        const SizedBox(height: 22),
                        PrimaryButton(
                          label: widget.isFirstProfile ? 'Go to Dashboard' : 'Save Profile',
                          enabled: _canSubmit,
                          onPressed: _busy ? null : _submit,
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
