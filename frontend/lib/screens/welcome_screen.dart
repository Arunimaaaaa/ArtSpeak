import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'caregiver_login_screen.dart';
import 'therapist_login_screen.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GradientBackdrop(
        child: SafeArea(
          child: Stack(
            children: [
              // top bar
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const ArtSpeakLogo(size: 40),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)],
                      ),
                      child: Text('v1.0', style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 12)),
                    ),
                  ],
                ),
              ),
              SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 90, 24, 24),
                child: Column(
                  children: [
                    // hero blob icon
                    Container(
                      width: 140,
                      height: 140,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppColors.primaryLight.withOpacity(0.25),
                      ),
                      child: Center(
                        child: Container(
                          width: 96,
                          height: 96,
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: AppGradients.primaryButton,
                          ),
                          child: const Icon(Icons.mic_none_rounded, color: Colors.white, size: 42),
                        ),
                      ),
                    ),
                    const SizedBox(height: 32),
                    Text('Welcome to ArtSpeak',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.fredoka(
                            fontSize: 32, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
                    const SizedBox(height: 16),
                    Text(
                      'Where every stroke tells a story and creativity becomes communication.',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.nunito(fontSize: 16, color: AppColors.primary, fontWeight: FontWeight.w600, height: 1.4),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Supporting children through art therapy — one session at a time.',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.nunito(fontSize: 14, color: AppColors.textMuted, height: 1.4),
                    ),
                    const SizedBox(height: 36),
                    _LoginCard(
                      filled: true,
                      icon: Icons.escalator_warning_rounded,
                      title: 'Caregiver Login',
                      subtitle: 'Manage child profiles, record and review art sessions',
                      cta: 'Get started',
                      onTap: () => Navigator.of(context).push(
                          MaterialPageRoute(builder: (_) => const CaregiverLoginScreen())),
                    ),
                    const SizedBox(height: 16),
                    _LoginCard(
                      filled: false,
                      icon: Icons.assignment_add,
                      title: 'Therapist Login',
                      subtitle: 'Review sessions, track progress and generate insights',
                      cta: 'Sign in',
                      onTap: () => Navigator.of(context).push(
                          MaterialPageRoute(builder: (_) => const TherapistLoginScreen())),
                    ),
                    const SizedBox(height: 28),
                    Text('© 2026 ArtSpeak — Art Therapy Platform',
                        style: GoogleFonts.nunito(color: AppColors.textMuted.withOpacity(0.7), fontSize: 12)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LoginCard extends StatelessWidget {
  final bool filled;
  final IconData icon;
  final String title;
  final String subtitle;
  final String cta;
  final VoidCallback onTap;

  const _LoginCard({
    required this.filled,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.cta,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final fg = filled ? Colors.white : AppColors.primaryDark;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            gradient: filled ? AppGradients.primaryCard : null,
            color: filled ? null : Colors.white,
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: filled ? AppColors.primary.withOpacity(0.35) : Colors.black.withOpacity(0.05),
                blurRadius: 18,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: filled ? Colors.white.withOpacity(0.22) : AppColors.accentSoft,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: filled ? Colors.white : AppColors.primary),
              ),
              const SizedBox(height: 16),
              Text(title,
                  style: GoogleFonts.fredoka(fontSize: 20, fontWeight: FontWeight.w700, color: fg)),
              const SizedBox(height: 6),
              Text(subtitle,
                  style: GoogleFonts.nunito(
                      fontSize: 13.5, color: filled ? Colors.white70 : AppColors.textMuted, height: 1.35)),
              const SizedBox(height: 14),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(cta,
                      style: GoogleFonts.fredoka(
                          fontSize: 14.5, fontWeight: FontWeight.w600, color: filled ? Colors.white : AppColors.primary)),
                  const SizedBox(width: 6),
                  Icon(Icons.arrow_forward, size: 16, color: filled ? Colors.white : AppColors.primary),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
