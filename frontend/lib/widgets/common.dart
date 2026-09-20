import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// Full-screen soft gradient backdrop used on every non-dashboard screen.
class GradientBackdrop extends StatelessWidget {
  final Widget child;
  const GradientBackdrop({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(gradient: AppGradients.background),
      child: child,
    );
  }
}

/// Floating pill-shaped "Back" button matching the reference screens.
class BackPill extends StatelessWidget {
  final VoidCallback? onTap;
  const BackPill({super.key, this.onTap});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.topLeft,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 0, 0),
        child: Material(
          color: Colors.white,
          borderRadius: BorderRadius.circular(30),
          elevation: 2,
          shadowColor: Colors.black12,
          child: InkWell(
            borderRadius: BorderRadius.circular(30),
            onTap: onTap ?? () => Navigator.of(context).maybePop(),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.arrow_back, size: 18, color: AppColors.primary),
                  const SizedBox(width: 6),
                  Text('Back',
                      style: GoogleFonts.fredoka(
                          color: AppColors.primary, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Big rounded gradient CTA button.
class PrimaryButton extends StatelessWidget {
  final String label;
  final IconData? trailingIcon;
  final VoidCallback? onPressed;
  final bool enabled;
  const PrimaryButton({
    super.key,
    required this.label,
    this.trailingIcon = Icons.arrow_forward,
    this.onPressed,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: enabled ? 1 : 0.5,
      child: Container(
        width: double.infinity,
        height: 56,
        decoration: BoxDecoration(
          gradient: AppGradients.primaryButton,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: AppColors.primary.withValues(alpha: 0.35),
              blurRadius: 16,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(16),
            onTap: enabled ? onPressed : null,
            child: Center(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(label,
                      style: GoogleFonts.fredoka(
                          color: Colors.white, fontSize: 17, fontWeight: FontWeight.w600)),
                  if (trailingIcon != null) ...[
                    const SizedBox(width: 8),
                    Icon(trailingIcon, color: Colors.white, size: 20),
                  ]
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class SecondaryButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final VoidCallback? onPressed;
  const SecondaryButton({super.key, required this.label, this.icon, this.onPressed});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          backgroundColor: Colors.white,
          side: const BorderSide(color: AppColors.accentSoft, width: 1.4),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (icon != null) ...[
              Icon(icon, color: AppColors.primary, size: 19),
              const SizedBox(width: 8),
            ],
            Text(label,
                style: GoogleFonts.fredoka(
                    color: AppColors.primary, fontSize: 16, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

/// The "1 Login ---- 2 Child Details" step header used on the caregiver flow.
class StepHeader extends StatelessWidget {
  final int activeStep; // 1 or 2
  const StepHeader({super.key, required this.activeStep});

  Widget _dot(int n, String label) {
    final active = n == activeStep;
    final done = n < activeStep;
    final filled = active || done;
    return Row(
      children: [
        CircleAvatar(
          radius: 14,
          backgroundColor: filled ? AppColors.primary : AppColors.accentSoft,
          child: Text('$n',
              style: GoogleFonts.fredoka(
                  color: filled ? Colors.white : AppColors.textMuted,
                  fontWeight: FontWeight.w600,
                  fontSize: 13)),
        ),
        const SizedBox(width: 8),
        Text(label,
            style: GoogleFonts.fredoka(
                color: filled ? AppColors.primary : AppColors.textMuted,
                fontWeight: FontWeight.w600,
                fontSize: 15)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        _dot(1, 'Login'),
        Container(width: 40, height: 2, margin: const EdgeInsets.symmetric(horizontal: 10), color: AppColors.accentSoft),
        _dot(2, 'Child Details'),
      ],
    );
  }
}

class ArtSpeakLogo extends StatelessWidget {
  final double size;
  const ArtSpeakLogo({super.key, this.size = 44});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            gradient: AppGradients.primaryButton,
            borderRadius: BorderRadius.circular(size * 0.28),
          ),
          child: Icon(Icons.graphic_eq_rounded, color: Colors.white, size: size * 0.55),
        ),
        const SizedBox(width: 12),
        Text('ArtSpeak',
            style: GoogleFonts.fredoka(
                fontSize: size * 0.5, fontWeight: FontWeight.w700, color: AppColors.primaryDark)),
      ],
    );
  }
}

/// Slim persistent footer bar used at the bottom of the main app screens
/// (the caregiver and therapist dashboards), giving the app a proper
/// mobile header/footer shell instead of a bare scrolling page.
class AppFooter extends StatelessWidget {
  final VoidCallback onLogout;
  const AppFooter({super.key, required this.onLogout});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.fromLTRB(20, 10, 12, 10 + MediaQuery.of(context).padding.bottom),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, -4))],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(width: 7, height: 7, decoration: const BoxDecoration(color: AppColors.primary, shape: BoxShape.circle)),
                const SizedBox(width: 8),
                Flexible(
                  child: Text('© 2026 ArtSpeak · Art Therapy Platform',
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 11.5)),
                ),
              ],
            ),
          ),
          TextButton.icon(
            onPressed: onLogout,
            icon: const Icon(Icons.logout_rounded, size: 16, color: AppColors.primary),
            label: Text('Log out', style: GoogleFonts.fredoka(color: AppColors.primary, fontWeight: FontWeight.w600, fontSize: 13)),
          ),
        ],
      ),
    );
  }
}

/// Consistent app header used on the main app screens: the ArtSpeak logo
/// plus a small screen-name subtitle beneath it.
class AppHeader extends StatelessWidget implements PreferredSizeWidget {
  final String subtitle;
  const AppHeader({super.key, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return AppBar(
      automaticallyImplyLeading: false,
      titleSpacing: 20,
      toolbarHeight: 58,
      title: const ArtSpeakLogo(size: 30),
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(24),
        child: Padding(
          padding: const EdgeInsets.only(left: 20, bottom: 8),
          child: Align(
            alignment: Alignment.centerLeft,
            child: Text(subtitle, style: GoogleFonts.nunito(color: AppColors.textMuted, fontSize: 12.5)),
          ),
        ),
      ),
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(82);
}

/// Small rounded stat tile used on the caregiver dashboard hero card.
class StatChip extends StatelessWidget {
  final String value;
  final String label;
  const StatChip({super.key, required this.value, required this.label});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 4),
        padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.18),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          children: [
            Text(value,
                style: GoogleFonts.fredoka(
                    fontSize: 22, fontWeight: FontWeight.w700, color: Colors.white)),
            const SizedBox(height: 2),
            Text(label, style: GoogleFonts.nunito(fontSize: 12, color: Colors.white70)),
          ],
        ),
      ),
    );
  }
}
