import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Central design tokens for ArtSpeak.
/// Mirrors the reference screenshots: soft indigo/blue gradients,
/// rounded "friendly" typography, generous corner radii, calm palette
/// (kept sensory-safe: no harsh saturation, no flashing elements).
class AppColors {
  static const primary = Color(0xFF3D5AFE); // main indigo-blue
  static const primaryDark = Color(0xFF2A3EB1);
  static const primaryLight = Color(0xFF7C93FF);
  static const accentSoft = Color(0xFFE8ECFF);
  static const bgTop = Color(0xFFEAF0FF);
  static const bgBottom = Color(0xFFF7F9FF);
  static const textDark = Color(0xFF1B2559);
  static const textMuted = Color(0xFF7C89B8);
  static const white = Colors.white;
  static const success = Color(0xFF3D5AFE);
  static const warnAmber = Color(0xFFF4A340);
  static const calmGreen = Color(0xFF4CAF8E);
  static const alertRed = Color(0xFFE0616B);

  static const List<Color> childColors = [
    Color(0xFF3D5AFE),
    Color(0xFF7C93FF),
    Color(0xFFA9B8FF),
    Color(0xFFE0616B),
    Color(0xFFF4A340),
    Color(0xFF4CAF8E),
    Color(0xFF8E6ED8),
  ];
}

class AppGradients {
  static const background = LinearGradient(
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
    colors: [AppColors.bgTop, AppColors.bgBottom],
  );

  static const primaryCard = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [AppColors.primary, AppColors.primaryDark],
  );

  static const primaryButton = LinearGradient(
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
    colors: [AppColors.primary, Color(0xFF5B72FF)],
  );
}

class AppTheme {
  static ThemeData light() {
    final base = ThemeData(useMaterial3: true, colorSchemeSeed: AppColors.primary);
    final textTheme = GoogleFonts.fredokaTextTheme(base.textTheme).copyWith(
      bodyMedium: GoogleFonts.nunito(fontSize: 15, color: AppColors.textDark),
      bodyLarge: GoogleFonts.nunito(fontSize: 16, color: AppColors.textDark),
      bodySmall: GoogleFonts.nunito(fontSize: 13, color: AppColors.textMuted),
    );
    return base.copyWith(
      scaffoldBackgroundColor: AppColors.bgTop,
      textTheme: textTheme,
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.white,
        contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
        hintStyle: GoogleFonts.nunito(color: AppColors.textMuted),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppColors.accentSoft, width: 1.4),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppColors.accentSoft, width: 1.4),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.8),
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.white,
        elevation: 0,
        centerTitle: true,
        surfaceTintColor: Colors.transparent,
        titleTextStyle: GoogleFonts.fredoka(
          fontSize: 18,
          fontWeight: FontWeight.w600,
          color: AppColors.primaryDark,
        ),
        iconTheme: const IconThemeData(color: AppColors.primary),
      ),
    );
  }
}
