"""
Caregiver Report Generator for ArtSpeak
=======================================
Synthesizes multimodal predictions from all 4 models into a structured,
compassionate, and actionable Caregiver Report.
"""

import json
from datetime import datetime
import numpy as np


class CaregiverReportBuilder:
    def __init__(self):
        pass

    def build_report(
        self,
        session_id: str,
        duration_minutes: int,
        facial_pred: dict,
        gaze_pred: dict,
        touch_pred: dict,
        physio_pred: dict,
    ) -> dict:
        """
        Build unified Caregiver Report dictionary and Markdown presentation.
        """
        # 1. Overall Regulation State & Score (Fused from Touch, Physiology & Face)
        touch_state = touch_pred.get("behavioral_state", "Calm_Regulated")
        stress_flag = physio_pred.get("stress_detected", False)
        face_emotion = facial_pred.get("dominant_emotion", "Neutral")

        # Compute Composite Regulation Index (0.0 to 1.0)
        reg_weights = {
            "Calm_Regulated": 0.95,
            "Engaged_Focused": 0.90,
            "Excited_Stimming": 0.75,
            "Withdrawn_FlatAffect": 0.55,
            "Anxious_Overstimulated": 0.30,
            "Distressed_PreMeltdown": 0.10,
        }
        base_reg = reg_weights.get(touch_state, 0.70)
        if stress_flag:
            base_reg = min(base_reg, 0.35)

        regulation_score_pct = int(round(base_reg * 100))

        # 2. Regulation State Breakdown Percentages
        state_probs = touch_pred.get("state_probabilities", {})
        breakdown = {
            "Calm & Regulated": round(state_probs.get("Calm_Regulated", 0.0) * 100, 1),
            "Engaged & Focused": round(state_probs.get("Engaged_Focused", 0.0) * 100, 1),
            "Excited / Stimming Flow": round(state_probs.get("Excited_Stimming", 0.0) * 100, 1),
            "Overstimulated / Anxious": round(state_probs.get("Anxious_Overstimulated", 0.0) * 100, 1),
            "Distressed / Pre-Meltdown": round(state_probs.get("Distressed_PreMeltdown", 0.0) * 100, 1),
            "Withdrawn / Low Energy": round(state_probs.get("Withdrawn_FlatAffect", 0.0) * 100, 1),
        }

        # 3. Affect Coordinates (Valence, Arousal, Dominance) Fused
        fused_valence = float(np.mean([
            facial_pred.get("valence", 0.0),
            touch_pred.get("touch_valence", 0.0),
        ]))
        fused_arousal = float(np.mean([
            facial_pred.get("arousal", 0.3),
            touch_pred.get("touch_arousal", 0.3),
            0.8 if stress_flag else 0.2,
        ]))
        fused_dominance = float(np.mean([
            facial_pred.get("dominance", 0.5),
            touch_pred.get("touch_dominance", 0.5),
        ]))

        # 4. Synthesize Meaningful Insights & Caregiver Recommendations
        insights = []
        recommendations = []

        if touch_state in ["Calm_Regulated", "Engaged_Focused"]:
            insights.append("The child showed smooth, rhythmic motor engagement with consistent tap cadence.")
            recommendations.append("The current sensory environment is comfortable. Continue offering creative open-ended drawing tools.")
        elif touch_state == "Excited_Stimming":
            insights.append("Rapid rhythmic stimming bursts detected; the child expressed high energy and joyful visual stimulation.")
            recommendations.append("Encourage this expressive motor outlet; consider integrating musical or rhythmic companion sounds.")
        elif touch_state in ["Anxious_Overstimulated", "Distressed_PreMeltdown"] or stress_flag:
            insights.append("Elevated touch force and autonomic arousal spikes detected, signaling emerging sensory overstimulation.")
            recommendations.append("Offer a gentle sensory pause, dimmed screen brightness, or soothing background ambient tones.")
        elif touch_state == "Withdrawn_FlatAffect":
            insights.append("Low interaction force and quiet visual dwell times, suggesting fatigue or passive observation.")
            recommendations.append("Allow self-paced exploration without demanding active verbal or rapid responses.")

        if gaze_pred.get("spatial_spread", 0.0) > 0.25:
            insights.append("Wide visual exploration across the canvas, reflecting active exploratory focus.")
        else:
            insights.append("Centered, deep visual foveation on focal elements of the artwork.")

        # 5. Build Markdown Presentation
        report_md = f"""# Artspeak Caregiver Session Report
**Session ID**: `{session_id}` | **Duration**: {duration_minutes} minutes | **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 🌟 Overall Session State: **{touch_state.replace('_', ' ')}**
**Composite Regulation Index**: `{regulation_score_pct}%`

> **Caregiver Summary**: {insights[0]}

---

## 📊 Multimodal Breakdown Across Trained Layers

### 1. Motor & Touch Rhythm (Layer 1)
- **Primary Behavioral State**: `{touch_state}`
- **Rhythm Stability & Cadence**: Regular touch flow with cadence `{touch_pred.get('touch_valence', 0.5):.2f}` valence.
- **Regulation Distribution**:
  - 🟢 **Calm / Regulated**: {breakdown['Calm & Regulated']}%
  - 🔵 **Engaged / Focused**: {breakdown['Engaged & Focused']}%
  - 🟣 **Excited / Stimming**: {breakdown['Excited / Stimming Flow']}%
  - 🟠 **Overstimulated**: {breakdown['Overstimulated / Anxious']}%
  - 🔴 **Distressed**: {breakdown['Distressed / Pre-Meltdown']}%
  - ⚪ **Withdrawn**: {breakdown['Withdrawn / Low Energy']}%

### 2. Facial Expression & 3D Affect (Layer 1 & 2)
- **Dominant Facial Expression**: `{face_emotion}`
- **Affect Vector (3D VAD)**:
  - **Valence (Pleasantness)**: `{fused_valence:+.2f}` (Range -1.0 to +1.0)
  - **Arousal (Energy Level)**: `{fused_arousal:.2f}` (Range 0.0 to 1.0)
  - **Dominance (Sense of Control)**: `{fused_dominance:.2f}` (Range 0.0 to 1.0)

### 3. Visual Attention & Gaze Dynamics (Layer 1)
- **Visual Exploration Pattern**: `{gaze_pred.get('attention_profile', 'Focused')}`
- **Fixation Count**: {gaze_pred.get('fixation_count', 12)} stable fixations
- **Spatial Spread**: `{gaze_pred.get('spatial_spread', 0.15):.3f}`

### 4. Physiological Autonomic Balance (Layer 2)
- **Stress & Overload Status**: `{physio_pred.get('status', 'Regulated')}`
- **Autonomic Overload Probability**: `{physio_pred.get('stress_probability', 0.05) * 100:.1f}%`

---

## 💡 Supportive Recommendations for Caregivers
""" + "\n".join([f"- **{rec}**" for rec in recommendations])

        return {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "overall_state": touch_state,
            "regulation_score_pct": regulation_score_pct,
            "affect_vector": {
                "valence": fused_valence,
                "arousal": fused_arousal,
                "dominance": fused_dominance,
            },
            "regulation_breakdown": breakdown,
            "facial_expression": face_emotion,
            "gaze_attention": gaze_pred,
            "physiological_stress": physio_pred,
            "insights": insights,
            "recommendations": recommendations,
            "markdown_report": report_md,
        }
