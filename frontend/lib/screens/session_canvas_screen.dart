import 'dart:async';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/session_record.dart';
import '../theme/app_theme.dart';
import '../services/art_speak_repository.dart';

/// Total length of the combined draw + facial-expression-capture session.
const int kSessionSeconds = 20;

class _Stroke {
  final List<Offset> points;
  final Color color;
  final double width;
  _Stroke(this.color, this.width) : points = [];
}

/// A simple coloring-book style outline the child is asked to trace/draw,
/// picked at random for each session.
enum _RefPicture { house, star, sun, flower, butterfly }

class SessionCanvasScreen extends StatefulWidget {
  final String childName;
  final String sessionId;
  const SessionCanvasScreen({super.key, required this.childName, required this.sessionId});

  @override
  State<SessionCanvasScreen> createState() => _SessionCanvasScreenState();
}

class _SessionCanvasScreenState extends State<SessionCanvasScreen> {
  final List<_Stroke> _strokes = [];
  Color _color = const Color(0xFF3D5AFE);
  double _brushSize = 6;
  Timer? _timer;
  int _secondsLeft = kSessionSeconds;
  bool _sessionOver = false;
  late final _RefPicture _refPicture;

  CameraController? _cameraController;
  bool _isRecording = false;
  bool _cameraStarting = true;
  String? _cameraError;
  XFile? _recordedVideo;
  late final DateTime _sessionStartedAt;
  bool _saving = false;

  final _palette = const [
    Color(0xFF3D5AFE),
    Color(0xFF7C93FF),
    Color(0xFFA9C2FF),
    Color(0xFFE0616B),
    Color(0xFFF4A340),
    Color(0xFF4CAF8E),
    Color(0xFF8E6ED8),
  ];
  final _sizes = const [3.0, 6.0, 10.0, 16.0];

  @override
  void initState() {
    super.initState();
    _sessionStartedAt = DateTime.now();
    _refPicture = _RefPicture.values[Random().nextInt(_RefPicture.values.length)];
    _initHiddenCameraAndStart();
  }

  /// Starts the front-camera recording and keeps a preview visible so camera
  /// permission and recording state are clear during a session.
  Future<void> _initHiddenCameraAndStart() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw StateError('No camera was found on this device.');
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(cam, ResolutionPreset.medium, enableAudio: false);
      await controller.initialize();
      await controller.startVideoRecording();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _cameraController = controller;
        _isRecording = true;
        _cameraStarting = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() {
          _cameraStarting = false;
          _cameraError = error.toString().replaceFirst('Bad state: ', '');
        });
      }
    }
    _startTimer();
  }

  void _startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _secondsLeft--);
      if (_secondsLeft <= 0) {
        t.cancel();
        _finishRecordingTimer();
      }
    });
  }

  Future<void> _finishRecordingTimer() async {
    await _stopHiddenCamera();
    if (mounted) setState(() => _sessionOver = true);
  }

  Future<void> _stopHiddenCamera() async {
    final controller = _cameraController;
    if (controller != null && _isRecording) {
      try {
        _recordedVideo = await controller.stopVideoRecording();
      } catch (error) {
        if (mounted) {
          setState(() => _cameraError = 'Camera recording failed: $error');
        }
      }
      _isRecording = false;
    }
  }

  void _startStroke(Offset p) {
    setState(() => _strokes.add(_Stroke(_color, _brushSize)..points.add(p)));
  }

  void _extendStroke(Offset p) {
    setState(() => _strokes.last.points.add(p));
  }

  Future<void> _endSession() async {
    if (_saving) return;
    setState(() => _saving = true);
    await _stopHiddenCamera();
    final strokeCount = _strokes.fold<int>(0, (sum, s) => sum + s.points.length);
    final events = <Map<String, dynamic>>[];
    var strokeId = 0;
    for (final stroke in _strokes) {
      for (var i = 0; i < stroke.points.length; i++) {
        final point = stroke.points[i];
        events.add({
          'timestamp_ms': DateTime.now().difference(_sessionStartedAt).inMilliseconds,
          'stroke_id': strokeId,
          'event_type': i == 0 ? 'down' : (i == stroke.points.length - 1 ? 'up' : 'move'),
          'x': point.dx,
          'y': point.dy,
          'pressure': 0.5,
          'brush_size': stroke.width,
        });
      }
      strokeId++;
    }
    try {
      await artSpeakRepository.saveTouchEvents(widget.sessionId, events);
      final video = _recordedVideo;
      if (video == null) throw StateError('No camera recording was produced. Check camera permission and try again.');
      await artSpeakRepository.saveVideo(widget.sessionId, await video.readAsBytes(), kSessionSeconds);
      await artSpeakRepository.markProcessing(widget.sessionId);
      await artSpeakRepository.requestInference(widget.sessionId);
      if (mounted) Navigator.of(context).pop(SessionRecord.processing(id: widget.sessionId, seconds: kSessionSeconds, strokeCount: strokeCount));
    } catch (error) {
      if (mounted) {
        setState(() => _saving = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(error.toString())));
      }
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: Column(
          children: [
            Text("${widget.childName}'s Session",
                style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, color: AppColors.primaryDark, fontSize: 16)),
            Text('${_strokes.length} strokes', style: GoogleFonts.nunito(fontSize: 11, color: AppColors.textMuted)),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: _sessionOver ? AppColors.alertRed.withOpacity(0.1) : AppColors.accentSoft,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    Icon(Icons.timer_outlined, size: 16, color: _sessionOver ? AppColors.alertRed : AppColors.primary),
                    const SizedBox(width: 6),
                    Text(_sessionOver ? "Time's up" : '$_secondsLeft s',
                        style: GoogleFonts.fredoka(
                            fontWeight: FontWeight.w700, color: _sessionOver ? AppColors.alertRed : AppColors.primary)),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      body: Row(
        children: [
          _ToolRail(
            palette: _palette,
            sizes: _sizes,
            selectedColor: _color,
            selectedSize: _brushSize,
            onColor: (c) => setState(() => _color = c),
            onSize: (s) => setState(() => _brushSize = s),
          ),
          Expanded(
            child: Column(
              children: [
                // Top half: the reference picture to trace/draw. No gesture
                // handling here at all, so it can never be scribbled on.
                Expanded(
                  flex: 1,
                  child: Container(
                    width: double.infinity,
                    color: const Color(0xFFFDF9F0),
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(
                              child: Text('Try to draw this picture below!',
                                  style: GoogleFonts.fredoka(fontWeight: FontWeight.w600, color: AppColors.primaryDark, fontSize: 14)),
                            ),
                            _CameraStatus(
                              controller: _cameraController,
                              starting: _cameraStarting,
                              recording: _isRecording,
                              error: _cameraError,
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Expanded(
                          child: Center(
                            child: AspectRatio(
                              aspectRatio: 1.3,
                              child: CustomPaint(painter: _ReferencePainter(_refPicture)),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const Divider(height: 2, thickness: 2, color: Color(0xFFE9ECF5)),
                // Bottom half: the actual drawing canvas.
                Expanded(
                  flex: 1,
                  child: Stack(
                    children: [
                      Container(color: const Color(0xFFF7F9FF)),
                      GestureDetector(
                        onPanStart: (d) => _startStroke(d.localPosition),
                        onPanUpdate: (d) => _extendStroke(d.localPosition),
                        child: CustomPaint(
                          painter: _CanvasPainter(_strokes),
                          size: Size.infinite,
                        ),
                      ),
                      if (_strokes.isEmpty && !_sessionOver)
                        Center(
                          child: Text('Draw here, ${widget.childName}…',
                              style: GoogleFonts.nunito(color: AppColors.textMuted.withOpacity(0.6), fontSize: 16)),
                        ),
                      if (_sessionOver)
                        Positioned(
                          left: 0,
                          right: 0,
                          bottom: 20,
                          child: Center(
                            child: Container(
                              padding: const EdgeInsets.all(6),
                              decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(20),
                                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.08), blurRadius: 18, offset: const Offset(0, 8))]),
                              child: ElevatedButton.icon(
                                onPressed: _saving ? null : _endSession,
                                icon: const Icon(Icons.check_circle_rounded),
                                label: Text('End Session', style: GoogleFonts.fredoka(fontWeight: FontWeight.w700, fontSize: 16)),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: AppColors.primary,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                                ),
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ToolRail extends StatelessWidget {
  final List<Color> palette;
  final List<double> sizes;
  final Color selectedColor;
  final double selectedSize;
  final ValueChanged<Color> onColor;
  final ValueChanged<double> onSize;

  const _ToolRail({
    required this.palette,
    required this.sizes,
    required this.selectedColor,
    required this.selectedSize,
    required this.onColor,
    required this.onSize,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 88,
      color: Colors.white,
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: SingleChildScrollView(
        child: Column(
          children: [
            Text('SIZE', style: GoogleFonts.nunito(fontSize: 10, color: AppColors.textMuted, letterSpacing: 1)),
            const SizedBox(height: 10),
            ...sizes.map((s) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: GestureDetector(
                    onTap: () => onSize(s),
                    child: Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: selectedSize == s ? AppColors.primary : AppColors.accentSoft, width: 1.6),
                      ),
                      child: Center(
                        child: Container(
                          width: s + 4,
                          height: s + 4,
                          decoration: BoxDecoration(color: AppColors.textDark.withOpacity(0.6), shape: BoxShape.circle),
                        ),
                      ),
                    ),
                  ),
                )),
            const SizedBox(height: 16),
            Text('COLOR', style: GoogleFonts.nunito(fontSize: 10, color: AppColors.textMuted, letterSpacing: 1)),
            const SizedBox(height: 10),
            ...palette.map((c) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: GestureDetector(
                    onTap: () => onColor(c),
                    child: Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: c,
                        shape: BoxShape.circle,
                        border: selectedColor == c ? Border.all(color: Colors.white, width: 3) : null,
                        boxShadow: selectedColor == c
                            ? [BoxShadow(color: c.withOpacity(0.6), blurRadius: 8, spreadRadius: 1)]
                            : [],
                      ),
                    ),
                  ),
                )),
          ],
        ),
      ),
    );
  }
}

class _CameraStatus extends StatelessWidget {
  final CameraController? controller;
  final bool starting;
  final bool recording;
  final String? error;

  const _CameraStatus({
    required this.controller,
    required this.starting,
    required this.recording,
    required this.error,
  });

  @override
  Widget build(BuildContext context) {
    if (starting) {
      return _statusBox(const CircularProgressIndicator(strokeWidth: 2), 'Starting camera');
    }
    if (error != null || controller == null) {
      return _statusBox(
        const Icon(Icons.videocam_off_outlined, color: AppColors.alertRed, size: 20),
        error ?? 'Camera unavailable',
      );
    }
    return Container(
      width: 150,
      height: 100,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), color: Colors.black),
      child: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller!),
          Positioned(
            left: 8,
            bottom: 7,
            child: DecoratedBox(
              decoration: BoxDecoration(color: Colors.black.withOpacity(0.65), borderRadius: BorderRadius.circular(10)),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.fiber_manual_record, size: 9, color: recording ? AppColors.alertRed : Colors.white),
                    const SizedBox(width: 4),
                    Text(recording ? 'Recording' : 'Camera ready', style: const TextStyle(color: Colors.white, fontSize: 10)),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusBox(Widget icon, String label) {
    return Container(
      constraints: const BoxConstraints(maxWidth: 150),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 7),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          icon,
          const SizedBox(width: 6),
          Flexible(child: Text(label, style: const TextStyle(fontSize: 10))),
        ],
      ),
    );
  }
}

class _CanvasPainter extends CustomPainter {
  final List<_Stroke> strokes;
  _CanvasPainter(this.strokes);

  @override
  void paint(Canvas canvas, Size size) {
    for (final stroke in strokes) {
      final paint = Paint()
        ..color = stroke.color
        ..strokeWidth = stroke.width
        ..strokeCap = StrokeCap.round
        ..style = PaintingStyle.stroke;
      for (int i = 0; i < stroke.points.length - 1; i++) {
        canvas.drawLine(stroke.points[i], stroke.points[i + 1], paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _CanvasPainter oldDelegate) => true;
}

/// Draws a simple coloring-book style outline for the child to copy.
class _ReferencePainter extends CustomPainter {
  final _RefPicture picture;
  _ReferencePainter(this.picture);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppColors.primaryDark
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;
    final w = size.width, h = size.height;

    switch (picture) {
      case _RefPicture.house:
        final path = Path()
          ..moveTo(w * 0.15, h * 0.95)
          ..lineTo(w * 0.15, h * 0.5)
          ..lineTo(w * 0.5, h * 0.15)
          ..lineTo(w * 0.85, h * 0.5)
          ..lineTo(w * 0.85, h * 0.95)
          ..close();
        canvas.drawPath(path, paint);
        canvas.drawRect(Rect.fromLTWH(w * 0.42, h * 0.6, w * 0.16, h * 0.35), paint);
        canvas.drawRect(Rect.fromLTWH(w * 0.22, h * 0.6, w * 0.14, h * 0.14), paint);
        break;
      case _RefPicture.star:
        final path = Path();
        const points = 5;
        final outerR = min(w, h) * 0.42;
        final innerR = outerR * 0.45;
        final cx = w / 2, cy = h / 2;
        for (int i = 0; i < points * 2; i++) {
          final r = i.isEven ? outerR : innerR;
          final angle = (pi / points) * i - pi / 2;
          final x = cx + r * cos(angle);
          final y = cy + r * sin(angle);
          if (i == 0) {
            path.moveTo(x, y);
          } else {
            path.lineTo(x, y);
          }
        }
        path.close();
        canvas.drawPath(path, paint);
        break;
      case _RefPicture.sun:
        final cx = w / 2, cy = h / 2;
        final r = min(w, h) * 0.22;
        canvas.drawCircle(Offset(cx, cy), r, paint);
        for (int i = 0; i < 8; i++) {
          final angle = (pi / 4) * i;
          final start = Offset(cx + r * 1.25 * cos(angle), cy + r * 1.25 * sin(angle));
          final end = Offset(cx + r * 1.7 * cos(angle), cy + r * 1.7 * sin(angle));
          canvas.drawLine(start, end, paint);
        }
        break;
      case _RefPicture.flower:
        final cx = w / 2, cy = h / 2;
        final petalR = min(w, h) * 0.16;
        for (int i = 0; i < 6; i++) {
          final angle = (pi / 3) * i;
          final petalCenter = Offset(cx + petalR * 1.1 * cos(angle), cy + petalR * 1.1 * sin(angle));
          canvas.drawCircle(petalCenter, petalR, paint);
        }
        canvas.drawCircle(Offset(cx, cy), petalR * 0.7, paint);
        canvas.drawLine(Offset(cx, cy + petalR * 1.6), Offset(cx, h * 0.95), paint);
        break;
      case _RefPicture.butterfly:
        final cx = w / 2, cy = h / 2;
        canvas.drawLine(Offset(cx, h * 0.15), Offset(cx, h * 0.9), paint);
        final leftWingTop = Path()
          ..moveTo(cx, cy * 0.75)
          ..quadraticBezierTo(w * 0.05, h * 0.1, w * 0.1, h * 0.45)
          ..quadraticBezierTo(w * 0.15, h * 0.65, cx, cy * 0.95)
          ..close();
        final rightWingTop = Path()
          ..moveTo(cx, cy * 0.75)
          ..quadraticBezierTo(w * 0.95, h * 0.1, w * 0.9, h * 0.45)
          ..quadraticBezierTo(w * 0.85, h * 0.65, cx, cy * 0.95)
          ..close();
        final leftWingBottom = Path()
          ..moveTo(cx, cy * 1.05)
          ..quadraticBezierTo(w * 0.1, h * 0.75, w * 0.2, h * 0.95)
          ..quadraticBezierTo(w * 0.3, h * 1.0, cx, cy * 1.2)
          ..close();
        final rightWingBottom = Path()
          ..moveTo(cx, cy * 1.05)
          ..quadraticBezierTo(w * 0.9, h * 0.75, w * 0.8, h * 0.95)
          ..quadraticBezierTo(w * 0.7, h * 1.0, cx, cy * 1.2)
          ..close();
        canvas.drawPath(leftWingTop, paint);
        canvas.drawPath(rightWingTop, paint);
        canvas.drawPath(leftWingBottom, paint);
        canvas.drawPath(rightWingBottom, paint);
        break;
    }
  }

  @override
  bool shouldRepaint(covariant _ReferencePainter oldDelegate) => oldDelegate.picture != picture;
}
