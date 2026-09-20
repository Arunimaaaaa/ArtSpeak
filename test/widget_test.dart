// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:artspeak/main.dart';

void main() {
  testWidgets('Artspeak home screen renders', (WidgetTester tester) async {
    await tester.pumpWidget(const ArtspeakApp());

    expect(find.text('artspeak'), findsOneWidget);
    expect(find.text('Welcome to\nArtspeak.'), findsOneWidget);
    expect(find.text('Start session'), findsOneWidget);
    expect(find.text('Caregiver login'), findsOneWidget);
    expect(find.text('Child login'), findsOneWidget);
  });
}
