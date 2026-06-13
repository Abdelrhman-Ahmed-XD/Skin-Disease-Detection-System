import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useState } from 'react';
import {
  LayoutAnimation, Platform, SafeAreaView, ScrollView,
  StyleSheet, Text, TouchableOpacity, UIManager, View,
} from 'react-native';
import { useTheme } from '../ThemeContext';

if (Platform.OS === 'android') {
  UIManager.setLayoutAnimationEnabledExperimental &&
    UIManager.setLayoutAnimationEnabledExperimental(true);
}

const faqs = [
  { question: "I can't log in or Sign In fails. What to do?", answers: ['Check your email and password carefully.', 'Ensure you have Internet access.', 'If you forgot your password, use the reset option on the login screen.', 'If it still fails, try reinstalling the app.'] },
  { question: 'How do I create an account?', answers: ['Open the app and tap "Sign Up" on the welcome screen.', 'Enter your name, email, and a strong password.', 'Verify your email via the OTP code sent to you.', 'Complete your profile (age, gender, skin type) and start using the app.'] },
  { question: 'I forgot my password. How can I reset it?', answers: ['Tap "Forgot Password?" on the login screen.', 'Enter your registered email address.', 'Check your inbox for a password reset link.', 'Follow the link and set a new password.', "If the email doesn't arrive, check your spam folder."] },
  { question: 'How does the AI skin analysis work?', answers: ['Take or upload a clear photo of the affected skin area.', 'The app sends it to an ensemble of five AI models for analysis.', 'Each model votes on the most likely condition.', 'You receive a result with a confidence percentage and condition details.', 'The AI is a screening tool. Always follow up with a dermatologist.'] },
  { question: 'How accurate is the AI diagnosis?', answers: ['SkinSight uses five specialized models: ConvNeXt-Base, DenseNet121, MaxViT-T, ResNeXt50, and U-Net++ with EfficientNet-B4.', 'Results include a confidence score. Higher means more certainty.', 'Accuracy is affected by photo quality and lighting.', 'Always consult a licensed dermatologist for a definitive diagnosis.'] },
  { question: 'What is the body map?', answers: ['The body map (available to signed-in users) lets you pin skin areas you want to monitor.', 'Each pin can have a photo attached via a scan.', 'Pins are saved to your account so you can track changes over time.', 'Create a free account to access the body map.'] },
  { question: 'What is the confidence score?', answers: ['The confidence score shows how certain the AI is about its prediction.', 'A high score (e.g. 90%+) means the models strongly agree.', 'A lower score means results are less definitive. Seek professional advice.', 'Confidence is affected by image quality, lighting, and visibility of the condition.'] },
  { question: 'Is my data safe and private?', answers: ['All account data is stored securely in Firebase.', 'Images and results are never sold or shared with third parties.', 'You can delete your account and all associated data at any time.'] },
  { question: 'Why do I only get one free guest scan?', answers: ['The free guest scan lets you try the AI analysis without creating an account.', 'Creating a free account gives you unlimited scans, full history, and the body map.', 'Sign up is free and takes under two minutes.'] },
  { question: 'The app crashes or freezes. What do I do?', answers: ['Close the app fully and reopen it.', 'Make sure your app is updated to the latest version.', 'Restart your device.', 'Reinstall the app if the problem continues.'] },
];

export default function HelpGuestPage() {
  const { colors, isDark } = useTheme();
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const pageBg     = isDark ? colors.background : '#D8E9F0';
  const accentColor = '#4BA3C7';

  const toggle = (index: number) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={24} color={isDark ? '#FFFFFF' : '#1F2937'} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: isDark ? '#fff' : '#000' }]}>Help</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={[styles.subtitle, { color: isDark ? '#fff' : '#374151' }]}>
          Frequently asked questions about SkinSight
        </Text>

        {faqs.map((faq, index) => {
          const isOpen = openIndex === index;
          return (
            <View
              key={index}
              style={[styles.card, { backgroundColor: colors.card }, isOpen && { borderColor: accentColor }]}
            >
              <TouchableOpacity
                style={styles.questionRow}
                onPress={() => toggle(index)}
                activeOpacity={0.7}
              >
                <Text style={[styles.questionText, { color: isOpen ? accentColor : isDark ? '#FFFFFF' : '#1F2937' }]}>
                  {faq.question}
                </Text>
                <View style={[styles.arrowWrapper, { borderColor: accentColor }, isOpen && { backgroundColor: accentColor }]}>
                  <Ionicons
                    name={isOpen ? 'chevron-down' : 'chevron-forward'}
                    size={16}
                    color={isOpen ? '#FFFFFF' : accentColor}
                  />
                </View>
              </TouchableOpacity>

              {isOpen && (
                <View style={styles.answersContainer}>
                  <View style={[styles.divider, { backgroundColor: isDark ? '#2A3F50' : '#E5F0F6' }]} />
                  {faq.answers.map((answer, i) => (
                    <View key={i} style={styles.answerRow}>
                      <Text style={[styles.bullet, { color: accentColor }]}>→</Text>
                      <Text style={[styles.answerText, { color: isDark ? '#FFFFFF' : '#1F2937' }]}>
                        {answer}
                      </Text>
                    </View>
                  ))}
                </View>
              )}
            </View>
          );
        })}

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:        { flex: 1 },
  header:           { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:       { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:      { fontSize: 22, fontWeight: 'bold' },
  scrollView:       { flex: 1 },
  scrollContent:    { paddingHorizontal: 15, paddingBottom: 20 },
  subtitle:         { fontSize: 13, marginBottom: 12, marginTop: 2, marginLeft: 2 },
  card:             { borderRadius: 14, marginBottom: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, borderWidth: 1, borderColor: 'transparent' },
  questionRow:      { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 16 },
  questionText:     { flex: 1, fontSize: 14, fontWeight: '500', lineHeight: 20, marginRight: 10 },
  arrowWrapper:     { width: 28, height: 28, borderRadius: 8, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  answersContainer: { paddingHorizontal: 16, paddingBottom: 14 },
  divider:          { height: 1, marginBottom: 12 },
  answerRow:        { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 7 },
  bullet:           { fontSize: 13, marginRight: 8, marginTop: 1, fontWeight: '600' },
  answerText:       { flex: 1, fontSize: 13, lineHeight: 19 },
});
