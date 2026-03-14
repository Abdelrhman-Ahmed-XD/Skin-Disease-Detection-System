import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React from 'react';
import { SafeAreaView, ScrollView, StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useCustomize } from '../Customize/Customizecontext';
import { useTheme } from '../ThemeContext';

export default function AboutPage() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();

  // ── أبيض في الدارك مود، أسود في الليت مود ─────────────────
  const textColor = isDark ? '#FFFFFF' : (settings.textColor || '#1F2937');

  const customText = {
    fontSize:   settings.fontSize,
    color:      textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  };

  const pageBg = isDark ? colors.background : settings.backgroundColor;

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={24} color={textColor} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, customText]}>{('aboutUs')}</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={[styles.mainTitle, customText, { textAlign:'center' }]}>{('appName')}</Text>
        <Text style={[styles.mainSubtitle, { color: '#2A7DA0', textAlign:'center', fontSize: settings.fontSize, fontFamily: customText.fontFamily }]}>{('usingAI')}</Text>
        <View style={[styles.divider, { backgroundColor: '#2A7DA0' }]} />

        <Section title={'Introduction'} colors={colors}  customText={customText}>
          In the age of advanced technology and artificial intelligence, we can now benefit from these modern technologies in the medical field in unprecedented ways. The Skin Disease Detection App is one of the most prominent smart medical applications that aims to help both patients and doctors diagnose skin diseases quickly and with high accuracy.
        </Section>

        <Section title={'App Concept'} colors={colors}  customText={customText}>
          The basic idea of the application is to enable the user to take a picture of the affected skin area using their mobile phone camera. The application then automatically analyzes this image and identifies the type of skin disease likely present, providing a comprehensive report that includes the name of the likely disease and its confidence level.
        </Section>

        <Section title={'App Objectives'} colors={colors}  customText={customText}>
        The application seeks to achieve key objectives: providing easy access to primary medical diagnostic services, reducing costs of routine medical visits, and raising users awareness of common skin diseases. The app helps doctors obtain quick second opinions while always emphasizing the necessity of consulting a specialist for a final diagnosis.
        </Section>

        <Section title={'Target Audiences'} colors={colors}  customText={customText}>
          The app targets patients with skin problems, general practitioners, dermatologists, pharmacists, and nurses who frequently encounter patients questions about skin diseases.
        </Section>

        <Text style={[styles.sectionTitle, customText, { textAlign:'left' }]}>{('keyFeatures')}</Text>

        <FeatureCard icon="camera-outline" title={'Smart Diagnosis with Images'} description="Take a picture directly with the camera or select one from your gallery. AI algorithms process the image within seconds, providing diagnostic results with a confidence level." colors={colors} isDark={isDark}  customText={customText} />
        <FeatureCard icon="library-outline" title={'Comprehensive Medical Database'} description='The application contains a rich database with detailed information on more than one hundred skin diseases, including causes, symptoms, complications, and treatment methods.' colors={colors} isDark={isDark}  customText={customText} />
        <FeatureCard icon="trending-up-outline" title={'Monitoring Condition Progress'} description= 'Document your skin condition over time, store multiple images of the affected area, and track its response to treatment with reminders for follow-up appointments.' colors={colors} isDark={isDark}  customText={customText} />
        <FeatureCard icon="people-outline" title={'Connecting with Doctors'} description='The application provides a platform for direct communication with dermatologists through an online medical consultation system.'colors={colors} isDark={isDark}  customText={customText} />

        <Section title={'Privacy and Security'} colors={colors}  customText={customText}>
            All images and medical data are stored encrypted on secure servers and are not shared with any third parties without explicit consent. The application complies with international laws related to health data protection, such as HIPAA.
        </Section>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children, colors, customText }: { title: string; children: React.ReactNode; colors: any; customText: any }) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
      <Text style={[styles.sectionTitle, customText, { textAlign:'left' }]}>{title}</Text>
      <Text style={[styles.sectionBody, customText, { textAlign:'left' }]}>{children}</Text>
    </View>
  );
}

function FeatureCard({ icon, title, description, colors, isDark, customText }: { icon: string; title: string; description: string; colors: any; isDark: boolean; customText: any }) {
  return (
    <View style={[styles.featureCard, { backgroundColor: colors.card, flexDirection: 'row' }]}>
      <View style={[styles.featureIconWrapper, { backgroundColor: isDark ? '#1A3040' : '#E8F4FA' }]}>
        <Ionicons name={icon as any} size={22} color="#2A7DA0" />
      </View>
      <View style={styles.featureTextWrapper}>
        <Text style={[styles.featureTitle, customText, { textAlign:'left' }]}>{title}</Text>
        <Text style={[styles.featureDescription, customText, { textAlign:'left' }]}>{description}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container:          { flex: 1 },
  header:             { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:         { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:        { fontSize: 20, fontWeight: 'bold' },
  scrollView:         { flex: 1 },
  scrollContent:      { paddingHorizontal: 16, paddingBottom: 20 },
  mainTitle:          { fontSize: 22, fontWeight: '800', textAlign: 'center', marginTop: 8 },
  mainSubtitle:       { fontSize: 15, fontWeight: '500', textAlign: 'center', marginTop: 4 },
  divider:            { height: 2, borderRadius: 2, marginVertical: 16, opacity: 0.3 },
  section:            { marginBottom: 20, borderRadius: 14, padding: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1 },
  sectionTitle:       { fontSize: 16, fontWeight: '700', marginBottom: 10, marginTop: 4 },
  sectionBody:        { fontSize: 14, lineHeight: 22 },
  featureCard:        { flexDirection: 'row', borderRadius: 14, padding: 14, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, alignItems: 'flex-start' },
  featureIconWrapper: { width: 42, height: 42, borderRadius: 11, alignItems: 'center', justifyContent: 'center', marginRight: 12, flexShrink: 0 },
  featureTextWrapper: { flex: 1 },
  featureTitle:       { fontSize: 14, fontWeight: '700', marginBottom: 4 },
  featureDescription: { fontSize: 13, lineHeight: 20 },
});