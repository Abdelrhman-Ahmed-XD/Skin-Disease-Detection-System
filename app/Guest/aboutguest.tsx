import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function AboutGuestPage() {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={24} color="#1F2937" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>About SkinSight</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Hero card */}
        <View style={styles.heroCard}>
          <Text style={styles.heroTitle}>SkinSight</Text>
          <View style={styles.heroDivider} />
          <Text style={styles.heroSubtitle}>AI Powered Skin Analysis</Text>
        </View>

        <Section title="Introduction">
          SkinSight is a smart app that uses artificial intelligence to analyze skin photos and detect potential skin conditions. It combines five specialized AI models to maximize diagnostic accuracy, and provides comprehensive results including a confidence score and detailed condition information.
        </Section>

        <Section title="How It Works">
          Capture a photo of the affected skin area or choose one from your gallery. The app sends it to an AI ensemble consisting of ConvNeXt-Base, DenseNet121, MaxViT-T, ResNeXt50, and U-Net++ with EfficientNet-B4. This combination improves accuracy and returns an instant diagnostic report with a confidence percentage.
        </Section>

        <Section title="App Objectives">
          SkinSight aims to provide accessible, preliminary AI powered analysis of skin conditions and raise user awareness. It is not a replacement for professional medical advice. It guides users toward making more informed decisions about their skin health.
        </Section>

        <Text style={styles.featuresHeading}>Key Features</Text>

        <FeatureCard
          icon="camera-outline"
          title="AI Skin Scan"
          description="Capture via camera or pick from your gallery. A five model AI ensemble analyzes the image and returns a diagnosis with a confidence score in seconds."
        />
        <FeatureCard
          icon="map-outline"
          title="Body Map"
          description="Pin locations on an interactive body map to mark skin areas you are monitoring. Zoom in, tap a pin to rescan, or long-press to remove it."
        />
        <FeatureCard
          icon="time-outline"
          title="Scan History"
          description="Every scan is saved with its photo, date, and results. Review your full history at any time to track how your condition changes."
        />
        <FeatureCard
          icon="shield-checkmark-outline"
          title="Privacy and Security"
          description="Your data is stored securely in Firebase and never shared with third parties. You can delete your account and all data at any time."
        />

        {/* Medical Disclaimer — warning card */}
        <View style={styles.disclaimerCard}>
          <View style={styles.disclaimerHeader}>
            <Ionicons name="warning-outline" size={20} color="#F59E0B" />
            <Text style={styles.disclaimerTitle}>Medical Disclaimer</Text>
          </View>
          <Text style={styles.disclaimerBody}>
            SkinSight is an awareness tool, not a substitute for professional medical diagnosis. AI results may not be accurate in all cases. Always consult a licensed dermatologist for a final diagnosis and treatment plan.
          </Text>
        </View>

        <View style={styles.guestNote}>
          <Ionicons name="information-circle-outline" size={18} color="#004F7F" />
          <Text style={styles.guestNoteText}>
            You are browsing as a guest. Create a free account to save scans, access history, and use the full body map.
          </Text>
        </View>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Text style={styles.sectionBody}>{children}</Text>
    </View>
  );
}

function FeatureCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <View style={styles.featureCard}>
      <View style={styles.featureIconWrapper}>
        <Ionicons name={icon as any} size={22} color="#2A7DA0" />
      </View>
      <View style={styles.featureTextWrapper}>
        <Text style={styles.featureTitle}>{title}</Text>
        <Text style={styles.featureDescription}>{description}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container:          { flex: 1, backgroundColor: '#D8E9F0' },
  header:             { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, backgroundColor: '#fff', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:         { width: 40, height: 40, borderRadius: 12, borderWidth: 1, borderColor: '#D1D5DB', alignItems: 'center', justifyContent: 'center' },
  headerTitle:        { fontSize: 20, fontWeight: 'bold', color: '#1F2937' },
  scrollView:         { flex: 1 },
  scrollContent:      { paddingHorizontal: 16, paddingBottom: 20 },
  heroCard:           { borderRadius: 16, backgroundColor: '#004F7F', padding: 22, alignItems: 'center', marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.15, shadowRadius: 8, elevation: 4 },
  heroTitle:          { fontSize: 26, fontWeight: '800', color: '#FFFFFF', textAlign: 'center', letterSpacing: 1 },
  heroDivider:        { width: 40, height: 3, backgroundColor: '#00A3A3', borderRadius: 2, marginVertical: 10 },
  heroSubtitle:       { fontSize: 14, color: '#C5E3ED', textAlign: 'center', fontStyle: 'italic' },
  section:            { marginBottom: 14, borderRadius: 14, padding: 16, backgroundColor: '#fff', borderLeftWidth: 3, borderLeftColor: '#2A7DA0', shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1 },
  sectionTitle:       { fontSize: 15, fontWeight: '700', marginBottom: 7, color: '#2A7DA0' },
  sectionBody:        { fontSize: 14, lineHeight: 22, color: '#374151' },
  featuresHeading:    { fontSize: 15, fontWeight: '700', color: '#1F2937', marginBottom: 10, marginTop: 6, paddingLeft: 4 },
  featureCard:        { flexDirection: 'row', borderRadius: 14, padding: 14, marginBottom: 10, backgroundColor: '#fff', shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, alignItems: 'flex-start' },
  featureIconWrapper: { width: 44, height: 44, borderRadius: 12, backgroundColor: '#E8F4FA', alignItems: 'center', justifyContent: 'center', marginRight: 12, flexShrink: 0 },
  featureTextWrapper: { flex: 1 },
  featureTitle:       { fontSize: 14, fontWeight: '700', marginBottom: 4, color: '#1F2937' },
  featureDescription: { fontSize: 13, lineHeight: 20, color: '#374151' },
  disclaimerCard:     { borderRadius: 14, borderWidth: 1.5, borderColor: '#F59E0B', backgroundColor: '#FFFBEB', padding: 16, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1 },
  disclaimerHeader:   { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  disclaimerTitle:    { fontSize: 15, fontWeight: '700', color: '#F59E0B', flex: 1 },
  disclaimerBody:     { fontSize: 14, lineHeight: 22, color: '#92400E' },
  guestNote:          { flexDirection: 'row', alignItems: 'flex-start', gap: 8, backgroundColor: '#E8F4FA', borderRadius: 12, padding: 14, marginTop: 4 },
  guestNoteText:      { flex: 1, fontSize: 13, lineHeight: 20, color: '#004F7F' },
});
