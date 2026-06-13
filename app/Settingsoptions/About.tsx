import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React from 'react';
import { SafeAreaView, ScrollView, StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';

export default function AboutPage() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();
  const { t, isArabic } = useTranslation(settings.language);

  const customText = {
    fontSize: settings.fontSize,
    color: isDark ? "#FFFFFF" : settings.textColor,
    fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
  };

  // ✅ background من settings في light mode
  const pageBg = isDark ? colors.background : settings.backgroundColor;

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name={isArabic ? "chevron-back" : "chevron-back"} size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.headerTitle, customText]}>{t('aboutUs')}</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Hero card */}
        <View style={[styles.heroCard, { backgroundColor: isDark ? '#0D2030' : '#004F7F' }]}>
          <Text style={styles.heroTitle}>{t('appName')}</Text>
          <View style={styles.heroDivider} />
          <Text style={styles.heroSubtitle}>{t('usingAI')}</Text>
        </View>

        <Section title={isArabic ? 'مقدمة' : 'Introduction'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'SkinSight تطبيق طبي ذكي يستخدم الذكاء الاصطناعي لتحليل صور الجلد واكتشاف الأمراض الجلدية المحتملة. يجمع التطبيق بين خمسة نماذج ذكاء اصطناعي متخصصة للحصول على أعلى دقة في التشخيص، مع تقديم نتائج شاملة تشمل نسبة الثقة ومعلومات عن الحالة.'
            : 'SkinSight is a smart medical app that uses artificial intelligence to analyze skin images and detect potential skin conditions. It combines five specialized AI models to maximize diagnostic accuracy, and provides comprehensive results including a confidence score and detailed condition information.'}
        </Section>

        <Section title={isArabic ? 'كيف يعمل التطبيق' : 'How It Works'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'التقط صورة لمنطقة الجلد المصابة أو اختر صورة من معرضك. يرسل التطبيق الصورة إلى نظام ذكاء اصطناعي متكامل يضم نماذج ConvNeXt-Base وDenseNet121 وMaxViT-T وResNeXt50 وU-Net++ مع EfficientNet-B4. يعمل هذا التجميع على تحسين الدقة، ويُعيد تقريراً تشخيصياً فورياً مع نسبة الثقة.'
            : 'Capture a photo of the affected skin area or choose one from your gallery. The app sends the image to an AI ensemble consisting of ConvNeXt-Base, DenseNet121, MaxViT-T, ResNeXt50, and U-Net++ with EfficientNet-B4. This combination improves accuracy and returns an instant diagnostic report with a confidence percentage.'}
        </Section>

        <Section title={isArabic ? 'أهداف التطبيق' : 'App Objectives'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'يهدف التطبيق إلى توفير وصول سهل ومبدئي لتحليل الحالات الجلدية، ورفع مستوى الوعي لدى المستخدمين. لا يُغني التطبيق عن استشارة طبيب متخصص، بل يُساعد في توجيه المستخدم نحو القرار الصحيح.'
            : 'The app aims to provide accessible, preliminary AI powered analysis of skin conditions and raise user awareness. It is not a replacement for professional medical advice. It guides users toward making more informed decisions.'}
        </Section>

        <Section title={isArabic ? 'الفئات المستهدفة' : 'Who Is It For?'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'يستهدف التطبيق أي شخص لديه مخاوف بشأن صحة جلده، سواء كنت تلاحظ تغيراً جلدياً وتريد فهماً أولياً، أو تبحث عن طريقة منظمة لتتبع حالتك الجلدية عبر الزمن.'
            : 'SkinSight is for anyone concerned about a skin change, whether you want a preliminary understanding of what you are seeing, or a structured way to track your skin health over time.'}
        </Section>

        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.sectionTitle, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('keyFeatures')}</Text>

        <FeatureCard
          icon="camera-outline"
          title={isArabic ? 'فحص الجلد بالذكاء الاصطناعي' : 'AI Skin Scan'}
          description={isArabic ? 'التقط صورة بالكاميرا أو اختر من المعرض. يعالجها نظام ذكاء اصطناعي متكامل من خمسة نماذج ويُعيد النتيجة مع نسبة الثقة خلال ثوانٍ.' : 'Capture via camera or pick from your gallery. A five model AI ensemble analyzes the image and returns a diagnosis with a confidence score in seconds.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="map-outline"
          title={isArabic ? 'خريطة الجسم' : 'Body Map'}
          description={isArabic ? 'ضع نقاطاً على خريطة الجسم لتحديد مناطق الجلد التي تراقبها. يمكنك تكبير الخريطة ومشاهدة الصور المرتبطة بكل نقطة.' : 'Pin locations on an interactive body map to mark skin areas you are monitoring. Zoom in, tap a pin to rescan, or long-press to remove it.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="time-outline"
          title={isArabic ? 'سجل الفحوصات' : 'Scan History'}
          description={isArabic ? 'يُسجّل التطبيق جميع فحوصاتك مع الصور والتواريخ والنتائج. راجع سجلك في أي وقت لمتابعة تطور حالتك.' : 'Every scan is saved with its photo, date, and results. Review your full history at any time to track how your condition changes.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="shield-checkmark-outline"
          title={isArabic ? 'الخصوصية والأمان' : 'Privacy & Security'}
          description={isArabic ? 'تُخزَّن بياناتك بأمان في Firebase ولا تُشارَك مع أطراف ثالثة. يمكنك حذف حسابك وبياناتك بالكامل في أي وقت.' : 'Your data is stored securely in Firebase and never shared with third parties. You can delete your account and all data at any time.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />

        {/* Medical Disclaimer — warning card */}
        <View style={[styles.disclaimerCard, { backgroundColor: isDark ? '#1A1200' : '#FFFBEB', borderColor: isDark ? '#92400E' : '#F59E0B' }]}>
          <View style={styles.disclaimerHeader}>
            <Ionicons name="warning-outline" size={20} color="#F59E0B" />
            <Text style={[styles.disclaimerTitle, { fontFamily: customText.fontFamily, textAlign: isArabic ? 'right' : 'left' }]}>
              {isArabic ? 'تنبيه طبي' : 'Medical Disclaimer'}
            </Text>
          </View>
          <Text style={[styles.disclaimerBody, customText, { textAlign: isArabic ? 'right' : 'left' }]}>
            {isArabic
              ? 'SkinSight أداة توعوية وليست بديلاً عن التشخيص الطبي المتخصص. النتائج المقدمة تعتمد على الذكاء الاصطناعي وقد لا تكون دقيقة في جميع الحالات. استشر طبيباً مختصاً دائماً للتشخيص النهائي والعلاج.'
              : 'SkinSight is an awareness tool, not a substitute for professional medical diagnosis. AI results may not be accurate in all cases. Always consult a licensed dermatologist for a final diagnosis and treatment plan.'}
          </Text>
        </View>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children, colors, isArabic, customText }: { title: string; children: React.ReactNode; colors: any; isArabic: boolean; customText: any }) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card, borderLeftColor: '#2A7DA0', borderLeftWidth: isArabic ? 0 : 3, borderRightColor: '#2A7DA0', borderRightWidth: isArabic ? 3 : 0 }]}>
      <Text style={[styles.sectionTitle, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{title}</Text>
      <Text style={[styles.sectionBody, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{children}</Text>
    </View>
  );
}

function FeatureCard({ icon, title, description, colors, isDark, isArabic, customText }: { icon: string; title: string; description: string; colors: any; isDark: boolean; isArabic: boolean; customText: any }) {
  return (
    <View style={[styles.featureCard, { backgroundColor: colors.card, flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
      <View style={[styles.featureIconWrapper, { backgroundColor: isDark ? '#1A3040' : '#E8F4FA' }]}>
        <Ionicons name={icon as any} size={22} color="#2A7DA0" />
      </View>
      <View style={styles.featureTextWrapper}>
        <Text style={[styles.featureTitle, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{title}</Text>
        <Text style={[styles.featureDescription, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{description}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container:          { flex: 1 },
  header:             { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:         { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:        { fontSize: 20 },
  scrollView:         { flex: 1 },
  scrollContent:      { paddingHorizontal: 16, paddingBottom: 20 },
  heroCard:           { borderRadius: 16, padding: 22, alignItems: 'center', marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.15, shadowRadius: 8, elevation: 4 },
  heroTitle:          { fontSize: 26, fontWeight: '800', color: '#FFFFFF', textAlign: 'center', letterSpacing: 1 },
  heroDivider:        { width: 40, height: 3, backgroundColor: '#00A3A3', borderRadius: 2, marginVertical: 10 },
  heroSubtitle:       { fontSize: 14, color: '#C5E3ED', textAlign: 'center', fontStyle: 'italic' },
  section:            { marginBottom: 16, borderRadius: 14, padding: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, overflow: 'hidden' },
  sectionTitle:       { fontSize: 16, fontWeight: '700', marginBottom: 8, marginTop: 2, color: '#2A7DA0' },
  sectionBody:        { fontSize: 14, lineHeight: 22 },
  featureCard:        { flexDirection: 'row', borderRadius: 14, padding: 14, marginBottom: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, alignItems: 'flex-start' },
  featureIconWrapper: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center', marginRight: 12, flexShrink: 0 },
  featureTextWrapper: { flex: 1 },
  featureTitle:       { fontSize: 14, fontWeight: '700', marginBottom: 4 },
  featureDescription: { fontSize: 13, lineHeight: 20 },
  disclaimerCard:     { borderRadius: 14, borderWidth: 1.5, padding: 16, marginBottom: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1 },
  disclaimerHeader:   { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  disclaimerTitle:    { fontSize: 15, fontWeight: '700', color: '#F59E0B', flex: 1 },
  disclaimerBody:     { fontSize: 14, lineHeight: 22 },
});