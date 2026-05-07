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
        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.mainTitle, customText, { textAlign: isArabic ? 'right' : 'center' }]}>{t('appName')}</Text>
        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.mainSubtitle, { color: '#2A7DA0', textAlign: isArabic ? 'right' : 'center', fontSize: settings.fontSize, fontFamily: customText.fontFamily }]}>{t('usingAI')}</Text>
        <View style={[styles.divider, { backgroundColor: '#2A7DA0' }]} />

        <Section title={isArabic ? 'مقدمة' : 'Introduction'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'في عصر التكنولوجيا المتقدمة والذكاء الاصطناعي، يمكننا الاستفادة من هذه التقنيات الحديثة في المجال الطبي بطرق غير مسبوقة. يُعد تطبيق كشف أمراض الجلد من أبرز التطبيقات الطبية الذكية التي تهدف إلى مساعدة المرضى والأطباء في تشخيص أمراض الجلد بسرعة وبدقة عالية.'
            : 'In the age of advanced technology and artificial intelligence, we can now benefit from these modern technologies in the medical field in unprecedented ways. The Skin Disease Detection App is one of the most prominent smart medical applications that aims to help both patients and doctors diagnose skin diseases quickly and with high accuracy.'}
        </Section>

        <Section title={isArabic ? 'فكرة التطبيق' : 'App Concept'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'الفكرة الأساسية للتطبيق هي تمكين المستخدم من التقاط صورة لمنطقة الجلد المصابة باستخدام كاميرا هاتفه المحمول. يقوم التطبيق بعد ذلك بتحليل هذه الصورة تلقائياً وتحديد نوع مرض الجلد المحتمل، وتقديم تقرير شامل يتضمن اسم المرض المحتمل ومستوى الثقة به.'
            : 'The basic idea of the application is to enable the user to take a picture of the affected skin area using their mobile phone camera. The application then automatically analyzes this image and identifies the type of skin disease likely present, providing a comprehensive report that includes the name of the likely disease and its confidence level.'}
        </Section>

        <Section title={isArabic ? 'أهداف التطبيق' : 'App Objectives'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'يسعى التطبيق إلى تحقيق أهداف رئيسية: توفير وصول سهل إلى خدمات التشخيص الطبي الأولية، وتقليل تكاليف الزيارات الطبية الروتينية، ورفع مستوى وعي المستخدمين بأمراض الجلد الشائعة. كما يساعد الأطباء على الحصول على آراء ثانوية سريعة مع التأكيد دائماً على ضرورة استشارة أخصائي للتشخيص النهائي.'
            : 'The application seeks to achieve key objectives: providing easy access to primary medical diagnostic services, reducing costs of routine medical visits, and raising users awareness of common skin diseases. The app helps doctors obtain quick second opinions while always emphasizing the necessity of consulting a specialist for a final diagnosis.'}
        </Section>

        <Section title={isArabic ? 'الفئات المستهدفة' : 'Target Audiences'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'يستهدف التطبيق المرضى الذين يعانون من مشاكل جلدية، والأطباء العامين، وأطباء الأمراض الجلدية، والصيادلة، والممرضين الذين يواجهون بشكل متكرر أسئلة المرضى حول أمراض الجلد.'
            : 'The app targets patients with skin problems, general practitioners, dermatologists, pharmacists, and nurses who frequently encounter patients questions about skin diseases.'}
        </Section>

        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.sectionTitle, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('keyFeatures')}</Text>

        <FeatureCard
          icon="camera-outline"
          title={isArabic ? 'التشخيص الذكي بالصور' : 'Smart Diagnosis with Images'}
          description={isArabic ? 'التقط صورة مباشرة بالكاميرا أو اختر من معرض صورك. تعالج خوارزميات الذكاء الاصطناعي الصورة في ثوانٍ، وتقدم نتائج تشخيصية مع مستوى الثقة.' : 'Take a picture directly with the camera or select one from your gallery. AI algorithms process the image within seconds, providing diagnostic results with a confidence level.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="library-outline"
          title={isArabic ? 'قاعدة بيانات طبية شاملة' : 'Comprehensive Medical Database'}
          description={isArabic ? 'يحتوي التطبيق على قاعدة بيانات غنية بمعلومات تفصيلية عن أكثر من مائة مرض جلدي، بما في ذلك الأسباب والأعراض والمضاعفات وطرق العلاج.' : 'The application contains a rich database with detailed information on more than one hundred skin diseases, including causes, symptoms, complications, and treatment methods.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="trending-up-outline"
          title={isArabic ? 'متابعة تطور الحالة' : 'Monitoring Condition Progress'}
          description={isArabic ? 'وثّق حالتك الجلدية بمرور الوقت، وخزّن صوراً متعددة للمنطقة المصابة، وتتبع استجابتها للعلاج مع تذكيرات لمواعيد المتابعة.' : 'Document your skin condition over time, store multiple images of the affected area, and track its response to treatment with reminders for follow-up appointments.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />
        <FeatureCard
          icon="people-outline"
          title={isArabic ? 'التواصل مع الأطباء' : 'Connecting with Doctors'}
          description={isArabic ? 'يوفر التطبيق منصة للتواصل المباشر مع أطباء الأمراض الجلدية من خلال نظام استشارات طبية عبر الإنترنت.' : 'The application provides a platform for direct communication with dermatologists through an online medical consultation system.'}
          colors={colors} isDark={isDark} isArabic={isArabic} customText={customText}
        />

        <Section title={isArabic ? 'الخصوصية والأمان' : 'Privacy and Security'} colors={colors} isArabic={isArabic} customText={customText}>
          {isArabic
            ? 'يتم تخزين جميع الصور والبيانات الطبية مشفرة على خوادم آمنة ولا تتم مشاركتها مع أي أطراف ثالثة دون موافقة صريحة. يمتثل التطبيق للقوانين الدولية المتعلقة بحماية بيانات الصحة، مثل قانون HIPAA.'
            : 'All images and medical data are stored encrypted on secure servers and are not shared with any third parties without explicit consent. The application complies with international laws related to health data protection, such as HIPAA.'}
        </Section>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children, colors, isArabic, customText }: { title: string; children: React.ReactNode; colors: any; isArabic: boolean; customText: any }) {
  return (
    <View style={[styles.section, { backgroundColor: colors.card }]}>
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
  headerTitle:        { fontSize: 20},
  scrollView:         { flex: 1 },
  scrollContent:      { paddingHorizontal: 16, paddingBottom: 20 },
  mainTitle:          { fontSize: 22,  textAlign: 'center', marginTop: 8 },
  mainSubtitle:       { fontSize: 15,  textAlign: 'center', marginTop: 4 },
  divider:            { height: 2, borderRadius: 2, marginVertical: 16, opacity: 0.3 },
  section:            { marginBottom: 20, borderRadius: 14, padding: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1 },
  sectionTitle:       { fontSize: 16,  marginBottom: 10, marginTop: 4 },
  sectionBody:        { fontSize: 14, lineHeight: 22 },
  featureCard:        { flexDirection: 'row', borderRadius: 14, padding: 14, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, alignItems: 'flex-start' },
  featureIconWrapper: { width: 42, height: 42, borderRadius: 11, alignItems: 'center', justifyContent: 'center', marginRight: 12, flexShrink: 0 },
  featureTextWrapper: { flex: 1 },
  featureTitle:       { fontSize: 20, marginBottom: 10 },
  featureDescription: { fontSize: 13, lineHeight: 20 },
});