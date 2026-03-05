import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useState } from 'react';
import {
  LayoutAnimation, Platform, SafeAreaView, ScrollView,
  StatusBar, StyleSheet, Text, TouchableOpacity, UIManager, View,
} from 'react-native';
import { useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';

if (Platform.OS === 'android') {
  UIManager.setLayoutAnimationEnabledExperimental &&
    UIManager.setLayoutAnimationEnabledExperimental(true);
}

const faqsEn = [
  { question: "I can't log in or \"Sign In\" fails — what to do?", answers: ['Double-check your email and password.', 'Ensure you have Internet access.', 'If you forgot your password, use the reset option.', 'If it still fails, contact support.'] },
  { question: 'How do I create an account?', answers: ['Download the app and open it.', 'Tap "Sign Up" on the welcome screen.', 'Enter your name, email, and a strong password.', 'Verify your email address via the link sent to you.', 'Complete your profile and start using the app.'] },
  { question: 'I forgot my password. How can I reset it?', answers: ['Tap "Forgot Password?" on the login screen.', 'Enter your registered email address.', 'Check your inbox for a password reset link.', 'Follow the link and set a new password.', "If the email doesn't arrive, check your spam folder."] },
  { question: 'Why do I need to allow permissions (camera / storage / notifications)?', answers: ['Camera permission is required to capture skin images for diagnosis.', 'Storage permission lets you upload images from your gallery.', 'Notification permission sends you appointment and follow-up reminders.', 'All permissions are used solely within the app and not shared.'] },
  { question: 'I want to delete my account / data. How do I do that?', answers: ['Go to Settings from the main menu.', 'Scroll down and tap "Delete Account".', 'Confirm your identity by entering your password.', 'Your account and all associated data will be permanently deleted.', 'This action cannot be undone.'] },
  { question: "I didn't receive the verification email / SMS?", answers: ['Check your spam or junk mail folder.', 'Make sure you entered the correct email/phone number.', 'Tap "Resend Verification" on the confirmation screen.', 'Wait a few minutes and try again.', 'If the issue persists, contact our support team.'] },
  { question: 'The app crashes / freezes at login screen?', answers: ['Close the app fully and reopen it.', 'Make sure your app is updated to the latest version.', 'Restart your device.', 'Check that your internet connection is stable.', 'Reinstall the app if the problem continues.'] },
  { question: 'How accurate is the AI skin diagnosis?', answers: ['The AI model is trained on millions of classified skin images.', 'It provides results with a confidence percentage for each diagnosis.', 'Accuracy can be affected by image quality and lighting.', 'Always consult a dermatologist for a final diagnosis.', 'The app is a supportive tool, not a medical replacement.'] },
  { question: 'Is my medical data safe and private?', answers: ['All data is encrypted and stored on secure servers.', 'Your images and results are never sold or shared without consent.', 'The app complies with HIPAA and international health data laws.', 'You can request full data deletion at any time.'] },
  { question: 'How do I consult a doctor through the app?', answers: ['Complete a skin scan to get your initial diagnosis.', 'Tap "Consult a Doctor" on the results screen.', 'Choose an available dermatologist from the list.', 'Share your diagnosis report and describe your symptoms.', 'Receive guidance directly through the in-app chat.'] },
];

const faqsAr = [
  { question: 'لا أستطيع تسجيل الدخول — ماذا أفعل؟', answers: ['تحقق من صحة بريدك الإلكتروني وكلمة المرور.', 'تأكد من أن لديك اتصالاً بالإنترنت.', 'إذا نسيت كلمة المرور، استخدم خيار إعادة التعيين.', 'إذا استمرت المشكلة، تواصل مع الدعم الفني.'] },
  { question: 'كيف أنشئ حساباً جديداً؟', answers: ['حمّل التطبيق وافتحه.', 'انقر على "إنشاء حساب" في شاشة الترحيب.', 'أدخل اسمك وبريدك الإلكتروني وكلمة مرور قوية.', 'تحقق من بريدك الإلكتروني عبر الرابط المرسل إليك.', 'أكمل ملفك الشخصي وابدأ استخدام التطبيق.'] },
  { question: 'نسيت كلمة المرور، كيف أعيد تعيينها؟', answers: ['انقر على "نسيت كلمة المرور؟" في شاشة تسجيل الدخول.', 'أدخل بريدك الإلكتروني المسجل.', 'تحقق من صندوق الوارد للحصول على رابط إعادة التعيين.', 'اتبع الرابط وعيّن كلمة مرور جديدة.', 'إذا لم يصل البريد، تحقق من مجلد الرسائل غير المرغوب فيها.'] },
  { question: 'لماذا أحتاج للسماح بالأذونات (الكاميرا / التخزين / الإشعارات)؟', answers: ['إذن الكاميرا مطلوب لالتقاط صور الجلد للتشخيص.', 'إذن التخزين يتيح لك رفع الصور من معرضك.', 'إذن الإشعارات يرسل لك تذكيرات المواعيد والمتابعة.', 'جميع الأذونات تُستخدم داخل التطبيق فقط ولا تُشارك.'] },
  { question: 'أريد حذف حسابي / بياناتي. كيف أفعل ذلك؟', answers: ['اذهب إلى الإعدادات من القائمة الرئيسية.', 'مرر لأسفل وانقر على "حذف الحساب".', 'أكد هويتك بإدخال كلمة المرور.', 'سيتم حذف حسابك وجميع البيانات المرتبطة به نهائياً.', 'هذا الإجراء لا يمكن التراجع عنه.'] },
  { question: 'لم أتلق بريد التحقق / الرسالة النصية؟', answers: ['تحقق من مجلد البريد غير المرغوب فيه.', 'تأكد من إدخال البريد الإلكتروني / رقم الهاتف الصحيح.', 'انقر على "إعادة إرسال التحقق" في شاشة التأكيد.', 'انتظر بضع دقائق وحاول مرة أخرى.', 'إذا استمرت المشكلة، تواصل مع فريق الدعم.'] },
  { question: 'التطبيق يتعطل / يتجمد عند شاشة تسجيل الدخول؟', answers: ['أغلق التطبيق تماماً وأعد فتحه.', 'تأكد من أن التطبيق محدّث لآخر إصدار.', 'أعد تشغيل جهازك.', 'تحقق من استقرار اتصالك بالإنترنت.', 'أعد تثبيت التطبيق إذا استمرت المشكلة.'] },
  { question: 'ما مدى دقة تشخيص الذكاء الاصطناعي للجلد؟', answers: ['يتم تدريب نموذج الذكاء الاصطناعي على ملايين صور الجلد المصنفة.', 'يقدم نتائج مع نسبة ثقة لكل تشخيص.', 'قد تتأثر الدقة بجودة الصورة والإضاءة.', 'استشر دائماً طبيب أمراض جلدية للتشخيص النهائي.', 'التطبيق أداة دعم وليس بديلاً طبياً.'] },
  { question: 'هل بياناتي الطبية آمنة وخاصة؟', answers: ['جميع البيانات مشفرة ومخزنة على خوادم آمنة.', 'صورك ونتائجك لا تُباع أو تُشارك بدون موافقتك.', 'التطبيق يمتثل لقانون HIPAA وقوانين حماية البيانات الدولية.', 'يمكنك طلب حذف بياناتك في أي وقت.'] },
  { question: 'كيف أستشير طبيباً عبر التطبيق؟', answers: ['أكمل فحص الجلد للحصول على تشخيصك الأولي.', 'انقر على "استشر طبيباً" في شاشة النتائج.', 'اختر طبيب أمراض جلدية متاح من القائمة.', 'شارك تقرير تشخيصك وصف أعراضك.', 'تلقَّ التوجيه مباشرة عبر الدردشة داخل التطبيق.'] },
];

export default function HelpPage() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();
  const { t, isArabic } = useTranslation(settings.language);
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const customText = {
    fontSize:   settings.fontSize,
    color:      settings.textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  };

  // ✅ الـ background بيجي من settings.backgroundColor لما مش dark، وفي dark بيجي colors.background
  const pageBg = isDark ? colors.background : settings.backgroundColor;

  const faqs = isArabic ? faqsAr : faqsEn;

  const toggle = (index: number) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setOpenIndex(openIndex === index ? null : index);
  };

  const accentColor = isDark ? '#4BA3C7' : '#2A7DA0';

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name={isArabic ? "chevron-back" : "chevron-back"} size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, customText]}>{t('help')}</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={[styles.subtitle, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('faqSubtitle')}</Text>

        {faqs.map((faq, index) => {
          const isOpen = openIndex === index;
          return (
            <View key={index} style={[styles.card, { backgroundColor: colors.card }, isOpen && { borderColor: accentColor }]}>
              <TouchableOpacity
                style={[styles.questionRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                onPress={() => toggle(index)}
                activeOpacity={0.7}
              >
                <Text style={[styles.questionText, customText, { color: isOpen ? accentColor : settings.textColor, textAlign: isArabic ? 'right' : 'left' }]}>
                  {faq.question}
                </Text>
                <View style={[styles.arrowWrapper, { borderColor: accentColor }, isOpen && { backgroundColor: accentColor }]}>
                  <Ionicons name={isOpen ? 'chevron-down' : (isArabic ? 'chevron-back' : 'chevron-forward')} size={16} color={isOpen ? '#FFFFFF' : accentColor} />
                </View>
              </TouchableOpacity>

              {isOpen && (
                <View style={styles.answersContainer}>
                  <View style={[styles.divider, { backgroundColor: isDark ? '#2A3F50' : '#E5F0F6' }]} />
                  {faq.answers.map((answer, i) => (
                    <View key={i} style={[styles.answerRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                      <Text style={[styles.bullet, { color: accentColor }]}>←</Text>
                      <Text style={[styles.answerText, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{answer}</Text>
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