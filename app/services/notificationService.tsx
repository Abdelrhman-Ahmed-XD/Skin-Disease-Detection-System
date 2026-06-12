import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform } from 'react-native';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../../Firebase/firebaseConfig';

// ─── إعداد طريقة عرض الإشعار لما يكون التطبيق شغال ───────────────────────────
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert:  true,   // للإصدارات القديمة
    shouldShowBanner: true,   // للإصدارات الجديدة
    shouldShowList:   true,   // للإصدارات الجديدة
    shouldPlaySound:  true,
    shouldSetBadge:   true,
  }),
});

// ─── طلب الإذن وجيب الـ token ─────────────────────────────────────────────────
export async function registerForPushNotifications(): Promise<string | null> {
  if (!Device.isDevice) {
    console.log('Push notifications only work on real devices.');
    return null;
  }

  const { status: existing } = await Notifications.getPermissionsAsync();
  let finalStatus = existing;

  if (existing !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  if (finalStatus !== 'granted') {
    console.log('Notification permission not granted.');
    return null;
  }

  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'SkinSight',
      importance: Notifications.AndroidImportance.MAX,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#00A3A3',
      sound: 'default',
    });
  }

  try {
    const token = (await Notifications.getExpoPushTokenAsync()).data;
    console.log('✅ Push token:', token);
    return token;
  } catch (e) {
    console.log('⚠️ Could not get push token:', e);
    return null;
  }
}

// ─── إرسال إشعار محلي + حفظه في Firestore ────────────────────────────────────
async function sendLocalAndFirestore(
  title: string,
  body: string,
  type: 'scan_result' | 'system',
  extra?: Record<string, any>,
) {
  // 1. إشعار محلي يظهر فوراً حتى لو التطبيق في الخلفية
  await Notifications.scheduleNotificationAsync({
    content: {
      title,
      body,
      sound: true,
      data: { type, ...extra },
    },
    trigger: null, // فوري
  });

  // 2. حفظ في Firestore عشان يظهر في صفحة الإشعارات جوه التطبيق
  const user = auth.currentUser;
  if (!user) return;

  try {
    await addDoc(collection(db, 'users', user.uid, 'notifications'), {
      type,
      title,
      message: body,
      isRead:    false,
      timestamp: serverTimestamp(),
      ...(extra ?? {}),
    });
  } catch (e) {
    console.log('⚠️ Firestore notification save error:', e);
  }
}

// ─── الدوال الجاهزة ────────────────────────────────────────────────────────────

/** بعد تحليل الصورة بالـ AI */
export const notifyScanComplete = (
  disease: string,
  confidence: number,
  imageUri?: string,
) =>
  sendLocalAndFirestore(
    '🔬 نتيجة الفحص جاهزة',
    `تم اكتشاف: ${disease} — نسبة الثقة ${confidence}%`,
    'scan_result',
    { disease, confidence, imageUri: imageUri ?? '' },
  );

/** بعد تسجيل الدخول بنجاح */
export const notifyLoginSuccess = (name: string) =>
  sendLocalAndFirestore(
    '👋 مرحباً بك في SkinSight',
    `أهلاً ${name}! تم تسجيل الدخول بنجاح`,
    'system',
  );

/** بعد تحديث الملف الشخصي */
export const notifyProfileUpdated = () =>
  sendLocalAndFirestore(
    '✅ تم تحديث الملف الشخصي',
    'تم حفظ بياناتك الشخصية بنجاح',
    'system',
  );

/** بعد تغيير كلمة المرور */
export const notifyPasswordChanged = () =>
  sendLocalAndFirestore(
    '🔐 تم تغيير كلمة المرور',
    'تم تحديث كلمة المرور الخاصة بك بنجاح',
    'system',
  );

/** لما يغير الوضع الداكن / الفاتح */
export const notifyThemeChanged = (isDark: boolean) =>
  sendLocalAndFirestore(
    '🎨 تم تغيير مظهر التطبيق',
    isDark ? 'تم تفعيل الوضع الداكن 🌙' : 'تم تفعيل الوضع الفاتح ☀️',
    'system',
  );

/** بعد حفظ إعدادات التخصيص */
export const notifyCustomizeChanged = () =>
  sendLocalAndFirestore(
    '🎨 تم تحديث التخصيص',
    'تم تطبيق إعداداتك الجديدة بنجاح',
    'system',
  );

/** بعد تحميل تقرير PDF */
export const notifyFileDownloaded = (reportNum: number) =>
  sendLocalAndFirestore(
    '📄 تم تحميل التقرير',
    `تم حفظ التقرير رقم ${reportNum} على جهازك بنجاح`,
    'system',
  );

/** بعد رفع صورة من المعرض */
export const notifyPhotoUploaded = () =>
  sendLocalAndFirestore(
    '📸 تم رفع الصورة',
    'تم رفع الصورة وجاري التحليل...',
    'system',
  );

/** بعد إنشاء الحساب */
export const notifyAccountCreated = (name: string) =>
  sendLocalAndFirestore(
    '🎉 مرحباً في SkinSight!',
    `تم إنشاء حسابك بنجاح يا ${name}. ابدأ أول فحص لك الآن!`,
    'system',
  );