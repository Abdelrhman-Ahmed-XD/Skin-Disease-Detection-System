import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../../Firebase/firebaseConfig';

const NOTIFICATIONS_ENABLED_KEY = 'notificationsEnabled';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert:  true,
    shouldShowBanner: true,
    shouldShowList:   true,
    shouldPlaySound:  true,
    shouldSetBadge:   true,
  }),
});

export async function registerForPushNotifications(): Promise<string | null> {
  if (!Device.isDevice) return null;

  const { status: existing } = await Notifications.getPermissionsAsync();
  let finalStatus = existing;

  if (existing !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  if (finalStatus !== 'granted') return null;

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
    return token;
  } catch {
    return null;
  }
}

async function isEnabled(): Promise<boolean> {
  const val = await AsyncStorage.getItem(NOTIFICATIONS_ENABLED_KEY);
  return val === null ? true : val === 'true';
}

async function sendLocalAndFirestore(
  title: string,
  body: string,
  type: 'scan_result' | 'system',
  extra?: Record<string, any>,
) {
  if (!(await isEnabled())) return;

  await Notifications.scheduleNotificationAsync({
    content: { title, body, sound: true, data: { type, ...extra } },
    trigger: null,
  });

  const user = auth.currentUser;
  if (!user) return;

  try {
    await addDoc(collection(db, 'users', user.uid, 'notifications'), {
      type,
      title,
      message: body,
      isRead: false,
      timestamp: serverTimestamp(),
      ...(extra ?? {}),
    });
  } catch {
    // Firestore write is best-effort
  }
}

/** Used by Camera.tsx which already saves its own Firestore entry — just fires the OS banner */
export async function scheduleScanNotification(disease: string, confidence: number): Promise<void> {
  if (!(await isEnabled())) return;
  await Notifications.scheduleNotificationAsync({
    content: {
      title: '🔬 Scan Analysis Complete',
      body: `AI detected ${disease} with ${confidence}% confidence.`,
      sound: true,
    },
    trigger: null,
  });
}

export const notifyScanComplete = (disease: string, confidence: number, imageUri?: string) =>
  sendLocalAndFirestore(
    '🔬 Scan Analysis Complete',
    `AI detected ${disease} with ${confidence}% confidence.`,
    'scan_result',
    { disease, confidence, imageUri: imageUri ?? '' },
  );

export const notifyLoginSuccess = (name: string) =>
  sendLocalAndFirestore(
    '👋 Welcome back to SkinSight',
    `Hello ${name}! You have successfully signed in.`,
    'system',
  );

export const notifyProfileUpdated = () =>
  sendLocalAndFirestore(
    '✅ Profile Updated',
    'Your personal information has been saved successfully.',
    'system',
  );

export const notifyPasswordChanged = () =>
  sendLocalAndFirestore(
    '🔐 Password Changed',
    'Your account password has been updated successfully.',
    'system',
  );

export const notifyThemeChanged = (isDark: boolean) =>
  sendLocalAndFirestore(
    '🎨 Appearance Changed',
    isDark ? 'Dark mode has been enabled.' : 'Light mode has been enabled.',
    'system',
  );

export const notifyCustomizeChanged = () =>
  sendLocalAndFirestore(
    '🎨 Settings Updated',
    'Your customization settings have been applied.',
    'system',
  );

export const notifyFileDownloaded = (reportNum: number) =>
  sendLocalAndFirestore(
    '📄 Report Downloaded',
    `Report #${reportNum} has been saved to your device.`,
    'system',
  );

export const notifyPhotoUploaded = () =>
  sendLocalAndFirestore(
    '📸 Photo Uploaded',
    'Your photo has been uploaded. AI analysis is in progress...',
    'system',
  );

export const notifyAccountCreated = (name: string) =>
  sendLocalAndFirestore(
    '🎉 Welcome to SkinSight!',
    `Your account has been created, ${name}. Start your first scan now!`,
    'system',
  );
