import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  LayoutAnimation, Platform, UIManager, StatusBar, ActivityIndicator, Image
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useTheme } from '../ThemeContext';
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';

// ── FIREBASE IMPORTS ──
import { collection, query, orderBy, onSnapshot, updateDoc, doc } from 'firebase/firestore';
import { db, auth } from '../../Firebase/firebaseConfig';

if (Platform.OS === 'android') {
  UIManager.setLayoutAnimationEnabledExperimental &&
    UIManager.setLayoutAnimationEnabledExperimental(true);
}

// ── DYNAMIC INTERFACE ──
export interface AppNotification {
  id: string;
  type: 'scan_result' | 'system';
  title: string;
  message: string;
  time: string;
  isRead: boolean;
  disease?: string;
  confidence?: number;
  timestamp?: any;
  imageUri?: string;
}

export default function NotificationsPage() {
  const router = useRouter();
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();
  const { t } = useTranslation(settings.language);

  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);

  const customText = {
    fontSize: settings.fontSize,
    color: isDark ? "#FFFFFF" : settings.textColor,
    fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
  };

  const pageBg = isDark ? colors.background : settings.backgroundColor;

  // ── HELPER: GET ICON COLOR ──
  const getIconColor = (disease?: string) => {
    if (disease === "Healthy" || disease === "No significant condition detected") return '#10B981';
    return '#EF4444';
  };

  // ── HELPER: TOGGLE EXPAND & MARK AS READ ──
  const toggleExpand = async (id: string, isRead: boolean) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpandedId(expandedId === id ? null : id);

    if (!isRead && auth.currentUser) {
        try {
            const notifRef = doc(db, 'users', auth.currentUser.uid, 'notifications', id);
            await updateDoc(notifRef, { isRead: true });
        } catch (error) {
            console.error("Failed to mark notification as read:", error);
        }
    }
  };

  useEffect(() => {
    const checkSettings = async () => {
      const enabledVal = await AsyncStorage.getItem('notificationsEnabled');
      setNotificationsEnabled(enabledVal === null ? true : enabledVal === 'true');
    };
    checkSettings();

    const user = auth.currentUser;
    if (!user) {
      setLoading(false);
      return;
    }

    const q = query(
      collection(db, 'users', user.uid, 'notifications'),
      orderBy('timestamp', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const fetchedNotifs: AppNotification[] = snapshot.docs.map(docSnap => {
        const data = docSnap.data();
        let timeString = "Just now";
        if (data.timestamp) {
            const date = data.timestamp.toDate();
            timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        return {
          id: docSnap.id,
          type: data.type || 'system',
          title: data.title || 'Notification',
          message: data.message || '',
          time: timeString,
          isRead: data.isRead || false,
          disease: data.disease || null,
          confidence: data.confidence || null,
          imageUri: data.imageUri || null,
        };
      });
      setNotifications(fetchedNotifs);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: pageBg }]}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={28} color={customText.color} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: customText.color, fontFamily: customText.fontFamily }]}>
          Notifications
        </Text>
      </View>

      {!notificationsEnabled ? (
        <View style={styles.disabledContainer}>
            <Ionicons name="notifications-off-outline" size={80} color="#9CA3AF" />
            <Text style={[styles.disabledTitle, customText]}>
                Notifications are currently turned off.
            </Text>
            <Text style={[styles.disabledSub, { fontFamily: customText.fontFamily }]}>
                Enable them in settings to receive real-time updates on your skin scans.
            </Text>
            <TouchableOpacity
                style={styles.settingsBtn}
                onPress={() => router.push("/Screensbar/Setting")}
            >
                <Text style={[styles.settingsBtnText, { fontFamily: customText.fontFamily }]}>
                    Go to Settings
                </Text>
            </TouchableOpacity>
        </View>
      ) : loading ? (
        <View style={styles.center}>
            <ActivityIndicator size="large" color="#00A3A3" />
        </View>
      ) : notifications.length === 0 ? (
        <View style={styles.center}>
            <Ionicons name="notifications-outline" size={60} color="#D1D5DB" />
            <Text style={[styles.emptyText, { fontFamily: customText.fontFamily }]}>No notifications yet.</Text>
        </View>
      ) : (
        <ScrollView style={styles.listContainer} showsVerticalScrollIndicator={false}>
          {notifications.map((notif) => {
            const isExpanded = expandedId === notif.id;
            const bgContainer = isDark ? colors.card : '#FFFFFF';
            const unreadBg = isDark ? '#1F2937' : '#F0F9FA';

            return (
              <View key={notif.id} style={[styles.notifWrapper, { backgroundColor: notif.isRead ? bgContainer : unreadBg }]}>
                <TouchableOpacity
                  activeOpacity={0.7}
                  onPress={() => toggleExpand(notif.id, notif.isRead)}
                  style={styles.notifHeader}
                >
                  <View style={styles.iconContainer}>
                    {notif.imageUri ? (
                        <Image
                            source={{ uri: notif.imageUri }}
                            style={[styles.thumb, { borderColor: isDark ? '#374151' : '#E5E7EB' }]}
                        />
                    ) : (
                        <Ionicons
                            name={notif.type === 'scan_result' ? 'scan-circle' : 'information-circle'}
                            size={40}
                            color={notif.type === 'scan_result' ? getIconColor(notif.disease) : '#00A3A3'}
                        />
                    )}
                    {!notif.isRead && <View style={styles.unreadDot} />}
                  </View>

                  <View style={styles.notifTextBlock}>
                    <Text style={[styles.notifTitle, { color: customText.color, fontFamily: customText.fontFamily, fontWeight: notif.isRead ? '500' : 'bold' }]}>
                      {notif.title}
                    </Text>
                    <Text style={[styles.notifMessage, { color: isDark ? '#9CA3AF' : '#6B7280', fontFamily: customText.fontFamily }]} numberOfLines={isExpanded ? undefined : 1}>
                      {notif.message}
                    </Text>
                    <Text style={[styles.notifTime, { color: isDark ? '#6B7280' : '#9CA3AF' }]}>{notif.time}</Text>
                  </View>

                  <Ionicons name={isExpanded ? "chevron-up" : "chevron-down"} size={20} color={isDark ? '#9CA3AF' : '#6B7280'} />
                </TouchableOpacity>

                {isExpanded && notif.type === 'scan_result' && (
                  <View style={styles.expandedContent}>
                    <View style={[styles.expandedDivider, { backgroundColor: isDark ? '#374151' : '#E5E7EB' }]} />
                    <Text style={[styles.expandedDisease, { color: customText.color, fontFamily: customText.fontFamily }]}>
                        Diagnosis: <Text style={{ color: getIconColor(notif.disease), fontWeight: 'bold' }}>{notif.disease}</Text>
                    </Text>
                    {notif.confidence && (
                        <Text style={[styles.expandedConfidence, { color: isDark ? '#D1D5DB' : '#4B5563', fontFamily: customText.fontFamily }]}>
                            AI Confidence: {notif.confidence}%
                        </Text>
                    )}
                    <TouchableOpacity
                        style={styles.viewReportButton}
                        onPress={() => router.push('/Screensbar/Reports')}
                    >
                        <Text style={styles.viewReportText}>View Full Report</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>
            );
          })}
          <View style={{ height: 40 }} />
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', marginTop: 50 },
  header: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12 },
  backButton: { padding: 4, marginRight: 12 },
  headerTitle: { fontSize: 20, fontWeight: 'bold' },
  listContainer: { flex: 1, paddingHorizontal: 16, paddingTop: 10 },
  notifWrapper: { borderRadius: 16, marginBottom: 12, overflow: 'hidden', elevation: 1, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3 },
  notifHeader: { flexDirection: 'row', padding: 16, alignItems: 'center' },
  iconContainer: { position: 'relative', marginRight: 12 },
  thumb: { width: 44, height: 44, borderRadius: 8, borderWidth: 1 },
  unreadDot: { position: 'absolute', top: 0, right: 0, width: 12, height: 12, borderRadius: 6, backgroundColor: '#EF4444', borderWidth: 2, borderColor: '#FFF' },
  notifTextBlock: { flex: 1, paddingRight: 10 },
  notifTitle: { fontSize: 16, marginBottom: 4 },
  notifMessage: { fontSize: 14, lineHeight: 20 },
  notifTime: { fontSize: 12, marginTop: 6 },
  expandedContent: { paddingHorizontal: 16, paddingBottom: 16 },
  expandedDivider: { height: 1, marginBottom: 12 },
  expandedDisease: { fontSize: 15, marginBottom: 6 },
  expandedConfidence: { fontSize: 14, marginBottom: 16 },
  viewReportButton: { backgroundColor: '#00A3A3', paddingVertical: 10, borderRadius: 8, alignItems: 'center' },
  viewReportText: { color: '#FFF', fontWeight: 'bold', fontSize: 14 },
  emptyText: { marginTop: 16, color: '#9CA3AF' },
  disabledContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 40 },
  disabledTitle: { textAlign: 'center', marginBottom: 12 },
  disabledSub: { textAlign: 'center', color: '#6B7280', marginBottom: 32 },
  settingsBtn: { backgroundColor: '#004F7F', paddingVertical: 16, paddingHorizontal: 40, borderRadius: 14, width: '100%', alignItems: 'center' },
  settingsBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 }
});