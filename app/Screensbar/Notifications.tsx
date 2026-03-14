import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { router } from 'expo-router';
import React, { useEffect, useState } from 'react';
import {
  Image, LayoutAnimation, Platform, ScrollView, StatusBar,
  StyleSheet, Text, TouchableOpacity, UIManager, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';
import {
  AppNotification, defaultNotifications,
  NOTIFICATIONS_ENABLED_KEY, NOTIFICATIONS_STORAGE_KEY,
} from './notificationsData';

if (Platform.OS === 'android') {
  UIManager.setLayoutAnimationEnabledExperimental &&
    UIManager.setLayoutAnimationEnabledExperimental(true);
}

export default function NotificationsPage() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();
  const { t, isArabic } = useTranslation(settings.language);

  const customText = {
    fontSize: settings.fontSize,
    color: isDark ? "#FFFFFF" : settings.textColor,
    fontFamily:
      settings.fontFamily === "System" ? undefined : settings.fontFamily,
  };

  // ✅ background من settings في light mode
  const pageBg = isDark ? colors.background : settings.backgroundColor;

  const [notifications, setNotifications]               = useState<AppNotification[]>([]);
  const [openId, setOpenId]                             = useState<number | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const enabledVal = await AsyncStorage.getItem(NOTIFICATIONS_ENABLED_KEY);
        const enabled = enabledVal === null ? true : enabledVal === 'true';
        setNotificationsEnabled(enabled);
        const saved = await AsyncStorage.getItem(NOTIFICATIONS_STORAGE_KEY);
        if (saved) {
          setNotifications(JSON.parse(saved));
        } else {
          setNotifications(defaultNotifications);
          await AsyncStorage.setItem(NOTIFICATIONS_STORAGE_KEY, JSON.stringify(defaultNotifications));
        }
      } catch {
        setNotifications(defaultNotifications);
      }
    };
    load();
  }, []);

  const saveNotifications = async (updated: AppNotification[]) => {
    setNotifications(updated);
    try { await AsyncStorage.setItem(NOTIFICATIONS_STORAGE_KEY, JSON.stringify(updated)); } catch {}
  };

  const unreadCount = notifications.filter((n) => !n.read).length;

  const handlePress = (id: number) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    const updated = notifications.map((n) => (n.id === id ? { ...n, read: true } : n));
    saveNotifications(updated);
    setOpenId((prev) => (prev === id ? null : id));
  };

  const markAllRead = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    saveNotifications(notifications.map((n) => ({ ...n, read: true })));
  };

  const cardBg       = isDark ? '#1E2A35' : '#FFFFFF';
  const cardUnreadBg = isDark ? '#1A2F3F' : '#F4FBFF';
  const accentColor  = isDark ? '#4BA3C7' : '#2A7DA0';

  if (!notificationsEnabled) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
        <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />
        <View style={[styles.header, { backgroundColor: colors.card }]}>
          <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
            <Ionicons name={isArabic ? "chevron-forward" : "chevron-back"} size={24} color={colors.text} />
          </TouchableOpacity>
          <View style={styles.headerTitleRow}>
            <Text style={[styles.headerTitle, customText]}>{t('notifications')}</Text>
          </View>
          <View style={{ width: 40 }} />
        </View>

        <View style={styles.disabledContainer}>
          <View style={[styles.disabledIconWrap, { backgroundColor: colors.card }]}>
            <Ionicons name="notifications-off-outline" size={48} color={colors.subText} />
          </View>
          <Text style={[styles.disabledTitle, customText]}>{t('notificationsDisabled')}</Text>
          <Text style={[styles.disabledSubtitle, customText]}>{t('notificationsDisabledSub')}</Text>
          <TouchableOpacity
            style={[styles.goToSettingsBtn, { backgroundColor: accentColor, flexDirection: isArabic ? 'row-reverse' : 'row' }]}
            onPress={() => router.push('/Screensbar/Setting')}
          >
            <Ionicons name="settings-outline" size={16} color="#FFFFFF" />
            <Text style={[styles.goToSettingsBtnText, { fontFamily: customText.fontFamily }]}>{t('goToSettings')}</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name={isArabic ? "chevron-back" : "chevron-back"} size={24} color={colors.text} />
        </TouchableOpacity>
        <View style={styles.headerTitleRow}>
          <Text style={[styles.headerTitle, customText]}>{t('notifications')}</Text>
        </View>
        <View style={{ width: 40 }} />
      </View>

      {unreadCount > 0 && (
<TouchableOpacity 
  style={[styles.markAllRow, { alignSelf: 'flex-end', marginRight: 16 }]} 
  onPress={markAllRead}
>
  <Text style={[styles.markAllText, customText, { color: accentColor }]}>{t('markAllRead')}</Text>
</TouchableOpacity>
      )}

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {notifications.map((notif) => {
          const isOpen = openId === notif.id;
          return (
            <TouchableOpacity
              key={notif.id}
              activeOpacity={0.85}
              onPress={() => handlePress(notif.id)}
              style={[
                styles.notifCard,
                { backgroundColor: notif.read ? cardBg : cardUnreadBg },
                isOpen && { borderColor: accentColor, elevation: 3 },
                !notif.read && { borderColor: isDark ? "#2A5570" : "#BEE0F0" },
              ]}
            >
              {!notif.read && (
                <View
                  style={[styles.unreadDot, { backgroundColor: accentColor }]}
                />
              )}

              <View
                style={[
                  styles.notifRow,
                  { flexDirection: isArabic ? "row-reverse" : "row" },
                ]}
              >
                <Image
                  source={{ uri: notif.image }}
                  style={[styles.notifImage, isOpen && styles.notifImageOpen]}
                  resizeMode="cover"
                />
                <View
                  style={[
                    styles.notifTextBlock,
                    { alignItems: isArabic ? "flex-end" : "flex-start" },
                  ]}
                >
                  <Text style={[styles.notifLabel, customText]}>
                    {t("skinDiseaseDetected")}
                  </Text>
                  <Text
                    style={[
                      styles.notifDisease,
                      customText,
                      {
                        color: isOpen
                          ? accentColor
                          : isDark
                            ? "#FFFFFF"
                            : settings.textColor,
                      },
                    ]}
                  >
                    {notif.disease}
                  </Text>
                  <Text style={[styles.notifTime, customText]}>
                    {notif.time}
                  </Text>
                </View>
                <View
                  style={[
                    styles.arrowBox,
                    { borderColor: accentColor },
                    isOpen && { backgroundColor: accentColor },
                  ]}
                >
                  <Ionicons
                    name={
                      isOpen
                        ? "chevron-down"
                        : isArabic
                          ? "chevron-back"
                          : "chevron-forward"
                    }
                    size={15}
                    color={isOpen ? "#FFFFFF" : accentColor}
                  />
                </View>
              </View>

              {isOpen && (
                <View style={styles.expandedContent}>
                  <View
                    style={[
                      styles.expandedDivider,
                      { backgroundColor: isDark ? "#2A3F50" : "#E5F0F6" },
                    ]}
                  />

                  <View
                    style={[
                      styles.confidenceRow,
                      { flexDirection: isArabic ? "row-reverse" : "row" },
                    ]}
                  >
                    <Ionicons
                      name="analytics-outline"
                      size={15}
                      color={accentColor}
                    />
                    <Text style={[styles.confidenceLabel, customText]}>
                      {t("aiConfidence")}
                    </Text>
                    <View
                      style={[
                        styles.confidenceBadge,
                        { backgroundColor: isDark ? "#1A3040" : "#E8F4FA" },
                      ]}
                    >
                      <Text
                        style={[
                          styles.confidenceBadgeText,
                          customText,
                          { color: accentColor },
                        ]}
                      >
                        {notif.details.confidence}
                      </Text>
                    </View>
                  </View>

                  <View style={styles.expandedSection}>
                    <View
                      style={[
                        styles.expandedSectionHeader,
                        { flexDirection: isArabic ? "row-reverse" : "row" },
                      ]}
                    >
                      <Ionicons
                        name="information-circle-outline"
                        size={15}
                        color={accentColor}
                      />
                      <Text style={[styles.expandedSectionTitle, customText]}>
                        {t("aboutCondition")}
                      </Text>
                    </View>
                    <Text
                      style={[
                        styles.expandedText,
                        customText,
                        { textAlign: isArabic ? "right" : "left" },
                      ]}
                    >
                      {notif.details.description}
                    </Text>
                  </View>

                  <View
                    style={[
                      styles.expandedSection,
                      styles.recommendationBox,
                      {
                        backgroundColor: isDark ? "#1A3040" : "#F0F9FF",
                        borderLeftColor: accentColor,
                      },
                    ]}
                  >
                    <View
                      style={[
                        styles.expandedSectionHeader,
                        { flexDirection: isArabic ? "row-reverse" : "row" },
                      ]}
                    >
                      <Ionicons
                        name="medkit-outline"
                        size={15}
                        color={accentColor}
                      />
                      <Text style={[styles.expandedSectionTitle, customText]}>
                        {t("recommendation")}
                      </Text>
                    </View>
                    <Text
                      style={[
                        styles.expandedText,
                        customText,
                        { textAlign: isArabic ? "right" : "left" },
                      ]}
                    >
                      {notif.details.recommendation}
                    </Text>
                  </View>

                  <TouchableOpacity
                    style={[
                      styles.consultButton,
                      {
                        backgroundColor: accentColor,
                        flexDirection: isArabic ? "row-reverse" : "row",
                      },
                    ]}
                  >
                    <Ionicons name="person-outline" size={15} color="#FFFFFF" />
                    <Text
                      style={[
                        styles.consultButtonText,
                        { fontFamily: customText.fontFamily },
                      ]}
                    >
                      {t("consultDoctor")}
                    </Text>
                  </TouchableOpacity>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:            { flex: 1 },
  header:               { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:           { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitleRow:       { flexDirection: 'row', alignItems: 'center' },
  headerTitle:          { fontSize: 22, fontWeight: 'bold' },
  markAllRow:           { marginBottom: 6, marginTop: -6 },
  markAllText:          { fontSize: 12, fontWeight: '600', textDecorationLine: 'underline' },
  scrollView:           { flex: 1 },
  scrollContent:        { paddingHorizontal: 15, paddingTop: 4 },
  disabledContainer:    { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 40 },
  disabledIconWrap:     { width: 90, height: 90, borderRadius: 45, alignItems: 'center', justifyContent: 'center', marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 6, elevation: 2 },
  disabledTitle:        { fontSize: 18, fontWeight: '700', marginBottom: 10, textAlign: 'center' },
  disabledSubtitle:     { fontSize: 14, textAlign: 'center', lineHeight: 22, marginBottom: 28 },
  goToSettingsBtn:      { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 12, paddingVertical: 12, paddingHorizontal: 24 },
  goToSettingsBtnText:  { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
  notifCard:            { borderRadius: 16, marginBottom: 10, borderWidth: 1, borderColor: 'transparent', shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 3, elevation: 1, overflow: 'hidden' },
  unreadDot:            { position: 'absolute', top: 14, left: 10, width: 8, height: 8, borderRadius: 4, zIndex: 10 },
  notifRow:             { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 14, paddingVertical: 12, paddingLeft: 22 },
  notifImage:           { width: 52, height: 52, borderRadius: 26, borderWidth: 2, borderColor: '#E5F0F6' },
  notifImageOpen:       { width: 56, height: 56, borderRadius: 28 },
  notifTextBlock:       { flex: 1, marginLeft: 12 },
  notifLabel:           { fontSize: 12, marginBottom: 2 },
  notifDisease:         { fontSize: 15, fontWeight: '600' },
  notifTime:            { fontSize: 11, marginTop: 3 },
  arrowBox:             { width: 28, height: 28, borderRadius: 8, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center', marginLeft: 8 },
  expandedContent:      { paddingHorizontal: 16, paddingBottom: 14 },
  expandedDivider:      { height: 1, marginBottom: 12 },
  confidenceRow:        { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  confidenceLabel:      { fontSize: 13, marginLeft: 5 },
  confidenceBadge:      { borderRadius: 8, paddingHorizontal: 8, paddingVertical: 2 },
  confidenceBadgeText:  { fontSize: 13, fontWeight: '700' },
  expandedSection:      { marginBottom: 10 },
  expandedSectionHeader:{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 },
  expandedSectionTitle: { fontSize: 13, fontWeight: '700', marginLeft: 5 },
  expandedText:         { fontSize: 13, lineHeight: 19, paddingLeft: 20 },
  recommendationBox:    { borderRadius: 10, padding: 10, borderLeftWidth: 3 },
  consultButton:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', borderRadius: 10, paddingVertical: 10, marginTop: 6, gap: 6 },
  consultButtonText:    { color: '#FFFFFF', fontSize: 13, fontWeight: '600' },
});