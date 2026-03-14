import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
    Alert, Dimensions, Image, ScrollView, StatusBar,
    StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';
import { loadMolesFromFirestore, deleteMole as deleteMoleService } from '../../Firebase/firestoreService';

// ── Custom Icon Images ─────────────────────────────────────────
const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

const { width } = Dimensions.get('window');

type Mole = { id: string; x: number; y: number; timestamp: number; photoUri?: string; bodyView: 'front' | 'back'; firestoreId?: string; analysis?: string; };

export default function HistoryPage() {
    const router = useRouter();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const customText = {
      fontSize: settings.fontSize,
      color: isDark ? "#FFFFFF" : settings.textColor,
      fontFamily:
        settings.fontFamily === "System" ? undefined : settings.fontFamily,
    };

    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const [moles, setMoles]         = useState<Mole[]>([]);
    const [loading, setLoading]     = useState(true);
    const [activeTab, setActiveTab] = useState<string>('History');

    useFocusEffect(
        React.useCallback(() => {
            setActiveTab('History');
            loadMoles();
        }, [])
    );

    const loadMoles = async () => {
        try {
            const data = await loadMolesFromFirestore();
            setMoles(data);
        } catch (err) {
            console.log('Error loading moles:', err);
        } finally {
            setLoading(false);
        }
    };

    const deleteMole = async (moleId: string) => {
        Alert.alert(t('deleteEntry'), t('deleteEntryConfirm'), [
            { text: t('cancel'), style: 'cancel' },
            {
                text: t('delete'), style: 'destructive', onPress: async () => {
                    const updated = await deleteMoleService(moleId);
                    setMoles(updated);
                },
            },
        ]);
    };

    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    };

    const bottomTabs = [
        { name: 'Home',     iconImg: Icons.home     },
        { name: 'Reports',  iconImg: Icons.reports  },
        { name: 'History',  iconImg: Icons.history  },
        { name: 'Settings', iconImg: Icons.settings },
    ];

    const handleTabPress = (tabName: string) => {
        setActiveTab(tabName);
        switch (tabName) {
            case 'Home':     router.push('/Screensbar/FirstHomePage'); break;
            case 'Camera':   router.push('/Screensbar/Camera');        break;
            case 'Reports':  router.push('/Screensbar/Reports');       break;
            case 'Settings': router.push('/Screensbar/Setting');       break;
        }
    };

    const tabLabels: Record<string, string> = {
        Home: t('home'), Reports: t('reportsTab'),
        History: t('historyTab'), Settings: t('settingsTab'),
    };

    return (
      <SafeAreaView
        style={[styles.container, { backgroundColor: pageBg }]}
        edges={["top"]}
      >
        <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

        <View style={[styles.header, { backgroundColor: colors.card }]}>
          <TouchableOpacity
            style={[styles.backButton, { borderColor: colors.border }]}
            onPress={() => router.back()}
          >
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </TouchableOpacity>
          <View style={styles.headerTitleRow}>
            <Text style={[styles.headerTitle, customText]}>{t("history")}</Text>
          </View>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {loading ? (
            <View style={styles.emptyContainer}>
              <Text style={[styles.emptyTitle, customText]}>
                {t("loading")}
              </Text>
            </View>
          ) : moles.length === 0 ? (
            <View style={styles.emptyContainer}>
              <Image
                source={Icons.history}
                style={styles.emptyIcon}
                resizeMode="contain"
              />
              <Text style={[styles.emptyTitle, customText]}>
                {t("noHistoryYet")}
              </Text>
              <Text
                style={[
                  styles.emptyText,
                  customText,
                  { color: colors.subText },
                ]}
              >
                {t("noHistorySubtitle")}
              </Text>
            </View>
          ) : (
            <>
              <Text
                style={[
                  styles.countLabel,
                  customText,
                  {
                    color: colors.subText,
                    textAlign: isArabic ? "right" : "left",
                  },
                ]}
              >
                {moles.length}{" "}
                {moles.length === 1 ? t("entry") : t("entriesFound")}
              </Text>
              {[...moles].reverse().map((mole, index) => (
                <View
                  key={mole.id}
                  style={[styles.card, { backgroundColor: colors.card }]}
                >
                  <View
                    style={[
                      styles.cardRow,
                      { flexDirection: isArabic ? "row-reverse" : "row" },
                    ]}
                  >
                    {mole.photoUri ? (
                      <TouchableOpacity
                        onPress={() =>
                          router.push({
                            pathname: "/Screensbar/Camera",
                            params: {
                              tapX: mole.x.toFixed(2),
                              tapY: mole.y.toFixed(2),
                              bodyView: mole.bodyView,
                              moleId: mole.id,
                              existingPhotoUri: mole.photoUri || "",
                              firestoreId: mole.firestoreId || "",
                            },
                          })
                        }
                        activeOpacity={0.85}
                      >
                        <Image
                          source={{ uri: mole.photoUri }}
                          style={styles.thumbnail}
                          resizeMode="cover"
                        />
                      </TouchableOpacity>
                    ) : (
                      <View
                        style={[
                          styles.thumbnailPlaceholder,
                          { backgroundColor: isDark ? "#2A3F50" : "#E8F4F8" },
                        ]}
                      >
                        <Ionicons
                          name="scan-outline"
                          size={28}
                          color={colors.primary}
                        />
                      </View>
                    )}

                    <View
                      style={[
                        styles.cardInfo,
                        { alignItems: isArabic ? "flex-end" : "flex-start" },
                      ]}
                    >
                      <Text style={[styles.cardTitle, customText]}>
                        {t("entryNum")}
                        {moles.length - index}
                      </Text>
                      <Text
                        style={[
                          styles.cardDate,
                          customText,
                          {
                            color: colors.subText,
                            fontSize: Math.max(11, settings.fontSize - 4),
                          },
                        ]}
                      >
                        {formatDate(mole.timestamp)}
                      </Text>
                      <View
                        style={[
                          ,
                          styles.badgeRow,
                          { flexDirection: isArabic ? "row-reverse" : "row" },
                        ]}
                      >
                        <View
                          style={[
                            styles.badge,
                            {
                              backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                              flexDirection: isArabic ? "row-reverse" : "row",
                            },
                          ]}
                        >
                          <Ionicons
                            name="body-outline"
                            size={12}
                            color={isDark ? "#fff" : "#004f7f"}
                          />
                          <Text
                            style={[
                              styles.badgeText,
                              {
                                color: isDark ? "#fff" : "#004f7f",
                                fontSize: Math.max(11, settings.fontSize - 4),
                              },
                            ]}
                          >
                            {mole.bodyView === "front"
                              ? t("frontBody")
                              : t("backBody")}
                          </Text>
                        </View>
                        {mole.analysis && (
                          <View
                            style={[
                              styles.badge,
                              {
                                backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                                flexDirection: isArabic ? "row-reverse" : "row",
                              },
                            ]}
                          >
                            <Ionicons
                              name="checkmark-circle-outline"
                              size={12}
                              color="#10B981"
                            />
                            <Text
                              style={[
                                { color: isDark ? "#fff" : "#004f7f" },
                                styles.badgeText,
                                {
                                  color: "#10B981",
                                  fontSize: Math.max(11, settings.fontSize - 4),
                                },
                              ]}
                            >
                              {t("analyzed")}
                            </Text>
                          </View>
                        )}
                      </View>
                    </View>

                    <TouchableOpacity
                      style={styles.deleteButton}
                      onPress={() => deleteMole(mole.id)}
                      activeOpacity={0.7}
                    >
                      <Ionicons
                        name="trash-outline"
                        size={20}
                        color="#EF4444"
                      />
                    </TouchableOpacity>
                  </View>

                  {mole.analysis && (
                    <View
                      style={[
                        styles.analysisBox,
                        { backgroundColor: isDark ? "#1A2F3F" : "#F4FBFF" },
                      ]}
                    >
                      <Text
                        style={[
                          styles.analysisText,
                          { color: isDark ? "#fff" : "#004f7f" },
                          {
                            color: colors.subText,
                            textAlign: isArabic ? "right" : "left",
                          },
                        ]}
                        numberOfLines={3}
                      >
                        {mole.analysis}
                      </Text>
                    </View>
                  )}
                </View>
              ))}
            </>
          )}
        </ScrollView>

        <View style={styles.bottomNavContainer}>
          <View
            style={[
              styles.bottomNav,
              { backgroundColor: colors.navBg, borderTopColor: colors.border },
            ]}
          >
            {["Home", "Reports"].map((tabName) => {
              const tab = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity
                  key={tab.name}
                  style={styles.navItem}
                  onPress={() => handleTabPress(tab.name)}
                >
                  <View
                    style={[
                      styles.navIcon,
                      { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                      isActive && {
                        backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                        borderWidth: 2,
                        borderColor: isDark ? "#00A3A3" : "#C5E3ED",
                      },
                    ]}
                  >
                    <Image
                      source={tab.iconImg}
                      style={styles.navIconImg}
                      resizeMode="contain"
                    />
                  </View>
                  <Text
                    style={[
                      styles.navText,
                      { color: isActive ? colors.navActive : colors.navText },
                      isActive && { fontWeight: "700" },
                    ]}
                  >
                    {tabLabels[tabName]}
                  </Text>
                </TouchableOpacity>
              );
            })}
            <View style={styles.navCenterSpacer} />
            {["History", "Settings"].map((tabName) => {
              const tab = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity
                  key={tab.name}
                  style={styles.navItem}
                  onPress={() => handleTabPress(tab.name)}
                >
                  <View
                    style={[
                      styles.navIcon,
                      { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                      isActive && {
                        backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                        borderWidth: 2,
                        borderColor: isDark ? "#00A3A3" : "#C5E3ED",
                      },
                    ]}
                  >
                    <Image
                      source={tab.iconImg}
                      style={styles.navIconImg}
                      resizeMode="contain"
                    />
                  </View>
                  <Text
                    style={[
                      styles.navText,
                      { color: isActive ? colors.navActive : colors.navText },
                      isActive && { fontWeight: "700" },
                    ]}
                  >
                    {tabLabels[tabName]}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <TouchableOpacity
            style={[
              styles.cameraButton,
              {
                backgroundColor: colors.navBg,
                borderColor: isDark ? "#374151" : "#C5E3ED",
              },
              activeTab === "Camera" && {
                borderColor: colors.navActive,
                backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
              },
            ]}
            onPress={() => handleTabPress("Camera")}
            activeOpacity={0.85}
          >
            <Ionicons
              name="camera-outline"
              size={30}
              color={activeTab === "Camera" ? colors.navActive : colors.navText}
            />
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:           { flex: 1 },
    header:              { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
    backButton:          { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
    headerTitleRow:      { flexDirection: 'row', alignItems: 'center' },
    headerTitle:         { fontSize: 22, fontWeight: 'bold' },
    scrollView:          { flex: 1 },
    scrollContent:       { padding: 16, paddingBottom: 110 },
    countLabel:          { fontSize: 13, fontWeight: '600', marginBottom: 12, marginLeft: 4 },
    emptyContainer:      { alignItems: 'center', justifyContent: 'center', paddingVertical: 80 },
    emptyIcon:           { width: 90, height: 90 },
    emptyTitle:          { fontSize: 20, fontWeight: '700', marginTop: 16 },
    emptyText:           { fontSize: 14, marginTop: 8, textAlign: 'center', paddingHorizontal: 40 },
    card:                { borderRadius: 16, marginBottom: 12, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
    cardRow:             { flexDirection: 'row', alignItems: 'center', padding: 12, gap: 12 },
    thumbnail:           { width: 72, height: 72, borderRadius: 12 },
    thumbnailPlaceholder:{ width: 72, height: 72, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
    cardInfo:            { flex: 1 },
    cardTitle:           { fontWeight: '700', marginBottom: 4 },
    cardDate:            { marginBottom: 6 },
    badgeRow:            { flexDirection: 'row', gap: 6 },
    badge:               { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
    badgeText:           { fontWeight: '600' },
    deleteButton:        { padding: 8 },
    analysisBox:         { paddingHorizontal: 12, paddingBottom: 12 },
    analysisText:        { lineHeight: 20 },
    bottomNavContainer:  { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
    bottomNav:           { flexDirection: 'row', paddingVertical: 10, borderTopWidth: 1, width: '100%', paddingBottom: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20 },
    navCenterSpacer:     { flex: 1 },
    navItem:             { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:             { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:           { width: 44, height: 44 },
    navText:             { fontSize: 11, fontWeight: '500' },
    cameraButton:        { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 6 },
});