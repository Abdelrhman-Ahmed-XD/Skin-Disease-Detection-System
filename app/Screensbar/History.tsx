import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
    Alert, Dimensions, Image, ScrollView, StatusBar,
    StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';
import { loadAllScansFromFirestore, deleteMole as deleteMoleService } from '../../Firebase/firestoreService';

const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

const { width } = Dimensions.get('window');

type MoleResult = {
    status?: string;
    disease?: string;
    confidence?: number;
    segmentedUrl?: string;
    description?: string;
    tips?: string[];
    precautions?: string[];
    message?: string;
};

type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back' | 'N/A' | string;
    firestoreId?: string;
    analysis?: string;
    source?: string;
    result?: MoleResult;
};

const shortRejected = (msg: string): string => {
    if (!msg) return '';
    const m = msg.toLowerCase();
    if (m.includes('no skin') || m.includes('no lesion') || m.includes('no dermatological')) return 'No lesion detected';
    if (m.includes('quality') || m.includes('blur') || m.includes('clear image')) return 'Poor image quality';
    if (m.includes('not skin') || m.includes('non-skin') || m.includes('non skin')) return 'Not a skin image';
    if (m.includes('segment')) return 'Segmentation failed';
    return '';
};

export default function HistoryPage() {
    const router = useRouter();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const customText = {
      fontSize: settings.fontSize,
      color: isDark ? "#FFFFFF" : settings.textColor,
      fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
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
            const data = await loadAllScansFromFirestore();
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

    const sortedMoles = [...moles].sort((a, b) => b.timestamp - a.timestamp);

    const getDateGroup = (timestamp: number): string => {
        const now   = new Date();
        const date  = new Date(timestamp);
        const diff  = (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24);
        if (diff < 1 && now.getDate() === date.getDate()) return isArabic ? 'اليوم' : 'Today';
        if (diff < 2 && (now.getDate() - date.getDate() === 1 || (now.getDate() === 1 && date.getDate() >= 28))) return isArabic ? 'أمس' : 'Yesterday';
        if (diff < 7)  return isArabic ? 'هذا الأسبوع' : 'This Week';
        if (diff < 30) return isArabic ? 'هذا الشهر'   : 'This Month';
        return isArabic ? 'أقدم' : 'Earlier';
    };

    const groupOrder = isArabic
        ? ['اليوم','أمس','هذا الأسبوع','هذا الشهر','أقدم']
        : ['Today','Yesterday','This Week','This Month','Earlier'];

    const grouped = sortedMoles.reduce<Record<string, typeof sortedMoles>>((acc, m) => {
        const g = getDateGroup(m.timestamp);
        if (!acc[g]) acc[g] = [];
        acc[g].push(m);
        return acc;
    }, {});

    const analyzedCount = moles.filter(m => m.result?.disease || m.analysis).length;

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
            <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.headerTitle, customText]}>{t("history")}</Text>
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
              <Text style={[styles.emptyTitle, customText]}>{t("loading")}</Text>
            </View>
          ) : moles.length === 0 ? (
            <View style={styles.emptyContainer}>
              <View style={[styles.emptyIconWrap, { backgroundColor: isDark ? '#1E2A35' : '#E8F4F8' }]}>
                <Image source={Icons.history} style={styles.emptyIcon} resizeMode="contain" />
              </View>
              <Text style={[styles.emptyTitle, customText]}>{t("noHistoryYet")}</Text>
              <Text style={[styles.emptyText, customText, { color: colors.subText }]}>{t("noHistorySubtitle")}</Text>
            </View>
          ) : (
            <>
              {/* ── Summary bar ── */}
              <View style={[styles.summaryBar, { backgroundColor: isDark ? '#0D2030' : '#004F7F' }]}>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryValue}>{moles.length}</Text>
                  <Text style={styles.summaryLabel}>{isArabic ? 'إجمالي' : 'Total'}</Text>
                </View>
                <View style={styles.summaryDivider} />
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryValue}>{analyzedCount}</Text>
                  <Text style={styles.summaryLabel}>{isArabic ? 'محللة' : 'Analyzed'}</Text>
                </View>
                <View style={styles.summaryDivider} />
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryValue}>{moles.length - analyzedCount}</Text>
                  <Text style={styles.summaryLabel}>{isArabic ? 'معلقة' : 'Pending'}</Text>
                </View>
              </View>

              {/* ── Grouped cards ── */}
              {groupOrder.filter(g => grouped[g]?.length > 0).map(groupLabel => (
                <View key={groupLabel}>
                  <View style={[styles.groupHeader, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                    <View style={[styles.groupLine, { backgroundColor: isDark ? '#2A3F50' : '#C5E3ED' }]} />
                    <Text style={[styles.groupLabel, { color: isDark ? '#9CA3AF' : '#6B7280', fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }]}>
                      {groupLabel}
                    </Text>
                    <View style={[styles.groupLine, { backgroundColor: isDark ? '#2A3F50' : '#C5E3ED' }]} />
                  </View>

                  {grouped[groupLabel].map((mole, idx) => {
                    const globalIdx = sortedMoles.findIndex(m => m.id === mole.id);
                    const reportNumber = moles.length - globalIdx;
                    const result       = mole.result || {};
                    const displayDisease = result.disease || mole.analysis || '';
                    const isAnalyzed   = !!displayDisease;
                    const isRejected   = !displayDisease && !!(result.message || result.status);
                    const rejectedMsg  = isRejected ? (result.message || result.status || '') : '';
                    const conf         = result.confidence ?? 0;
                    const accentColor  = isAnalyzed
                        ? (conf >= 70 ? '#22C55E' : conf >= 50 ? '#F59E0B' : '#EF4444')
                        : isRejected ? '#EF4444' : '#9CA3AF';
                    const confColor    = accentColor;

                    return (
                      <TouchableOpacity
                        key={mole.id}
                        activeOpacity={0.92}
                        onPress={() => mole.photoUri && router.push({
                          pathname: '/Screensbar/Reportdetails',
                          params: {
                            moleId:      mole.id,
                            photoUri:    mole.photoUri || '',
                            timestamp:   mole.timestamp.toString(),
                            bodyView:    mole.bodyView,
                            x:           mole.x.toString(),
                            y:           mole.y.toString(),
                            analysis:    displayDisease,
                            reportIndex: (reportNumber - 1).toString(),
                            result:      JSON.stringify(result),
                          },
                        })}
                      >
                        <View style={[styles.card, { backgroundColor: colors.card, flexDirection: isArabic ? 'row-reverse' : 'row', borderWidth: 0.5, borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)' }]}>
                          {/* Accent bar */}
                          <View style={[styles.accentBar, { backgroundColor: accentColor }]} />

                          {/* Thumbnail */}
                          <View style={styles.thumbWrap}>
                            {mole.photoUri ? (
                              <Image source={{ uri: mole.photoUri }} style={styles.thumbnail} resizeMode="cover" />
                            ) : (
                              <View style={[styles.thumbPlaceholder, { backgroundColor: isDark ? '#2A3F50' : '#E8F4F8' }]}>
                                <Ionicons name="scan-outline" size={26} color={colors.primary} />
                              </View>
                            )}
                            {(isAnalyzed || isRejected) && (
                              <View style={[styles.thumbBadge, { backgroundColor: accentColor }]}>
                                <Ionicons name={isRejected ? 'close' : 'checkmark'} size={10} color="#fff" />
                              </View>
                            )}
                          </View>

                          {/* Main info */}
                          <View style={[styles.cardBody, { alignItems: isArabic ? 'flex-end' : 'flex-start' }]}>
                            <View style={[styles.cardTopRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                              <Text style={[styles.cardTitle, customText, { fontWeight: '700' }]}>
                                {t('reportNum')} {reportNumber}
                              </Text>
                              <Text style={[styles.cardDate, { color: colors.subText, fontSize: Math.max(10, settings.fontSize - 4) }]}>
                                {formatDate(mole.timestamp)}
                              </Text>
                            </View>

                            {displayDisease ? (
                              <Text style={[styles.diseaseText, { color: isDark ? '#00C8C8' : '#004F7F', fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }]} numberOfLines={2}>
                                {displayDisease}
                              </Text>
                            ) : isRejected ? (
                              <Text style={[styles.diseaseText, { color: '#EF4444', fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }]} numberOfLines={1}>
                                {isArabic ? 'تحليل مرفوض' : 'Analysis Rejected'}{shortRejected(rejectedMsg) ? ` · ${shortRejected(rejectedMsg)}` : ''}
                              </Text>
                            ) : null}

                            <View style={[styles.badgeRow, { flexDirection: isArabic ? 'row-reverse' : 'row', marginTop: 6 }]}>
                              <View style={[styles.chip, { backgroundColor: isDark ? '#1E3A4A' : '#EBF5FB' }]}>
                                <Ionicons name="body-outline" size={11} color={isDark ? '#C5E3ED' : '#004F7F'} />
                                <Text style={[styles.chipText, { color: isDark ? '#C5E3ED' : '#004F7F' }]}>
                                  {mole.bodyView === 'front' ? t('frontBody') : mole.bodyView === 'back' ? t('backBody') : 'N/A'}
                                </Text>
                              </View>
                              {conf > 0 && (
                                <View style={[styles.chip, { backgroundColor: isDark ? '#1E3A4A' : '#EBF5FB' }]}>
                                  <Text style={[styles.chipText, { color: confColor, fontWeight: '700' }]}>{conf}%</Text>
                                </View>
                              )}
                              {isAnalyzed && (
                                <View style={[styles.chip, { backgroundColor: isDark ? '#0D2A1A' : '#F0FDF4', borderColor: '#22C55E', borderWidth: 1 }]}>
                                  <Ionicons name="checkmark-circle" size={11} color="#22C55E" />
                                  <Text style={[styles.chipText, { color: '#22C55E' }]}>
                                    {isArabic ? 'محلل' : 'Analyzed'}
                                  </Text>
                                </View>
                              )}
                              {isRejected && (
                                <View style={[styles.chip, { backgroundColor: isDark ? '#2A0D0D' : '#FEF2F2', borderColor: '#EF4444', borderWidth: 1 }]}>
                                  <Ionicons name="close-circle" size={11} color="#EF4444" />
                                  <Text style={[styles.chipText, { color: '#EF4444' }]}>
                                    {isArabic ? 'مرفوض' : 'Rejected'}
                                  </Text>
                                </View>
                              )}
                            </View>

                            {conf > 0 && (
                              <View style={styles.confBarWrap}>
                                <View style={[styles.confBarBg, { backgroundColor: isDark ? '#2A3F50' : '#E5E7EB' }]}>
                                  <View style={[styles.confBarFill, { width: `${conf}%` as any, backgroundColor: confColor }]} />
                                </View>
                              </View>
                            )}
                          </View>

                          {/* Delete */}
                          <TouchableOpacity
                            style={styles.deleteBtn}
                            onPress={() => deleteMole(mole.id)}
                            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                          >
                            <Ionicons name="trash-outline" size={18} color="#EF4444" />
                          </TouchableOpacity>
                        </View>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              ))}
            </>
          )}
        </ScrollView>

        {/* ── Floating Bottom Nav ── */}
        <View style={styles.bottomNavContainer}>
          <View
            style={[
              styles.bottomNav,
              {
                backgroundColor: colors.navBg,
                borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
              },
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
                      { backgroundColor: colors.navBg },
                      isActive && {
                        backgroundColor: isDark ? "#1E3A4A" : pageBg,
                        borderWidth: 2,
                        borderColor: isDark ? "#00A3A3" : "#2A7DA0",
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
                      { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
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
                      { backgroundColor: colors.navBg },
                      isActive && {
                        backgroundColor: isDark ? "#1E3A4A" : pageBg,
                        borderWidth: 2,
                        borderColor: isDark ? "#00A3A3" : "#2A7DA0",
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
                      { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
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
    container:       { flex: 1 },
    header:          { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
    backButton:      { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
    headerTitleRow:  { flexDirection: 'row', alignItems: 'center' },
    headerTitle:     { fontSize: 22 },
    scrollView:      { flex: 1 },
    scrollContent:   { padding: 16, paddingBottom: 130 },
    // ── Empty state ──
    emptyContainer:  { alignItems: 'center', justifyContent: 'center', paddingVertical: 80 },
    emptyIconWrap:   { width: 100, height: 100, borderRadius: 50, alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
    emptyIcon:       { width: 56, height: 56 },
    emptyTitle:      { fontSize: 20, fontWeight: '700', marginTop: 16 },
    emptyText:       { fontSize: 14, marginTop: 8, textAlign: 'center', paddingHorizontal: 40 },
    // ── Summary bar ──
    summaryBar:      { flexDirection: 'row', borderRadius: 14, marginBottom: 16, paddingVertical: 14, paddingHorizontal: 8 },
    summaryItem:     { flex: 1, alignItems: 'center' },
    summaryValue:    { fontSize: 22, fontWeight: '800', color: '#fff' },
    summaryLabel:    { fontSize: 11, color: 'rgba(255,255,255,0.7)', marginTop: 2, fontWeight: '600' },
    summaryDivider:  { width: 1, backgroundColor: 'rgba(255,255,255,0.2)', marginVertical: 4 },
    // ── Group headers ──
    groupHeader:     { flexDirection: 'row', alignItems: 'center', marginBottom: 10, marginTop: 4 },
    groupLine:       { flex: 1, height: 1 },
    groupLabel:      { fontSize: 12, fontWeight: '700', marginHorizontal: 10, textTransform: 'uppercase', letterSpacing: 0.8 },
    // ── Cards ──
    card:            { borderRadius: 14, marginBottom: 10, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.07, shadowRadius: 6, elevation: 3, alignItems: 'stretch' },
    accentBar:       { width: 4, alignSelf: 'stretch' },
    thumbWrap:       { position: 'relative', margin: 12, marginRight: 0 },
    thumbnail:       { width: 76, height: 80, borderRadius: 10 },
    thumbPlaceholder:{ width: 76, height: 80, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
    thumbBadge:      { position: 'absolute', bottom: 4, right: 4, width: 18, height: 18, borderRadius: 9, alignItems: 'center', justifyContent: 'center' },
    cardBody:        { flex: 1, padding: 12, paddingLeft: 10 },
    cardTopRow:      { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
    cardTitle:       { fontSize: 14 },
    cardDate:        { fontSize: 11 },
    diseaseText:     { fontSize: 13, fontWeight: '600', marginBottom: 4 },
    pendingText:     { fontSize: 12, fontStyle: 'italic', marginBottom: 4 },
    badgeRow:        { flexDirection: 'row', gap: 6, flexWrap: 'wrap' },
    chip:            { flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 7, paddingVertical: 3, borderRadius: 8 },
    chipText:        { fontSize: 11, fontWeight: '600' },
    confBarWrap:     { marginTop: 6 },
    confBarBg:       { height: 4, borderRadius: 2, overflow: 'hidden' },
    confBarFill:     { height: '100%', borderRadius: 2 },
    deleteBtn:       { paddingRight: 12, paddingLeft: 4, justifyContent: 'center' },
    // ── Floating Nav ──
    bottomNavContainer: {
        position: 'absolute',
        bottom: 16,
        left: 16,
        right: 16,
        alignItems: 'center',
    },
    bottomNav: {
        flexDirection: 'row',
        paddingVertical: 10,
        paddingBottom: 14,
        borderRadius: 28,
        borderWidth: 1,
        width: '100%',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.12,
        shadowRadius: 12,
        elevation: 8,
    },
    navCenterSpacer:     { flex: 1 },
    navItem:             { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:             { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:          { width: 32, height: 32 },
    navText:             { fontSize: 11, fontWeight: '500' },
    cameraButton: {
        position: 'absolute',
        top: -26,
        alignSelf: 'center',
        width: 60,
        height: 60,
        borderRadius: 30,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.12,
        shadowRadius: 6,
        elevation: 6,
    },
});