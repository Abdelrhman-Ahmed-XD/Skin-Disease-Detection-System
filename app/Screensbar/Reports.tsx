import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';
import { useFocusEffect, useRouter } from 'expo-router';
import { shareAsync } from 'expo-sharing';
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator, Alert, Dimensions, Image, ScrollView,
    StatusBar, StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';
import { loadAllScansFromFirestore } from '../../Firebase/firestoreService';

// ── Custom Icon Images ─────────────────────────────────────────
const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

const { width } = Dimensions.get('window');

// FIXED: Updated type to allow 'N/A' and source string
type Mole = {
    id: string; x: number; y: number; timestamp: number;
    photoUri?: string; bodyView: 'front' | 'back' | 'N/A' | string; analysis?: string; source?: string;
};

const getImageBase64 = async (uri: string): Promise<string> => {
    try {
        if (!uri) return '';
        if (uri.startsWith('http://') || uri.startsWith('https://')) {
            try {
                const downloadRes = await FileSystem.downloadAsync(
                    uri,
                    FileSystem.cacheDirectory + 'tmp_report_img.jpg'
                );
                const base64 = await FileSystem.readAsStringAsync(downloadRes.uri, {
                    encoding: FileSystem.EncodingType.Base64,
                });
                return `data:image/jpeg;base64,${base64}`;
            } catch {
                return uri;
            }
        }
        const base64 = await FileSystem.readAsStringAsync(uri, {
            encoding: FileSystem.EncodingType.Base64,
        });
        const ext  = uri.split('.').pop()?.toLowerCase() || 'jpeg';
        const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
        return `data:${mime};base64,${base64}`;
    } catch (e) {
        console.log('getImageBase64 error:', e);
        return uri;
    }
};

const buildReportHTML = (params: {
    reportIndex: number; date: string; bodyView: string;
    x: number; y: number; moleId: string; analysis: string;
    imageBase64: string; frontBody: string; backBody: string;
    patientName: string; age: string; gender: string;
    hairColor: string; eyeColor: string; skinColor: string;
}) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Georgia, 'Times New Roman', serif; background: #D8E9F0; padding: 30px 20px; }
    .page { max-width: 700px; margin: 0 auto; background: #D8E9F0; border-radius: 16px; overflow: hidden; }
    .header { background: #004F7F; padding: 36px 24px 28px; text-align: center; }
    .brand { font-size: 48px; font-weight: bold; color: #ffffff; letter-spacing: 2px; line-height: 1.2; }
    .brand-s { color: #00A3A3; font-size: 56px; }
    .tagline { color: #C5E3ED; font-size: 13px; margin-top: 6px; font-style: italic; letter-spacing: 3px; }
    .header-divider { width: 60px; height: 3px; background: #00A3A3; margin: 14px auto 0; border-radius: 10px; }
    .banner { background: #00A3A3; padding: 10px 20px; text-align: center; }
    .banner p { color: #fff; font-size: 13px; font-style: italic; letter-spacing: 0.5px; }
    .report-title-bar { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 20px 24px 16px; display: flex; justify-content: space-between; align-items: center; }
    .report-num { font-size: 22px; font-weight: bold; color: #004F7F; }
    .report-date { font-size: 13px; color: #6B7280; font-family: system-ui, sans-serif; }
    .patient-section { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 16px 24px; border-top: 1px solid #E5F0F6; }
    .patient-title { font-size: 14px; font-weight: bold; color: #004F7F; margin-bottom: 12px; font-family: system-ui, sans-serif; text-transform: uppercase; letter-spacing: 0.5px; }
    .patient-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .patient-item { background: #F4FBFF; border-radius: 8px; padding: 10px 12px; border: 1px solid #C5E3ED; }
    .patient-label { font-size: 10px; color: #9CA3AF; font-family: system-ui, sans-serif; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
    .patient-value { font-size: 13px; font-weight: bold; color: #1F2937; font-family: system-ui, sans-serif; }
    .image-section { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 0 24px 20px; text-align: center; }
    .image-section img { max-width: 100%; max-height: 320px; border-radius: 12px; border: 3px solid #C5E3ED; object-fit: cover; }
    .info-section { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 0 24px 20px; }
    .info-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 4px; }
    .info-item { background: #F4FBFF; border-radius: 10px; padding: 12px; border: 1px solid #C5E3ED; }
    .info-label { font-size: 11px; color: #9CA3AF; font-family: system-ui, sans-serif; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .info-value { font-size: 14px; font-weight: bold; color: #004F7F; font-family: system-ui, sans-serif; }
    .analysis-section { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 20px 24px; }
    .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid #E5E7EB; }
    .section-title { font-size: 17px; font-weight: bold; color: #004F7F; }
    .analysis-box { background: #D8E9F0; border-radius: 12px; padding: 18px 20px; border: 1px solid #C5E3ED; }
    .analysis-text { font-size: 14px; color: #374151; line-height: 1.8; font-family: system-ui, sans-serif; }
    .warning-section { background: #ffffff; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; padding: 0 24px 20px; }
    .warning-box { background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px; padding: 14px 16px; }
    .warning-text { font-size: 12px; color: #856404; line-height: 1.6; font-family: system-ui, sans-serif; }
    .footer { background: #004F7F; padding: 28px 20px; text-align: center; }
    .footer-divider { width: 40px; height: 2px; background: #00A3A3; margin: 0 auto 18px; border-radius: 10px; }
    .footer-brand { font-size: 20px; font-weight: bold; color: #fff; margin-bottom: 6px; }
    .footer-brand-s { color: #00A3A3; }
    .footer-copy { color: #C5E3ED; font-size: 11px; font-family: system-ui, sans-serif; }
    .footer-note { color: #8ab4c9; font-size: 10px; margin-top: 5px; font-family: system-ui, sans-serif; }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="brand"><span class="brand-s">S</span>kinsight</div>
      <div class="tagline">Snap. Detect. Protect.</div>
      <div class="header-divider"></div>
    </div>
    <div class="banner"><p>Skin Analysis Report</p></div>
    <div class="report-title-bar">
      <div class="report-num">Report #${params.reportIndex + 1}</div>
      <div class="report-date">${params.date}</div>
    </div>
    <div class="patient-section">
      <div class="patient-title">Patient Information</div>
      <div class="patient-grid">
        <div class="patient-item"><div class="patient-label">Patient Name</div><div class="patient-value">${params.patientName}</div></div>
        <div class="patient-item"><div class="patient-label">Age</div><div class="patient-value">${params.age}</div></div>
        <div class="patient-item"><div class="patient-label">Gender</div><div class="patient-value">${params.gender}</div></div>
        <div class="patient-item"><div class="patient-label">Hair Color</div><div class="patient-value">${params.hairColor}</div></div>
        <div class="patient-item"><div class="patient-label">Eye Color</div><div class="patient-value">${params.eyeColor}</div></div>
        <div class="patient-item"><div class="patient-label">Skin Tone</div><div class="patient-value">${params.skinColor}</div></div>
      </div>
    </div>
    <div class="image-section">
      ${params.imageBase64 ? `<img src="${params.imageBase64}" alt="Skin Analysis" />` : '<p style="color:#9CA3AF;padding:20px;">No image available</p>'}
    </div>
    <div class="info-section">
      <div class="info-grid">
        <div class="info-item"><div class="info-label">Location</div><div class="info-value">${params.bodyView === 'front' ? params.frontBody : params.bodyView === 'back' ? params.backBody : 'N/A'}</div></div>
        <div class="info-item"><div class="info-label">Coordinates</div><div class="info-value">x: ${params.x.toFixed(1)}, y: ${params.y.toFixed(1)}</div></div>
        <div class="info-item"><div class="info-label">Report ID</div><div class="info-value">${params.moleId.substring(0, 10)}...</div></div>
      </div>
    </div>
    <div class="analysis-section">
      <div class="section-header"><div class="section-title">Analysis Results</div></div>
      <div class="analysis-box"><p class="analysis-text">${params.analysis}</p></div>
    </div>
    <div class="warning-section">
      <div class="warning-box">
        <p class="warning-text">⚠️ <strong>Medical Disclaimer:</strong> This report is generated by an AI model and is intended for informational purposes only. Always consult a qualified dermatologist or healthcare provider for any skin concerns.</p>
      </div>
    </div>
    <div class="footer">
      <div class="footer-divider"></div>
      <div class="footer-brand"><span class="footer-brand-s">S</span>kinsight</div>
      <div class="footer-copy">© 2026 SkinSight. All rights reserved.</div>
      <div class="footer-note">📧 skinsight.help.2025@gmail.com</div>
    </div>
  </div>
</body>
</html>
`;

export default function ReportsPage() {
    const router = useRouter();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const customText = {
        fontSize:   settings.fontSize,
        color:      settings.textColor,
        fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
    };

    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const [moles, setMoles]                   = useState<Mole[]>([]);
    const [loading, setLoading]               = useState(true);
    const [downloadingAll, setDownloadingAll] = useState(false);
    const [downloadingId,  setDownloadingId]  = useState<string | null>(null);
    const [activeTab, setActiveTab]           = useState<string>('Reports');

    const [patientName, setPatientName] = useState('N/A');
    const [age,         setAge]         = useState('N/A');
    const [gender,      setGender]      = useState('N/A');
    const [hairColor,   setHairColor]   = useState('N/A');
    const [eyeColor,    setEyeColor]    = useState('N/A');
    const [skinColor,   setSkinColor]   = useState('N/A');

    useEffect(() => {
        AsyncStorage.getItem('signupDraft').then(saved => {
            if (!saved) return;
            const d = JSON.parse(saved);
            setPatientName(`${d.firstName || ''} ${d.lastName || ''}`.trim() || 'N/A');
            setGender(d.gender ? d.gender.charAt(0).toUpperCase() + d.gender.slice(1) : 'N/A');
            setHairColor(d.hairColor || 'N/A');
            setEyeColor(d.eyeColor   || 'N/A');
            const skinMap: Record<string,string> = { '#F5E0D3':'Very Light','#EACAA7':'Light','#D1A67A':'Medium','#B57D50':'Tan','#A05C38':'Brown','#8B4513':'Dark Brown','#7A3E11':'Deep','#603311':'Ebony' };
            setSkinColor(d.skinColor ? (skinMap[d.skinColor] || d.skinColor) : 'N/A');
            if (d.birthYear && d.birthMonth && d.birthDay) {
                const dob = new Date(d.birthYear, d.birthMonth - 1, d.birthDay);
                setAge(`${Math.floor((Date.now() - dob.getTime()) / (1000*60*60*24*365.25))} years`);
            }
        }).catch(() => {});
    }, []);

    useEffect(() => { loadMoles(); }, []);

    useFocusEffect(
        React.useCallback(() => {
            setActiveTab('Reports');
            loadMoles();
        }, [])
    );

    const loadMoles = async () => {
        try {
            const data = await loadAllScansFromFirestore();
            setMoles(data.filter((m: Mole) => m.photoUri));
        } catch (err) {
            console.log('Error loading moles:', err);
        } finally {
            setLoading(false);
        }
    };

    const downloadSingleReport = async (mole: Mole, index: number) => {
        if (downloadingId || downloadingAll) return;
        try {
            setDownloadingId(mole.id);
            if (!mole.photoUri) return;
            const imageBase64 = await getImageBase64(mole.photoUri);
            const html = buildReportHTML({
                reportIndex: index,
                date: new Date(mole.timestamp).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
                bodyView: mole.bodyView, x: mole.x, y: mole.y, moleId: mole.id,
                analysis: mole.analysis || t('analysisInProgress'),
                imageBase64, frontBody: t('frontBody'), backBody: t('backBody'),
                patientName, age, gender, hairColor, eyeColor, skinColor,
            });
            const { uri } = await Print.printToFileAsync({ html, base64: false });
            await shareAsync(uri, { UTI: '.pdf', mimeType: 'application/pdf', dialogTitle: `SkinSight Report #${index + 1}` });
        } catch (error: any) {
            if (!String(error?.message || '').includes('Another share request')) {
                Alert.alert(t('error'), 'Failed to download report.');
            }
        } finally {
            setDownloadingId(null);
        }
    };

    const downloadAllReports = async () => {
        if (downloadingId || downloadingAll) return;
        try {
            setDownloadingAll(true);
            if (moles.length === 0) { Alert.alert(t('noReportsYet'), t('noReportsToDownload')); return; }
            const pages: string[] = [];
            for (let i = 0; i < moles.length; i++) {
                const mole        = moles[i];
                const imageBase64 = await getImageBase64(mole.photoUri || '');
                const page        = buildReportHTML({
                    reportIndex: i,
                    date: new Date(mole.timestamp).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
                    bodyView: mole.bodyView, x: mole.x, y: mole.y, moleId: mole.id,
                    analysis: mole.analysis || t('analysisInProgress'),
                    imageBase64, frontBody: t('frontBody'), backBody: t('backBody'),
                    patientName, age, gender, hairColor, eyeColor, skinColor,
                });
                const bodyContent = page.replace(/[\s\S]*<body>/, '').replace(/<\/body>[\s\S]*/, '');
                pages.push(bodyContent);
            }
            const firstFull = buildReportHTML({ reportIndex: 0, date: '', bodyView: 'front', x: 0, y: 0, moleId: '', analysis: '', imageBase64: '', frontBody: '', backBody: '', patientName: '', age: '', gender: '', hairColor: '', eyeColor: '', skinColor: '' });
            const styleMatch = firstFull.match(/<style>([\s\S]*?)<\/style>/);
            const sharedStyle = styleMatch ? styleMatch[1] : '';
            const allPagesHtml = pages.join('<div style="page-break-after:always;height:1px;"></div>');
            const html = `<!DOCTYPE html><html><head><meta charset="utf-8"/><style>${sharedStyle}</style></head><body>${allPagesHtml}</body></html>`;
            const { uri } = await Print.printToFileAsync({ html, base64: false });
            await shareAsync(uri, { UTI: '.pdf', mimeType: 'application/pdf', dialogTitle: t('downloadAll') });
        } catch (error: any) {
            Alert.alert(t('error'), 'Failed to download reports.');
        } finally {
            setDownloadingAll(false);
        }
    };

    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' });
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
            case 'History':  router.push('/Screensbar/History');       break;
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
          <Text
            style={[
              styles.headerTitle,
              customText,
              { color: isDark ? "#fff" : "#374151" },
            ]}
          >
            {t("reports")}
          </Text>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={colors.primary} />
              <Text style={[styles.loadingText, customText]}>
                {t("loadingReports")}
              </Text>
            </View>
          ) : moles.length === 0 ? (
            <View style={styles.emptyContainer}>
              <Image
                source={Icons.reports}
                style={styles.emptyIcon}
                resizeMode="contain"
              />
              <Text
                style={[
                  styles.emptyTitle,
                  customText,
                  { color: isDark ? "#fff" : "#374151" },
                ]}
              >
                {t("noReportsYet")}
              </Text>
              <Text
                style={[
                  styles.emptyText,
                  customText,
                  { color: colors.subText },
                ]}
              >
                {t("noReportsSubtitle")}
              </Text>
            </View>
          ) : (
            <>
              {moles.map((mole, index) => (
                <View
                  key={mole.id}
                  style={[styles.reportCard, { backgroundColor: colors.card }]}
                >
                  <TouchableOpacity
                    style={styles.imageContainer}
                    onPress={() =>
                      router.push({
                        pathname: "/Screensbar/Reportdetails",
                        params: {
                          moleId: mole.id,
                          photoUri: mole.photoUri,
                          timestamp: mole.timestamp.toString(),
                          bodyView: mole.bodyView,
                          x: mole.x.toString(),
                          y: mole.y.toString(),
                          analysis: mole.analysis || "",
                          reportIndex: index.toString(),
                        },
                      })
                    }
                    activeOpacity={0.9}
                  >
                    <Image
                      source={{ uri: mole.photoUri }}
                      style={styles.reportImage}
                      resizeMode="cover"
                    />
                    <View style={styles.imageBadge}>
                      <Text style={styles.imageBadgeText}>
                        {mole.bodyView === "front"
                          ? t("frontBody")
                          : mole.bodyView === "back"
                          ? t("backBody")
                          : "N/A"}
                      </Text>
                    </View>
                    <View style={styles.expandIcon}>
                      <Ionicons
                        name="expand-outline"
                        size={20}
                        color="#FFFFFF"
                      />
                    </View>
                  </TouchableOpacity>

                  <View style={styles.reportContent}>
                    <View
                      style={[
                        styles.reportHeader,
                        { flexDirection: isArabic ? "row-reverse" : "row" },
                      ]}
                    >
                      <Text style={[styles.reportTitle, customText]}>
                        {t("reportNum")}
                        {index + 1}
                      </Text>
                      <Text
                        style={[
                          styles.reportDate,
                          customText,
                          {
                            color: colors.subText,
                            fontSize: Math.max(11, settings.fontSize - 3),
                          },
                        ]}
                      >
                        {formatDate(mole.timestamp)}
                      </Text>
                    </View>
                    <Text
                      style={[
                        styles.reportText,
                        customText,
                        {
                          color: colors.subText,
                          textAlign: isArabic ? "right" : "left",
                        },
                      ]}
                    >
                      {mole.analysis || t("analysisInProgress")}
                    </Text>
                    <TouchableOpacity
                      style={[
                        styles.downloadButton,
                        {
                          backgroundColor: isDark ? "#004F7F" : "#E8F4F8",
                          borderColor: isDark ? "#374151" : "#C5E3ED",
                          flexDirection: isArabic ? "row-reverse" : "row",
                          alignSelf: isArabic ? "flex-start" : "flex-end",
                          opacity: downloadingId || downloadingAll ? 0.5 : 1,
                        },
                      ]}
                      onPress={() => downloadSingleReport(mole, index)}
                      activeOpacity={0.8}
                      disabled={!!downloadingId || downloadingAll}
                    >
                      {downloadingId === mole.id ? (
                        <ActivityIndicator
                          size="small"
                          color={colors.primary}
                        />
                      ) : (
                        <>
                          <Ionicons
                            name="download-outline"
                            size={18}
                            color={isDark ? "#E8F4F8" : "#374151"}
                          />
                          <Text
                            style={[
                              styles.downloadButtonText,
                              { color: isDark ? "#E8F4F8" : "#374151" },
                            ]}
                          >
                            {t("downloadPDF")}
                          </Text>
                        </>
                      )}
                    </TouchableOpacity>
                  </View>
                </View>
              ))}

              <TouchableOpacity
                style={[
                  styles.downloadAllButton,
                  {
                    backgroundColor: colors.primary,
                    flexDirection: isArabic ? "row-reverse" : "row",
                  },
                ]}
                onPress={downloadAllReports}
                disabled={downloadingAll}
                activeOpacity={0.8}
              >
                {downloadingAll ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <>
                    <Ionicons
                      name="cloud-download-outline"
                      size={22}
                      color="#FFFFFF"
                    />
                    <Text style={styles.downloadAllText}>
                      {t("downloadAll")}
                    </Text>
                  </>
                )}
              </TouchableOpacity>
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
                      {
                        backgroundColor: isDark ? "#152030" : "#F9FAFB",
                      },
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
                      {
                        color: isActive ? colors.navActive : colors.navText,
                      },
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
                      {
                        backgroundColor: isDark ? "#152030" : "#F9FAFB",
                      },
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
                      {
                        color: isActive ? colors.navActive : colors.navText,
                      },
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
    container:          { flex: 1 },
    header:             { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
    backButton:         { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
    headerTitle:        { fontSize: 22, fontWeight: 'bold' },
    scrollView:         { flex: 1 },
    scrollContent:      { padding: 16, paddingBottom: 100 },
    loadingContainer:   { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 100 },
    loadingText:        { marginTop: 12, fontSize: 16 },
    emptyContainer:     { alignItems: 'center', justifyContent: 'center', paddingVertical: 80 },
    emptyIcon:           { width: 90, height: 90 },
    emptyTitle:         { fontSize: 20, fontWeight: '700', marginTop: 16 },
    emptyText:          { fontSize: 14, marginTop: 8, textAlign: 'center', paddingHorizontal: 40 },
    reportCard:         { borderRadius: 16, marginBottom: 16, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
    imageContainer:     { position: 'relative', width: '100%', height: 200 },
    reportImage:        { width: '100%', height: '100%' },
    imageBadge:         { position: 'absolute', top: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
    imageBadgeText:     { color: '#FFFFFF', fontSize: 12, fontWeight: '600' },
    expandIcon:         { position: 'absolute', bottom: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.8)', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
    reportContent:      { padding: 16 },
    reportHeader:       { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
    reportTitle:        { fontSize: 18, fontWeight: '700' },
    reportDate:         { fontSize: 12 },
    reportText:         { fontSize: 14, lineHeight: 20, marginBottom: 16 },
    downloadButton:     { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 10, paddingHorizontal: 16, borderRadius: 12, borderWidth: 1 },
    downloadButtonText: { fontSize: 14, fontWeight: '600', marginLeft: 6 },
    downloadAllButton:  { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, borderRadius: 16, marginTop: 8, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 4, marginBottom: 25 },
    downloadAllText:    { fontSize: 16, fontWeight: '700', color: '#FFFFFF', marginLeft: 8 },
    bottomNavContainer: { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
    bottomNav:          { flexDirection: 'row', paddingVertical: 10, borderTopWidth: 1, width: '100%', paddingBottom: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20 },
    navCenterSpacer:    { flex: 1 },
    navItem:            { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:            { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:         { width: 44, height: 44 },
    navText:            { fontSize: 11, fontWeight: '500' },
    cameraButton:       { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 6 },
});