import {Ionicons} from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';
import {useLocalSearchParams, useRouter} from 'expo-router';
import {shareAsync} from 'expo-sharing';
import React, {useEffect, useState} from 'react';
import {
    Alert, Dimensions, Image, Platform, ScrollView, StatusBar,
    StyleSheet, Text, TouchableOpacity, View, ActivityIndicator,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {FONT_FAMILY_MAP, useCustomize} from '../Customize/Customizecontext';
import {useTranslation} from '../Customize/translations';
import {useTheme} from '../ThemeContext';

const {height} = Dimensions.get('window');

// ── Save PDF (same as Reports page) ──────────────────────────
const savePDF = async (uri: string, filename: string) => {
    if (Platform.OS === 'android') {
        try {
            const permissions =
                await FileSystem.StorageAccessFramework.requestDirectoryPermissionsAsync();
            if (permissions.granted) {
                const base64 = await FileSystem.readAsStringAsync(uri, {
                    encoding: FileSystem.EncodingType.Base64,
                });
                const destUri = await FileSystem.StorageAccessFramework.createFileAsync(
                    permissions.directoryUri,
                    filename,
                    'application/pdf',
                );
                await FileSystem.writeAsStringAsync(destUri, base64, {
                    encoding: FileSystem.EncodingType.Base64,
                });
                Alert.alert('✅ Downloaded', `"${filename}" saved successfully.`);
                return;
            }
        } catch (_) {}
    }
    await shareAsync(uri, {
        UTI: '.pdf',
        mimeType: 'application/pdf',
        dialogTitle: filename,
    });
};

// ── Convert local URI → base64 data URL ──────────────────────
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
        const ext = uri.split('.').pop()?.toLowerCase() || 'jpeg';
        const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
        return `data:${mime};base64,${base64}`;
    } catch (e) {
        console.log('getImageBase64 error:', e);
        return uri;
    }
};

// ── Skin color hex → label ────────────────────────────────────
const skinColorLabel = (hex: string | null) => {
    const map: Record<string, string> = {
        '#F5E0D3': 'Very Light', '#EACAA7': 'Light',
        '#D1A67A': 'Medium', '#B57D50': 'Tan',
        '#A05C38': 'Brown', '#8B4513': 'Dark Brown',
        '#7A3E11': 'Deep', '#603311': 'Ebony',
    };
    return hex ? (map[hex] || hex) : 'N/A';
};

// ── Build PDF HTML ────────────────────────────────────────────
const buildReportHTML = (params: {
    reportIndex: number;
    date: string;
    bodyView: string;
    x: number;
    y: number;
    moleId: string;
    analysis: string;
    imageBase64: string;
    frontBody: string;
    backBody: string;
    patientName: string;
    age: string;
    gender: string;
    hairColor: string;
    eyeColor: string;
    skinColor: string;
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
        <div class="patient-item">
          <div class="patient-label">Patient Name</div>
          <div class="patient-value">${params.patientName}</div>
        </div>
        <div class="patient-item">
          <div class="patient-label">Age</div>
          <div class="patient-value">${params.age}</div>
        </div>
        <div class="patient-item">
          <div class="patient-label">Gender</div>
          <div class="patient-value">${params.gender}</div>
        </div>
        <div class="patient-item">
          <div class="patient-label">Hair Color</div>
          <div class="patient-value">${params.hairColor}</div>
        </div>
        <div class="patient-item">
          <div class="patient-label">Eye Color</div>
          <div class="patient-value">${params.eyeColor}</div>
        </div>
        <div class="patient-item">
          <div class="patient-label">Skin Tone</div>
          <div class="patient-value">${params.skinColor}</div>
        </div>
      </div>
    </div>

    <div class="image-section">
      ${params.imageBase64 ? `<img src="${params.imageBase64}" alt="Skin Analysis" />` : '<p style="color:#9CA3AF;padding:20px;">No image available</p>'}
    </div>

    <div class="info-section">
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">Location</div>
          <div class="info-value">${params.bodyView === 'front' ? params.frontBody : params.bodyView === 'back' ? params.backBody : 'N/A'}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Coordinates</div>
          <div class="info-value">x: ${params.x.toFixed(1)}, y: ${params.y.toFixed(1)}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Report ID</div>
          <div class="info-value">${params.moleId.substring(0, 10)}...</div>
        </div>
      </div>
    </div>

    <div class="analysis-section">
      <div class="section-header">
        <div class="section-title">Analysis Results</div>
      </div>
      <div class="analysis-box">
        <p class="analysis-text">${params.analysis}</p>
      </div>
    </div>

    <div class="warning-section">
      <div class="warning-box">
        <p class="warning-text">
          ⚠️ <strong>Medical Disclaimer:</strong> This report is generated by an AI model and is
          intended for informational purposes only. It is not a substitute for professional medical
          advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare
          provider for any skin concerns.
        </p>
      </div>
    </div>

    <div class="footer">
      <div class="footer-divider"></div>
      <div class="footer-brand"><span class="footer-brand-s">S</span>kinsight</div>
      <div class="footer-copy">© 2026 SkinSight. All rights reserved.</div>
      <div class="footer-note">This is an automated report. Please do not reply directly.</div>
      <div class="footer-note">📧 skinsight.help.2025@gmail.com</div>
    </div>

  </div>
</body>
</html>
`;

export default function ReportDetailsPage() {
    const router = useRouter();
    const {colors, isDark} = useTheme();
    const {settings} = useCustomize();
    const {t, isArabic} = useTranslation(settings.language);
    const [isDownloading, setIsDownloading] = useState(false);

    // ── User profile loaded from AsyncStorage ─────────────────
    const [patientName, setPatientName] = useState('N/A');
    const [age, setAge] = useState('N/A');
    const [gender, setGender] = useState('N/A');
    const [hairColor, setHairColor] = useState('N/A');
    const [eyeColor, setEyeColor] = useState('N/A');
    const [skinColor, setSkinColor] = useState('N/A');

    useEffect(() => {
        AsyncStorage.getItem('signupDraft').then(saved => {
            if (!saved) return;
            const d = JSON.parse(saved);
            const firstName = d.firstName || '';
            const lastName = d.lastName || '';
            setPatientName(`${firstName} ${lastName}`.trim() || 'N/A');
            setGender(d.gender ? (d.gender.charAt(0).toUpperCase() + d.gender.slice(1)) : 'N/A');
            setHairColor(d.hairColor || 'N/A');
            setEyeColor(d.eyeColor || 'N/A');
            setSkinColor(skinColorLabel(d.skinColor));

            if (d.birthYear && d.birthMonth && d.birthDay) {
                const dob = new Date(d.birthYear, d.birthMonth - 1, d.birthDay);
                const diff = Date.now() - dob.getTime();
                const ageYears = Math.floor(diff / (1000 * 60 * 60 * 24 * 365.25));
                setAge(`${ageYears} years`);
            }
        }).catch(() => {});
    }, []);

    const customText = {
        fontSize: settings.fontSize,
        color: settings.textColor,
        fontFamily: FONT_FAMILY_MAP[settings.fontFamily]
    };
    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const params = useLocalSearchParams();
    const moleId = params.moleId as string;
    const photoUri = params.photoUri as string;
    const timestamp = parseInt(params.timestamp as string);
    const bodyView = params.bodyView as string;
    const x = parseFloat(params.x as string);
    const y = parseFloat(params.y as string);
    const analysis = params.analysis as string || t('analysisInProgress');
    const reportIndex = parseInt(params.reportIndex as string);

    const formatDate = (ts: number) =>
        new Date(ts).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', {
            month: 'long', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });

    // ── Download Report (same logic as Reports page) ──────────
    const downloadReport = async () => {
        if (isDownloading) return;
        setIsDownloading(true);
        try {
            const imageBase64 = await getImageBase64(photoUri);

            const html = buildReportHTML({
                reportIndex,
                date: formatDate(timestamp),
                bodyView,
                x, y,
                moleId,
                analysis,
                imageBase64,
                frontBody: t('frontBody'),
                backBody: t('backBody'),
                patientName,
                age,
                gender,
                hairColor,
                eyeColor,
                skinColor,
            });

            const { uri } = await Print.printToFileAsync({ html, base64: false });
            await savePDF(uri, `SkinSight_Report_${reportIndex + 1}.pdf`);

        } catch (error: any) {
            if (!String(error?.message || '').includes('Another share request')) {
                Alert.alert(t('error'), 'Failed to generate the report. Please try again.');
            }
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <SafeAreaView
            style={[styles.container, {backgroundColor: pageBg}]}
            edges={["top"]}
        >
            <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg}/>

            {/* Header */}
            <View style={[styles.header, {backgroundColor: colors.card}]}>
                <TouchableOpacity
                    style={[styles.backButton, {borderColor: colors.border}]}
                    onPress={() => router.back()}
                >
                    <Ionicons name="chevron-back" size={24} color={colors.text}/>
                </TouchableOpacity>
                <Text
                    style={[
                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                        styles.headerTitle,
                        customText,
                        {color: isDark ? "#fff" : "#000"},
                    ]}
                >
                    {t("reportDetails")}
                </Text>
                <TouchableOpacity
                    style={[
                        styles.downloadHeaderButton,
                        {backgroundColor: isDark ? "#004f7f" : "#E8F4F8"},
                    ]}
                    onPress={downloadReport}
                    disabled={isDownloading}
                >
                    {isDownloading ? (
                        <ActivityIndicator size="small" color={colors.primary}/>
                    ) : (
                        <Ionicons
                            name="download-outline"
                            size={24}
                            color={isDark ? "#fff" : "#004f7f"}
                        />
                    )}
                </TouchableOpacity>
            </View>

            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                {/* Scan Image */}
                <View style={styles.imageContainer}>
                    {photoUri ? (
                        <Image
                            source={{uri: photoUri}}
                            style={styles.mainImage}
                            resizeMode="cover"
                        />
                    ) : (
                        <View
                            style={[
                                styles.mainImage,
                                {
                                    backgroundColor: colors.card,
                                    alignItems: "center",
                                    justifyContent: "center",
                                },
                            ]}
                        >
                            <Ionicons
                                name="image-outline"
                                size={48}
                                color={colors.subText}
                            />
                        </View>
                    )}
                    <View style={styles.imageBadge}>
                        <Text
                            style={[
                                styles.imageBadgeText,
                                {fontFamily: customText.fontFamily},
                            ]}
                        >
                            {bodyView === "front" ? t("frontBody") : bodyView === "back" ? t("backBody") : "N/A"}
                        </Text>
                    </View>
                </View>

                {/* Patient Info Card */}
                <View style={[styles.infoCard, {backgroundColor: colors.card}]}>
                    <View
                        style={[styles.infoHeader, {borderBottomColor: colors.border}]}
                    >
                        <Ionicons
                            name="person-outline"
                            size={20}
                            color={isDark ? "#fff" : "#004f7f"}
                        />
                        <Text
                            style={[
                                { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                styles.sectionTitle,
                                customText,
                                {color: isDark ? "#fff" : "#004f7f"},
                            ]}
                        >
                            Patient Information
                        </Text>
                    </View>
                    <View style={[styles.patientGrid]}>
                        {[
                            {label: "Name", value: patientName},
                            {label: "Age", value: age},
                            {label: "Gender", value: gender},
                            {label: "Hair Color", value: hairColor},
                            {label: "Eye Color", value: eyeColor},
                            {label: "Skin Tone", value: skinColor},
                        ].map((item) => (
                            <View
                                key={item.label}
                                style={[
                                    styles.patientItem,
                                    {
                                        backgroundColor: isDark ? "#004f7f" : "#F4FBFF",
                                        borderColor: "#fff",
                                    },
                                ]}
                            >
                                <Text
                                    style={[
                                        styles.infoLabel,
                                        customText,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {item.label}
                                </Text>
                                <Text
                                    style={[
                                        styles.infoValue,
                                        customText,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {item.value}
                                </Text>
                            </View>
                        ))}
                    </View>
                </View>

                {/* Scan Info Card */}
                <View style={[styles.infoCard, {backgroundColor: colors.card}]}>
                    <View
                        style={[
                            styles.infoHeader,
                            {
                                borderBottomColor: colors.border,
                                flexDirection: isArabic ? "row-reverse" : "row",
                            },
                        ]}
                    >
                        <Text
                            style={[
                                { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                styles.reportNumber,
                                customText,
                                {color: isDark ? "#fff" : "#004f7f"},
                            ]}
                        >
                            {t("reportNum")}
                            {reportIndex + 1}
                        </Text>
                        <Text
                            style={[
                                { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                styles.dateText,
                                {color: isDark ? "#fff" : "#004f7f"},
                            ]}
                        >
                            {formatDate(timestamp)}
                        </Text>
                    </View>
                    <View style={[styles.infoGrid]}>
                        <View
                            style={[
                                styles.infoItem,
                                {flexDirection: isArabic ? "row-reverse" : "row"},
                            ]}
                        >
                            <Ionicons
                                name="location-outline"
                                size={20}
                                color={isDark ? "#fff" : "#004f7f"}
                            />
                            <View
                                style={[
                                    styles.infoTextContainer,
                                    {alignItems: isArabic ? "flex-end" : "flex-start"},
                                ]}
                            >
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoLabel,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {t("location")}
                                </Text>
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoValue,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {bodyView === "front" ? t("frontBody") : bodyView === "back" ? t("backBody") : "N/A"} {bodyView === "front" || bodyView === "back" ? "Body" : ""}
                                </Text>
                            </View>
                        </View>
                        <View
                            style={[
                                styles.infoItem,
                                {flexDirection: isArabic ? "row-reverse" : "row"},
                            ]}
                        >
                            <Ionicons
                                name="navigate-outline"
                                size={20}
                                color={isDark ? "#fff" : "#004f7f"}
                            />
                            <View
                                style={[
                                    styles.infoTextContainer,
                                    {alignItems: isArabic ? "flex-end" : "flex-start"},
                                ]}
                            >
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoLabel,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {t("coordinates")}
                                </Text>
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoValue,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    x: {x.toFixed(1)}, y: {y.toFixed(1)}
                                </Text>
                            </View>
                        </View>
                        <View
                            style={[
                                
                                styles.infoItem,
                                {flexDirection: isArabic ? "row-reverse" : "row"},
                            ]}
                        >
                            <Ionicons
                                name="finger-print-outline"
                                size={20}
                                color={isDark ? "#fff" : "#004f7f"}
                            />
                            <View
                                style={[
                                    styles.infoTextContainer,
                                    {alignItems: isArabic ? "flex-end" : "flex-start"},
                                ]}
                            >
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoLabel,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {t("reportId")}
                                </Text>
                                <Text
                                    style={[
                                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                                        styles.infoValue,
                                        {color: isDark ? "#fff" : "#004f7f"},
                                    ]}
                                >
                                    {moleId.substring(0, 12)}...
                                </Text>
                            </View>
                        </View>
                    </View>
                </View>

                {/* Analysis Card */}
                <View style={[styles.analysisCard, {backgroundColor: colors.card}]}>
                    <View
                        style={[
                            styles.analysisHeader,
                            {
                                borderBottomColor: colors.border,
                                flexDirection: isArabic ? "row-reverse" : "row",
                            },
                        ]}
                    >
                        <Ionicons name="document-text" size={24} color={isDark ? "#fff" : "#004f7f"}/>
                        <Text
                            style={[
                                styles.analysisTitle,
                                {color: isDark ? "#fff" : "#004f7f"},
                                {fontFamily: FONT_FAMILY_MAP[settings.fontFamily]},
                            ]}
                        >
                            {t("analysisResults")}
                        </Text>
                    </View>
                    <Text
                        style={[
                            styles.analysisText,
                            {color: isDark ? "#fff" : "#004f7f"},
                            {textAlign: isArabic ? "right" : "left"},
                            {fontFamily: FONT_FAMILY_MAP[settings.fontFamily]},
                        ]}
                    >
                        {analysis}
                    </Text>
                </View>

                {/* Download Button */}
                <TouchableOpacity
                    style={[ styles.downloadButton,
                        styles.downloadButton,
                        {
                            backgroundColor: colors.primary,
                            flexDirection: isArabic ? "row-reverse" : "row",
                            opacity: isDownloading ? 0.7 : 1,
                        },
                    ]}
                    onPress={downloadReport}
                    activeOpacity={0.8}
                    disabled={isDownloading}
                >
                    {isDownloading ? (
                        <View style={{flexDirection: "row", alignItems: "center", gap: 10}}>
                            <ActivityIndicator size="small" color="#fff"/>
                            <Text style={[styles.downloadButtonText, { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },]}>
                                Generating PDF...
                            </Text>
                        </View>
                    ) : (
                        <>
                            <Ionicons name="cloud-download-outline" size={24} color="#FFFFFF"/>
                            <Text style={[styles.downloadButtonText, {fontFamily: customText.fontFamily}]}>
                                {t("downloadAsPDF")}
                            </Text>
                        </>
                    )}
                </TouchableOpacity>

                {/* Disclaimer */}
                <View
                    style={[
                        styles.warningCard,
                        {
                            backgroundColor: isDark ? "#2D2000" : "#FEF3C7",
                            borderColor: isDark ? "#5C4000" : "#FCD34D",
                            flexDirection: isArabic ? "row-reverse" : "row",
                        },
                    ]}
                >
                    <Ionicons name="information-circle-outline" size={20} color="#F59E0B"/>
                    <Text
                        style={[
                            styles.warningText,
                            customText,
                            {
                                color: isDark ? "#FCD34D" : "#92400E",
                                textAlign: isArabic ? "right" : "left",
                                fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
                            },
                        ]}
                    >
                        {t("medicalDisclaimer")}
                    </Text>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {flex: 1},
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 15,
        shadowColor: '#000',
        shadowOffset: {width: 0, height: 2},
        shadowOpacity: 0.06,
        shadowRadius: 4,
        elevation: 2,
        margin: 15,
    },
    backButton: {
        width: 40, height: 40, borderRadius: 12, borderWidth: 1,
        alignItems: 'center', justifyContent: 'center',
    },
    downloadHeaderButton: {
        width: 40, height: 40, borderRadius: 12,
        alignItems: 'center', justifyContent: 'center',
    },
    headerTitle: {fontSize: 20},
    scrollView: {flex: 1},
    scrollContent: {padding: 16, paddingBottom: 40},
    imageContainer: {
        position: 'relative', width: '100%', height: height * 0.45,
        borderRadius: 16, overflow: 'hidden', marginBottom: 16,
    },
    mainImage: {width: '100%', height: '100%'},
    imageBadge: {
        position: 'absolute', top: 16, right: 16,
        backgroundColor: 'rgba(0,79,127,0.9)',
        paddingHorizontal: 16, paddingVertical: 8, borderRadius: 16,
    },
    imageBadgeText: {color: '#FFFFFF', fontSize: 14},
    infoCard: {
        borderRadius: 16, padding: 20, marginBottom: 16,
        shadowColor: '#000', shadowOffset: {width: 0, height: 2},
        shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
    },
    infoHeader: {
        flexDirection: 'row', alignItems: 'center', gap: 10,
        marginBottom: 16, paddingBottom: 12, borderBottomWidth: 1,
    },
    sectionTitle: {fontSize: 16},
    reportNumber: {fontSize: 20 },
    dateText: {fontSize: 13, marginLeft: 'auto'},
    patientGrid: {flexDirection: 'row', flexWrap: 'wrap', gap: 10},
    patientItem: {width: '30%', flexGrow: 1, borderRadius: 10, padding: 10, borderWidth: 1},
    infoGrid: {gap: 16},
    infoItem: {flexDirection: 'row', alignItems: 'center', gap: 12},
    infoTextContainer: {flex: 1},
    infoLabel: {fontSize: 12, marginBottom: 2},
    infoValue: {fontSize: 15, fontWeight: '600'},
    analysisCard: {
        borderRadius: 16, padding: 20, marginBottom: 16,
        shadowColor: '#000', shadowOffset: {width: 0, height: 2},
        shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
    },
    analysisHeader: {
        flexDirection: 'row', alignItems: 'center', gap: 10,
        marginBottom: 16, paddingBottom: 12, borderBottomWidth: 1,
    },
    analysisTitle: {fontSize: 18},
    analysisText: {fontSize: 15, lineHeight: 24},
    downloadButton: {
        flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
        paddingVertical: 16, borderRadius: 16, marginBottom: 16, gap: 10,
    },
    downloadButtonText: {fontSize: 16, color: '#FFFFFF'},
    warningCard: {
        flexDirection: 'row', alignItems: 'center',
        padding: 16, borderRadius: 12, gap: 12, borderWidth: 1,
    },
    warningText: {flex: 1, fontSize: 13, lineHeight: 18},
});