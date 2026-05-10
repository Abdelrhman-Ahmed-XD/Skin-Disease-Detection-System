import {Ionicons} from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';
import {useLocalSearchParams, useRouter} from 'expo-router';
import {shareAsync} from 'expo-sharing';
import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {
    Alert, Image, Platform, ScrollView, StatusBar,
    StyleSheet, Text, TouchableOpacity, View, ActivityIndicator, Linking
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {FONT_FAMILY_MAP, useCustomize} from '../Customize/Customizecontext';
import {useTranslation} from '../Customize/translations';
import {useTheme} from '../ThemeContext';

// ── Save PDF ──────────────────────────────────────────────────
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
        '#D1A67A': 'Medium',    '#B57D50': 'Tan',
        '#A05C38': 'Brown',     '#8B4513': 'Dark Brown',
        '#7A3E11': 'Deep',      '#603311': 'Ebony',
    };
    return hex ? (map[hex] || hex) : 'N/A';
};

const getPlatform = (source?: string): string =>
    (source || '').toLowerCase().includes('web') ? 'Web' : 'Mobile App';

// ── EXACT SAME buildReportHTML as ReportsPage ─────────────────
const buildReportHTML = (params: {
    reportIndex: number; date: string; bodyView: string;
    moleId: string; analysis: string; imageBase64: string;
    frontBody: string; backBody: string; platform: string; description: string;
    patientName: string; age: string; gender: string;
    hairColor: string; eyeColor: string; skinColor: string;
    confidence: number;
    diseaseDescription: string;
}) => `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,'Times New Roman',serif;background:#D8E9F0;padding:18px 14px}
.page{max-width:700px;margin:0 auto;background:#D8E9F0;border-radius:14px;overflow:hidden}
.header{background:#004F7F;padding:22px 22px 16px;text-align:center}
.brand{font-size:38px;font-weight:bold;color:#fff;letter-spacing:2px}
.brand-s{color:#00A3A3;font-size:46px}
.tagline{color:#C5E3ED;font-size:11px;margin-top:3px;font-style:italic;letter-spacing:3px}
.hdiv{width:46px;height:3px;background:#00A3A3;margin:8px auto 0;border-radius:10px}
.banner{background:#00A3A3;padding:7px 18px;text-align:center}
.banner p{color:#fff;font-size:11px;font-style:italic;letter-spacing:.5px}
.title-bar{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:12px 22px;display:flex;justify-content:space-between;align-items:center}
.rnum{font-size:19px;font-weight:bold;color:#004F7F}
.rdate{font-size:11px;color:#6B7280;font-family:system-ui,sans-serif}
.psec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:10px 22px;border-top:1px solid #E5F0F6}
.ptitle{font-size:11px;font-weight:bold;color:#004F7F;margin-bottom:7px;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.5px}
.pgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}
.pi{background:#F4FBFF;border-radius:7px;padding:7px 9px;border:1px solid #C5E3ED}
.pl{font-size:8px;color:#9CA3AF;font-family:system-ui,sans-serif;margin-bottom:2px;text-transform:uppercase;letter-spacing:.4px}
.pv{font-size:11px;font-weight:bold;color:#1F2937;font-family:system-ui,sans-serif}
.imgsec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:0 22px 12px;text-align:center}
.imgsec img{max-width:100%;max-height:200px;border-radius:10px;border:3px solid #C5E3ED;object-fit:cover}
.infosec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:0 22px 12px}
.igrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:9px;margin-top:4px}
.ii{background:#F4FBFF;border-radius:9px;padding:9px;border:1px solid #C5E3ED}
.il{font-size:9px;color:#9CA3AF;font-family:system-ui,sans-serif;margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px}
.iv{font-size:12px;font-weight:bold;color:#004F7F;font-family:system-ui,sans-serif}
.asec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:12px 22px}
.shdr{margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #E5E7EB}
.stitle{font-size:14px;font-weight:bold;color:#004F7F}
.abox{background:#D8E9F0;border-radius:9px;padding:12px 15px;border:1px solid #C5E3ED}
.atxt{font-size:12px;color:#374151;line-height:1.65;font-family:system-ui,sans-serif}
.wsec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:0 22px 12px}
.wbox{background:#fff3cd;border-left:4px solid #ffc107;border-radius:7px;padding:9px 13px}
.wtxt{font-size:10px;color:#856404;line-height:1.5;font-family:system-ui,sans-serif}
.footer{background:#004F7F;padding:16px 18px;text-align:center}
.fdiv{width:34px;height:2px;background:#00A3A3;margin:0 auto 10px;border-radius:10px}
.fbrand{font-size:15px;font-weight:bold;color:#fff;margin-bottom:3px}
.fbs{color:#00A3A3}
.fcopy{color:#C5E3ED;font-size:9px;font-family:system-ui,sans-serif}
.fnote{color:#8ab4c9;font-size:8px;margin-top:3px;font-family:system-ui,sans-serif}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div class="brand"><span class="brand-s">S</span>kinsight</div>
    <div class="tagline">Snap. Detect. Protect.</div>
    <div class="hdiv"></div>
  </div>
  <div class="banner"><p>Skin Analysis Report</p></div>
  <div class="title-bar">
    <div class="rnum">Report #${params.reportIndex + 1}</div>
    <div class="rdate">${params.date}</div>
  </div>
  <div class="psec">
    <div class="ptitle">Patient Information</div>
    <div class="pgrid">
      <div class="pi"><div class="pl">Patient Name</div><div class="pv">${params.patientName}</div></div>
      <div class="pi"><div class="pl">Age</div><div class="pv">${params.age}</div></div>
      <div class="pi"><div class="pl">Gender</div><div class="pv">${params.gender}</div></div>
      <div class="pi"><div class="pl">Hair Color</div><div class="pv">${params.hairColor}</div></div>
      <div class="pi"><div class="pl">Eye Color</div><div class="pv">${params.eyeColor}</div></div>
      <div class="pi"><div class="pl">Skin Tone</div><div class="pv">${params.skinColor}</div></div>
    </div>
  </div>
  <div class="imgsec">
    ${params.imageBase64
        ? `<img src="${params.imageBase64}" alt="Skin Analysis"/>`
        : '<p style="color:#9CA3AF;padding:14px;">No image available</p>'}
  </div>
  <div class="infosec">
    <div class="igrid">
      <div class="ii"><div class="il">Location</div><div class="iv">${params.bodyView === 'front' ? params.frontBody : params.bodyView === 'back' ? params.backBody : 'N/A'}</div></div>
      <div class="ii"><div class="il">Platform</div><div class="iv">${params.platform}</div></div>
      <div class="ii"><div class="il">Description</div><div class="iv">${params.description || 'N/A'}</div></div>
    </div>
  </div>
  <div class="asec">
    <div class="shdr"><div class="stitle">Analysis Results</div></div>
    <div style="display:flex;justify-content:space-between;align-items:center;background:#E8F4F8;border-radius:9px;padding:12px 15px;border:1px solid #C5E3ED;margin-bottom:10px">
      <div>
        <div style="font-size:9px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px">Detected Condition</div>
        <div style="font-size:15px;font-weight:bold;color:#004F7F;font-family:system-ui,sans-serif">${params.analysis}</div>
      </div>
      ${params.confidence > 0 ? `
      <div style="text-align:center;min-width:90px">
        <div style="font-size:9px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px">AI Confidence</div>
        <div style="font-size:18px;font-weight:bold;font-family:system-ui,sans-serif;color:${params.confidence >= 80 ? '#22C55E' : params.confidence >= 60 ? '#F59E0B' : '#EF4444'}">${params.confidence}%</div>
        <div style="width:80px;height:6px;background:#E5E7EB;border-radius:3px;overflow:hidden;margin-top:5px">
          <div style="height:100%;width:${params.confidence}%;border-radius:3px;background:${params.confidence >= 80 ? '#22C55E' : params.confidence >= 60 ? '#F59E0B' : '#EF4444'}"></div>
        </div>
      </div>` : ''}
    </div>
    ${params.diseaseDescription ? `
    <div>
      <div style="font-size:9px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px">About This Condition</div>
      <div class="abox"><p class="atxt">${params.diseaseDescription}</p></div>
    </div>` : `<div class="abox"><p class="atxt">${params.analysis}</p></div>`}
  </div>
  <div class="wsec">
    <div class="wbox">
      <p class="wtxt">⚠️ <strong>Medical Disclaimer:</strong> This report is generated by an AI model and is intended for informational purposes only. Always consult a qualified dermatologist or healthcare provider for any skin concerns.</p>
    </div>
  </div>
  <div class="footer">
    <div class="fdiv"></div>
    <div class="fbrand"><span class="fbs">S</span>kinsight</div>
    <div class="fcopy">© 2026 SkinSight. All rights reserved.</div>
    <div class="fnote">📧 skinsight.help.2025@gmail.com</div>
  </div>
</div>
</body>
</html>`;

export default function ReportDetailsPage() {
    const router = useRouter();
    const {colors, isDark} = useTheme();
    const {settings} = useCustomize();
    const {t, isArabic} = useTranslation(settings.language);
    const [isDownloading, setIsDownloading] = useState(false);
    const [isReady, setIsReady] = useState(false);

    // ── Parse ALL route params ────────────────────────────────
    const rawParams = useLocalSearchParams();

    const moleId      = (rawParams.moleId       as string) || '';
    const photoUri    = (rawParams.photoUri      as string) || '';
    const timestamp   = parseInt((rawParams.timestamp     as string) || '0');
    const bodyView    = (rawParams.bodyView      as string) || '';
    const reportIndex = parseInt((rawParams.reportIndex   as string) || '0');
    // These two are now passed from ReportsPage so the PDF matches exactly
    const source      = (rawParams.source        as string) || '';
    const descParam   = (rawParams.description   as string) || '';

    // ── Parse AI result ───────────────────────────────────────
    const aiResult = useMemo(() => {
        try { if (rawParams.result) return JSON.parse(rawParams.result as string); }
        catch (e) { console.log('Failed to parse aiResult', e); }
        return {};
    }, [rawParams.result]);

    const analysis    = aiResult.disease      || (rawParams.analysis as string) || t('analysisInProgress');
    const confidence  = aiResult.confidence   || 0;
    const maskUrl     = aiResult.segmentedUrl || aiResult.segmented_url || '';
    const description = aiResult.description  || '';
    const tips        = aiResult.tips         || [];
    const precautions = aiResult.precautions  || [];
    const sources     = aiResult.sources      || [];

    // ── User profile ──────────────────────────────────────────
    const [patientName, setPatientName] = useState('N/A');
    const [age,         setAge        ] = useState('N/A');
    const [gender,      setGender     ] = useState('N/A');
    const [hairColor,   setHairColor  ] = useState('N/A');
    const [eyeColor,    setEyeColor   ] = useState('N/A');
    const [skinColor,   setSkinColor  ] = useState('N/A');

    useEffect(() => {
        const loadPatientData = async () => {
            const keys = ['userProfile', 'profileData', 'signupDraft'];
            let d: any = null;
            for (const key of keys) {
                try {
                    const saved = await AsyncStorage.getItem(key);
                    if (saved) {
                        const parsed = JSON.parse(saved);
                        if (parsed && (parsed.firstName || parsed.name || parsed.hairColor)) {
                            d = parsed; break;
                        }
                    }
                } catch (_) {}
            }
            if (d) {
                const firstName = d.firstName || d.name?.split(' ')[0] || '';
                const lastName  = d.lastName  || d.name?.split(' ').slice(1).join(' ') || '';
                setPatientName(`${firstName} ${lastName}`.trim() || 'N/A');
                setGender(d.gender ? (d.gender.charAt(0).toUpperCase() + d.gender.slice(1)) : 'N/A');
                setHairColor(d.hairColor || 'N/A');
                setEyeColor(d.eyeColor   || 'N/A');
                setSkinColor(skinColorLabel(d.skinColor));
                if (d.birthYear && d.birthMonth && d.birthDay) {
                    const dob = new Date(d.birthYear, d.birthMonth - 1, d.birthDay);
                    const ageYears = Math.floor((Date.now() - dob.getTime()) / (1000 * 60 * 60 * 24 * 365.25));
                    setAge(`${ageYears} years`);
                } else if (d.age) {
                    setAge(typeof d.age === 'number' ? `${d.age} years` : d.age);
                }
            }
        };
        loadPatientData().catch(() => {}).finally(() => setIsReady(true));
    }, []);

    const customText = {
        fontSize: settings.fontSize,
        color: settings.textColor,
        fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
    };
    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const formatDate = useCallback((ts: number) =>
        new Date(ts).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', {
            month: 'long', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        }), [isArabic]);

    // ── Download — builds IDENTICAL PDF to ReportsPage ────────
    const downloadReport = useCallback(async () => {
        if (isDownloading || !isReady) return;
        setIsDownloading(true);
        try {
            const imageBase64 = await getImageBase64(photoUri);
            const html = buildReportHTML({
                reportIndex,
                date:               formatDate(timestamp),
                bodyView,
                moleId,
                analysis,
                imageBase64,
                frontBody:          t('frontBody'),
                backBody:           t('backBody'),
                patientName,
                age,
                gender,
                hairColor,
                eyeColor,
                skinColor,
                platform:           getPlatform(source),   // same as ReportsPage
                description:        descParam || 'N/A',    // same as ReportsPage
                confidence,
                diseaseDescription: description,
            });
            const {uri} = await Print.printToFileAsync({html, base64: false});
            await savePDF(uri, `SkinSight_Report_${reportIndex + 1}.pdf`);
        } catch (error: any) {
            if (!String(error?.message || '').includes('Another share request')) {
                Alert.alert(t('error'), 'Failed to generate the report. Please try again.');
            }
        } finally {
            setIsDownloading(false);
        }
    }, [
        isDownloading, isReady, photoUri, reportIndex, timestamp, bodyView,
        moleId, analysis, patientName, age, gender, hairColor, eyeColor,
        skinColor, source, descParam, confidence, description, formatDate, t,
    ]);

    return (
        <SafeAreaView style={[styles.container, {backgroundColor: pageBg}]} edges={["top"]}>
            <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg}/>

            {/* Header */}
            <View style={[styles.header, {backgroundColor: colors.card}]}>
                <TouchableOpacity style={[styles.backButton, {borderColor: colors.border}]} onPress={() => router.back()}>
                    <Ionicons name="chevron-back" size={24} color={colors.text}/>
                </TouchableOpacity>
                <Text style={[{fontFamily: FONT_FAMILY_MAP[settings.fontFamily]}, styles.headerTitle, customText, {color: isDark ? "#fff" : "#000"}]}>
                    {t("reportDetails")}
                </Text>
                {/* Header download — same function as bottom button */}
                <TouchableOpacity
                    style={[styles.downloadHeaderButton, {backgroundColor: isDark ? "#004f7f" : "#E8F4F8"}]}
                    onPress={downloadReport}
                    disabled={isDownloading || !isReady}
                >
                    {isDownloading
                        ? <ActivityIndicator size="small" color={colors.primary}/>
                        : <Ionicons name="download-outline" size={24} color={isDark ? "#fff" : "#004f7f"}/>}
                </TouchableOpacity>
            </View>

            <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

                {/* Side-by-Side Images */}
                <View style={styles.imagePairContainer}>
                    <View style={styles.imageBox}>
                        <Text style={[styles.imageLabel, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>Original</Text>
                        <Image source={{uri: photoUri}} style={[styles.scanImage, {backgroundColor: colors.card}]}/>
                        <View style={styles.imageBadge}>
                            <Text style={[styles.imageBadgeText, {fontFamily: customText.fontFamily}]}>
                                {bodyView === "front" ? t("frontBody") : bodyView === "back" ? t("backBody") : "N/A"}
                            </Text>
                        </View>
                    </View>
                    {maskUrl ? (
                        <View style={styles.imageBox}>
                            <Text style={[styles.imageLabel, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>U-Net Mask</Text>
                            <Image source={{uri: maskUrl}} style={[styles.scanImage, {borderColor: '#00A3A3', borderWidth: 2, backgroundColor: '#000'}]}/>
                        </View>
                    ) : null}
                </View>

                {/* Patient Information Card */}
                <View style={[styles.infoCard, {backgroundColor: colors.card}]}>
                    <View style={[styles.infoHeader, {borderBottomColor: colors.border}]}>
                        <Ionicons name="person-outline" size={20} color={isDark ? "#fff" : "#004f7f"}/>
                        <Text style={[{fontFamily: FONT_FAMILY_MAP[settings.fontFamily]}, styles.sectionTitle, customText, {color: isDark ? "#fff" : "#004f7f"}]}>
                            Patient Information
                        </Text>
                    </View>
                    <View style={styles.patientGrid}>
                        {[
                            {label: "Name",       value: patientName},
                            {label: "Age",        value: age},
                            {label: "Gender",     value: gender},
                            {label: "Hair Color", value: hairColor},
                            {label: "Eye Color",  value: eyeColor},
                            {label: "Skin Tone",  value: skinColor},
                        ].map((item) => (
                            <View key={item.label} style={[styles.patientItem, {backgroundColor: isDark ? "#004f7f" : "#F4FBFF", borderColor: "#fff"}]}>
                                <Text style={[styles.infoLabel, customText, {color: isDark ? "#fff" : "#004f7f"}]}>{item.label}</Text>
                                <Text style={[styles.infoValue, customText, {color: isDark ? "#fff" : "#004f7f"}]}>{item.value}</Text>
                            </View>
                        ))}
                    </View>
                </View>

                {/* Detected Condition + Confidence Bar */}
                <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                    <Text style={[styles.sectionTitle, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>Detected Condition</Text>
                    <Text style={[{fontFamily: customText.fontFamily}, styles.diseaseTitle, {color: isDark ? '#fff' : '#004F7F'}]}>{analysis}</Text>
                    <View style={styles.confidenceRow}>
                        <Text style={[{fontFamily: customText.fontFamily}, styles.confLabel, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>AI Confidence</Text>
                        <Text style={[{fontFamily: customText.fontFamily}, styles.confValue, {color: isDark ? '#fff' : '#004F7F'}]}>{confidence}%</Text>
                    </View>
                    <View style={[styles.confBarBg, {backgroundColor: isDark ? '#374151' : '#E5E7EB'}]}>
                        <View style={[styles.confBarFill, {
                            width: `${confidence}%` as any,
                            backgroundColor: confidence >= 80 ? '#22C55E' : confidence >= 60 ? '#F59E0B' : '#EF4444',
                        }]}/>
                    </View>
                </View>

                {/* About This Condition */}
                {description ? (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>About this condition</Text>
                        <Text style={[{fontFamily: customText.fontFamily}, styles.descriptionText, {color: isDark ? '#D1D5DB' : '#374151'}]}>{description}</Text>
                    </View>
                ) : null}

                {/* Tips */}
                {tips.length > 0 && (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: '#00A3A3'}]}>Tips &amp; Recommendations</Text>
                        {tips.map((tip: string, idx: number) => (
                            <View key={idx} style={styles.bulletRow}>
                                <Ionicons name="checkmark-circle" size={18} color="#00A3A3" style={{marginTop: 2}}/>
                                <Text style={[{fontFamily: customText.fontFamily}, styles.bulletText, {color: isDark ? '#D1D5DB' : '#374151'}]}>{tip}</Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* Precautions */}
                {precautions.length > 0 && (
                    <View style={[styles.card, {backgroundColor: isDark ? '#2a1111' : '#FEF2F2', borderColor: '#EF444455', borderWidth: 1}]}>
                        <Text style={[styles.sectionTitle, {color: '#EF4444'}]}>When to see a doctor</Text>
                        {precautions.map((pre: string, idx: number) => (
                            <View key={idx} style={styles.bulletRow}>
                                <Ionicons name="medical" size={16} color="#EF4444" style={{marginTop: 3}}/>
                                <Text style={[{fontFamily: customText.fontFamily}, styles.bulletText, {color: isDark ? '#FECACA' : '#991B1B'}]}>{pre}</Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* Medical References */}
                {sources.length > 0 && (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: '#6B7280'}]}>Medical References</Text>
                        {sources.map((src: string, idx: number) => {
                            const match = src.match(/(.*?)\s*\((https?:\/\/[^)]+)\)/);
                            let displayName = src;
                            let linkUrl = `https://www.google.com/search?q=${encodeURIComponent(src + ' skin condition')}`;
                            if (match) { displayName = match[1].trim(); linkUrl = match[2].trim(); }
                            else if (src.startsWith('http')) { displayName = "View Medical Reference"; linkUrl = src.trim(); }
                            return (
                                <TouchableOpacity key={idx} style={styles.bulletRow} activeOpacity={0.7} onPress={() => Linking.openURL(linkUrl)}>
                                    <Ionicons name="link-outline" size={18} color="#00A3A3" style={{marginTop: 2}}/>
                                    <Text style={[{fontFamily: customText.fontFamily, fontStyle: 'italic', textDecorationLine: 'underline'}, styles.bulletText, {color: '#00A3A3'}]}>
                                        {displayName}
                                    </Text>
                                </TouchableOpacity>
                            );
                        })}
                    </View>
                )}

                {/* Bottom Download Button */}
                <TouchableOpacity
                    style={[styles.downloadButton, {
                        backgroundColor: colors.primary,
                        flexDirection: isArabic ? "row-reverse" : "row",
                        opacity: (isDownloading || !isReady) ? 0.7 : 1,
                    }]}
                    onPress={downloadReport}
                    activeOpacity={0.8}
                    disabled={isDownloading || !isReady}
                >
                    {isDownloading ? (
                        <View style={{flexDirection: "row", alignItems: "center", gap: 10}}>
                            <ActivityIndicator size="small" color="#fff"/>
                            <Text style={[styles.downloadButtonText, {fontFamily: FONT_FAMILY_MAP[settings.fontFamily]}]}>Generating PDF...</Text>
                        </View>
                    ) : (
                        <>
                            <Ionicons name="cloud-download-outline" size={24} color="#FFFFFF"/>
                            <Text style={[styles.downloadButtonText, {fontFamily: customText.fontFamily}]}>{t("downloadAsPDF")}</Text>
                        </>
                    )}
                </TouchableOpacity>

                {/* Disclaimer */}
                <View style={[styles.warningCard, {
                    backgroundColor: isDark ? "#2D2000" : "#FEF3C7",
                    borderColor: isDark ? "#5C4000" : "#FCD34D",
                    flexDirection: isArabic ? "row-reverse" : "row",
                }]}>
                    <Ionicons name="information-circle-outline" size={20} color="#F59E0B"/>
                    <Text style={[styles.warningText, customText, {
                        color: isDark ? "#FCD34D" : "#92400E",
                        textAlign: isArabic ? "right" : "left",
                        fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
                    }]}>
                        {t("medicalDisclaimer")}
                    </Text>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:           {flex: 1},
    header:              {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15},
    backButton:          {width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center'},
    downloadHeaderButton:{width: 40, height: 40, borderRadius: 12, alignItems: 'center', justifyContent: 'center'},
    headerTitle:         {fontSize: 20},
    scrollView:          {flex: 1},
    scrollContent:       {padding: 16, paddingBottom: 40},
    imagePairContainer:  {flexDirection: 'row', justifyContent: 'center', gap: 16, marginBottom: 20},
    imageBox:            {flex: 1, alignItems: 'center'},
    imageLabel:          {fontSize: 12, marginBottom: 6, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5},
    scanImage:           {width: '100%', aspectRatio: 1, borderRadius: 16, overflow: 'hidden'},
    card:                {borderRadius: 16, padding: 20, marginBottom: 16, borderWidth: 1, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2},
    sectionTitle:        {fontSize: 12, fontWeight: '700', textTransform: 'uppercase', marginBottom: 8, letterSpacing: 0.5},
    diseaseTitle:        {fontSize: 24, fontWeight: '800', marginBottom: 16},
    confidenceRow:       {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8},
    confLabel:           {fontSize: 14, fontWeight: '600'},
    confValue:           {fontSize: 18, fontWeight: '800'},
    confBarBg:           {height: 8, borderRadius: 4, overflow: 'hidden'},
    confBarFill:         {height: '100%', borderRadius: 4},
    descriptionText:     {fontSize: 15, lineHeight: 24},
    bulletRow:           {flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12},
    bulletText:          {fontSize: 15, lineHeight: 22, flex: 1},
    infoCard:            {borderRadius: 16, padding: 20, marginBottom: 16, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3},
    infoHeader:          {flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 16, paddingBottom: 12, borderBottomWidth: 1},
    patientGrid:         {flexDirection: 'row', flexWrap: 'wrap', gap: 10},
    patientItem:         {width: '45%', flexGrow: 1, borderRadius: 10, padding: 10, borderWidth: 1},
    infoGrid:            {gap: 16},
    infoItem:            {flexDirection: 'row', alignItems: 'center', gap: 12},
    infoTextContainer:   {flex: 1},
    infoLabel:           {fontSize: 12, marginBottom: 2},
    infoValue:           {fontSize: 15, fontWeight: '600'},
    downloadButton:      {flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, borderRadius: 16, marginBottom: 16, gap: 10},
    downloadButtonText:  {fontSize: 16, color: '#FFFFFF'},
    warningCard:         {flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 12, gap: 12, borderWidth: 1},
    warningText:         {flex: 1, fontSize: 13, lineHeight: 18},
    imageBadge:          {position: 'absolute', bottom: 8, right: 8, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8},
    imageBadgeText:      {color: '#FFFFFF', fontSize: 11, fontWeight: 'bold'},
});