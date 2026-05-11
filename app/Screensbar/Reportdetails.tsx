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
                    FileSystem.cacheDirectory + 'tmp_img_' + Date.now() + '.jpg'
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

const confColor = (c: number) =>
    c >= 80 ? '#22C55E' : c >= 60 ? '#F59E0B' : '#EF4444';

// ── buildReportHTML ───────────────────────────────────────────
const buildReportHTML = (params: {
    reportIndex:        number;
    date:               string;
    bodyView:           string;
    moleId:             string;
    analysis:           string;
    imageBase64:        string;
    maskBase64:         string;
    frontBody:          string;
    backBody:           string;
    platform:           string;
    description:        string;
    patientName:        string;
    age:                string;
    gender:             string;
    hairColor:          string;
    eyeColor:           string;
    skinColor:          string;
    confidence:         number;
    diseaseDescription: string;
    tips:               string[];
    precautions:        string[];
    sources:            string[];
}) => {
    const cc  = confColor(params.confidence);
    const loc = params.bodyView === 'front'
        ? params.frontBody
        : params.bodyView === 'back'
        ? params.backBody
        : 'N/A';

    const patCell = (label: string, value: string) => `
      <td style="width:33.3%;padding:2px 3px;vertical-align:top">
        <div style="background:#F4FBFF;border-radius:6px;padding:5px 8px;border:1px solid #C5E3ED">
          <div style="font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.3px;margin-bottom:1px">${label}</div>
          <div style="font-size:10px;font-weight:bold;color:#1F2937;font-family:system-ui,sans-serif">${value}</div>
        </div>
      </td>`;

    const infoCell = (label: string, value: string) => `
      <td style="width:50%;padding:2px 3px;vertical-align:top">
        <div style="background:#F4FBFF;border-radius:7px;padding:6px 8px;border:1px solid #C5E3ED">
          <div style="font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.3px;margin-bottom:2px">${label}</div>
          <div style="font-size:11px;font-weight:bold;color:#004F7F;font-family:system-ui,sans-serif">${value}</div>
        </div>
      </td>`;

    const bullet = (text: string, dotColor: string, textColor: string) => `
      <div style="display:flex;align-items:flex-start;gap:7px;margin-bottom:4px">
        <div style="width:6px;height:6px;border-radius:3px;background:${dotColor};flex-shrink:0;margin-top:4px"></div>
        <div style="font-size:10px;color:${textColor};line-height:1.5;font-family:system-ui,sans-serif;flex:1">${text}</div>
      </div>`;

    const tipsHtml = params.tips.length > 0 ? `
      <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:6px 18px">
        <div style="font-size:10px;font-weight:700;color:#00A3A3;text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px;font-family:system-ui,sans-serif">✅ Tips &amp; Recommendations</div>
        ${params.tips.map(t => bullet(t, '#00A3A3', '#374151')).join('')}
      </div>` : '';

    const precHtml = params.precautions.length > 0 ? `
      <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:6px 18px">
        <div style="background:#FEF2F2;border-left:3px solid #EF4444;border-radius:6px;padding:7px 10px">
          <div style="font-size:10px;font-weight:700;color:#EF4444;text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px;font-family:system-ui,sans-serif">⚠️ When to See a Doctor</div>
          ${params.precautions.map(p => bullet(p, '#EF4444', '#991B1B')).join('')}
        </div>
      </div>` : '';

    const srcHtml = params.sources.length > 0 ? `
      <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:6px 18px">
        <div style="font-size:10px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px;font-family:system-ui,sans-serif">🔗 Medical References</div>
        ${params.sources.map(src => {
            const m    = src.match(/(.*?)\s*\((https?:\/\/[^)]+)\)/);
            const name = m ? m[1].trim() : src.startsWith('http') ? 'View Reference' : src;
            return bullet(name, '#9CA3AF', '#00A3A3');
        }).join('')}
      </div>` : '';

    return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,'Times New Roman',serif;background:#D8E9F0;padding:10px}
.page{max-width:700px;margin:0 auto;background:#D8E9F0;border-radius:12px;overflow:hidden}
</style>
</head>
<body>
<div class="page">

  <div style="background:#004F7F;padding:12px 22px 9px;text-align:center">
    <div style="font-size:28px;font-weight:bold;color:#fff;letter-spacing:2px">
      <span style="color:#00A3A3;font-size:34px">S</span>kinsight
    </div>
    <div style="color:#C5E3ED;font-size:9px;margin-top:2px;font-style:italic;letter-spacing:3px">Snap. Detect. Protect.</div>
    <div style="width:36px;height:3px;background:#00A3A3;margin:5px auto 0;border-radius:10px"></div>
  </div>

  <div style="background:#00A3A3;padding:4px 18px;text-align:center">
    <p style="color:#fff;font-size:9px;font-style:italic;letter-spacing:.5px">Skin Analysis Report</p>
  </div>

  <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:6px 18px;display:flex;justify-content:space-between;align-items:center">
    <div style="font-size:15px;font-weight:bold;color:#004F7F">Report #${params.reportIndex + 1}</div>
    <div style="font-size:9px;color:#6B7280;font-family:system-ui,sans-serif">${params.date}</div>
  </div>

  <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:6px 18px;border-top:1px solid #E5F0F6">
    <div style="font-size:9px;font-weight:bold;color:#004F7F;margin-bottom:4px;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.5px">Patient Information</div>
    <table style="width:100%;border-collapse:collapse">
      <tr>
        ${patCell("Patient Name", params.patientName)}
        ${patCell("Age", params.age)}
        ${patCell("Gender", params.gender)}
      </tr>
      <tr>
        ${patCell("Hair Color", params.hairColor)}
        ${patCell("Eye Color", params.eyeColor)}
        ${patCell("Skin Tone", params.skinColor)}
      </tr>
    </table>
  </div>

  <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:7px 18px">
    <table style="width:100%;border-collapse:collapse">
      <tr>
        <td style="width:${params.maskBase64 ? '50%' : '100%'};padding:0 ${params.maskBase64 ? '6px' : '0'} 0 0;vertical-align:top;text-align:center">
          <div style="font-size:9px;font-weight:700;color:#6B7280;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:5px">Original Image</div>
          ${params.imageBase64
            ? `<img src="${params.imageBase64}" style="width:100%;max-height:130px;border-radius:10px;border:3px solid #C5E3ED;object-fit:cover;display:block"/>`
            : `<div style="width:100%;height:100px;border-radius:10px;border:2px dashed #C5E3ED;background:#F4FBFF;display:flex;align-items:center;justify-content:center;color:#9CA3AF;font-size:9px;font-family:system-ui,sans-serif">No Image</div>`}
        </td>
        ${params.maskBase64 ? `
        <td style="width:50%;padding:0 0 0 6px;vertical-align:top;text-align:center">
          <div style="font-size:9px;font-weight:700;color:#00E5FF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:5px">U-Net Mask</div>
          <img src="${params.maskBase64}" style="width:100%;max-height:130px;border-radius:10px;border:3px solid #00E5FF;object-fit:contain;background:#000000;display:block"/>
        </td>` : ''}
      </tr>
    </table>
  </div>

  <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:5px 18px">
    <table style="width:100%;border-collapse:collapse">
      <tr>
        ${infoCell("Location", loc)}
        ${infoCell("Platform", params.platform)}
      </tr>
    </table>
  </div>

  <div style="background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:8px 18px">
    <div style="font-size:12px;font-weight:bold;color:#004F7F;padding-bottom:6px;border-bottom:1px solid #E5E7EB;margin-bottom:7px">Analysis Results</div>
    <div style="background:#E8F4F8;border-radius:8px;padding:8px 11px;border:1px solid #C5E3ED;margin-bottom:7px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div>
          <div style="font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.3px;margin-bottom:2px">Detected Condition</div>
          <div style="font-size:14px;font-weight:bold;color:${cc};font-family:system-ui,sans-serif">${params.analysis}</div>
        </div>
        ${params.confidence > 0 ? `
        <div style="text-align:right;min-width:72px">
          <div style="font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.3px;margin-bottom:2px">AI Confidence</div>
          <div style="font-size:16px;font-weight:bold;font-family:system-ui,sans-serif;color:${cc}">${params.confidence}%</div>
        </div>` : ''}
      </div>
      ${params.confidence > 0 ? `
      <div style="width:100%;height:5px;background:#E5E7EB;border-radius:3px;overflow:hidden">
        <div style="height:100%;width:${params.confidence}%;border-radius:3px;background:${cc}"></div>
      </div>` : ''}
    </div>
    ${params.diseaseDescription ? `
    <div style="font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.3px;margin-bottom:4px">About This Condition</div>
    <div style="background:#D8E9F0;border-radius:7px;padding:8px 10px;border:1px solid #C5E3ED">
      <p style="font-size:10px;color:#374151;line-height:1.55;font-family:system-ui,sans-serif">${params.diseaseDescription}</p>
    </div>` : ''}
  </div>

  ${tipsHtml}
  ${precHtml}
  ${srcHtml}

  <div style="background:#004F7F;padding:10px 18px;text-align:center">
    <div style="width:26px;height:2px;background:#00A3A3;margin:0 auto 6px;border-radius:10px"></div>
    <div style="font-size:12px;font-weight:bold;color:#fff;margin-bottom:2px"><span style="color:#00A3A3">S</span>kinsight</div>
    <div style="color:#C5E3ED;font-size:7px;font-family:system-ui,sans-serif">© 2026 SkinSight. All rights reserved.</div>
    <div style="color:#8ab4c9;font-size:6px;margin-top:2px;font-family:system-ui,sans-serif">📧 skinsight.help.2025@gmail.com</div>
  </div>

</div>
</body>
</html>`;
};

// ══════════════════════════════════════════════════════════════
// ReportDetailsPage Component
// ══════════════════════════════════════════════════════════════
export default function ReportDetailsPage() {
    const router = useRouter();
    const {colors, isDark} = useTheme();
    const {settings} = useCustomize();
    const {t, isArabic} = useTranslation(settings.language);
    const [isDownloading, setIsDownloading] = useState(false);
    const [isReady, setIsReady] = useState(false);

    const rawParams = useLocalSearchParams();

    const moleId      = (rawParams.moleId      as string) || '';
    const photoUri    = (rawParams.photoUri     as string) || '';
    const timestamp   = parseInt((rawParams.timestamp    as string) || '0');
    const bodyView    = (rawParams.bodyView     as string) || '';
    const reportIndex = parseInt((rawParams.reportIndex  as string) || '0');
    const source      = (rawParams.source       as string) || '';
    const descParam   = (rawParams.description  as string) || '';

    const aiResult = useMemo(() => {
        try {
            if (rawParams.result) return JSON.parse(rawParams.result as string);
        } catch (e) {}
        return {};
    }, [rawParams.result]);

    const analysis    = aiResult.disease      || (rawParams.analysis as string) || t('analysisInProgress');
    const confidence  = aiResult.confidence   || 0;
    const maskUrl     = aiResult.segmentedUrl || aiResult.segmented_url || '';
    const description = aiResult.description  || '';
    const tips        = aiResult.tips         || [];
    const precautions = aiResult.precautions  || [];
    const sources     = aiResult.sources      || [];

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
                setGender(d.gender ? d.gender.charAt(0).toUpperCase() + d.gender.slice(1) : 'N/A');
                setHairColor(d.hairColor || 'N/A');
                setEyeColor(d.eyeColor   || 'N/A');
                setSkinColor(skinColorLabel(d.skinColor));
                if (d.birthYear && d.birthMonth && d.birthDay) {
                    const dob = new Date(d.birthYear, d.birthMonth - 1, d.birthDay);
                    setAge(`${Math.floor((Date.now() - dob.getTime()) / (1000 * 60 * 60 * 24 * 365.25))} years`);
                } else if (d.age) {
                    setAge(typeof d.age === 'number' ? `${d.age} years` : d.age);
                }
            }
        };
        loadPatientData().catch(() => {}).finally(() => setIsReady(true));
    }, []);

    const customText = {
        fontSize:   settings.fontSize,
        color:      settings.textColor,
        fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
    };
    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const formatDate = useCallback((ts: number) =>
        new Date(ts).toLocaleDateString(isArabic ? 'ar-EG' : 'en-US', {
            month: 'long', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        }), [isArabic]);

    // ── Download ──────────────────────────────────────────────
    const downloadReport = useCallback(async () => {
        if (isDownloading || !isReady) return;
        setIsDownloading(true);
        try {
          const [imageBase64, maskBase64] = await Promise.all([
            getImageBase64(photoUri),
            maskUrl ? getImageBase64(maskUrl) : Promise.resolve(""),
          ]);

            const html = buildReportHTML({
                reportIndex,
                date:               formatDate(timestamp),
                bodyView,
                moleId,
                analysis,
                imageBase64,
                maskBase64,
                frontBody:          t('frontBody'),
                backBody:           t('backBody'),
                patientName,
                age,
                gender,
                hairColor,
                eyeColor,
                skinColor,
                platform:           getPlatform(source),
                description:        descParam || 'N/A',
                confidence,
                diseaseDescription: description,
                tips,
                precautions,
                sources,
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
        isDownloading, isReady, photoUri, maskUrl, reportIndex, timestamp,
        bodyView, moleId, analysis, patientName, age, gender, hairColor,
        eyeColor, skinColor, source, descParam, confidence, description,
        tips, precautions, sources, formatDate, t,
    ]);

    return (
        <SafeAreaView style={[styles.container, {backgroundColor: pageBg}]} edges={["top"]}>
            <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg}/>

            {/* Header */}
            <View style={[styles.header, {backgroundColor: colors.card}]}>
                <TouchableOpacity
                    style={[styles.backButton, {borderColor: colors.border}]}
                    onPress={() => router.back()}
                >
                    <Ionicons name="chevron-back" size={24} color={colors.text}/>
                </TouchableOpacity>
                <Text style={[
                    styles.headerTitle, customText,
                    {fontFamily: FONT_FAMILY_MAP[settings.fontFamily], color: isDark ? "#fff" : "#000"},
                ]}>
                    {t("reportDetails")}
                </Text>
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

            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                {/* Side-by-Side Images */}
                <View style={styles.imagePairContainer}>
                    <View style={styles.imageBox}>
                        <Text style={[styles.imageLabel, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>
                            Original
                        </Text>
                        <Image
                            source={{uri: photoUri}}
                            style={[styles.scanImage, {backgroundColor: colors.card}]}
                            resizeMode="cover"
                        />
                        <View style={styles.imageBadge}>
                            <Text style={[styles.imageBadgeText, {fontFamily: customText.fontFamily}]}>
                                {bodyView === "front" ? t("frontBody")
                                    : bodyView === "back" ? t("backBody") : "N/A"}
                            </Text>
                        </View>
                    </View>
                    {maskUrl ? (
                        <View style={styles.imageBox}>
                            <Text style={[styles.imageLabel, {color: '#00E5FF'}]}>
                                U-Net Mask
                            </Text>
                            <Image
                                source={{uri: maskUrl}}
                                style={[styles.scanImage, {
                                    borderColor:     '#00E5FF',
                                    borderWidth:     2,
                                    backgroundColor: '#000000',
                                }]}
                                resizeMode="contain"
                            />
                        </View>
                    ) : null}
                </View>

                {/* Patient Information */}
                <View style={[styles.infoCard, {backgroundColor: colors.card}]}>
                    <View style={[styles.infoHeader, {borderBottomColor: colors.border}]}>
                        <Ionicons name="person-outline" size={20} color={isDark ? "#fff" : "#004f7f"}/>
                        <Text style={[
                            styles.sectionTitle, customText,
                            {fontFamily: FONT_FAMILY_MAP[settings.fontFamily], color: isDark ? "#fff" : "#004f7f"},
                        ]}>
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
                            <View key={item.label} style={[
                                styles.patientItem,
                                {backgroundColor: isDark ? "#004f7f" : "#F4FBFF", borderColor: "#fff"},
                            ]}>
                                <Text style={[styles.infoLabel, customText, {color: isDark ? "#fff" : "#004f7f"}]}>
                                    {item.label}
                                </Text>
                                <Text style={[styles.infoValue, customText, {color: isDark ? "#fff" : "#004f7f"}]}>
                                    {item.value}
                                </Text>
                            </View>
                        ))}
                    </View>
                </View>

                {/* Detected Condition + Confidence */}
                <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                    <Text style={[styles.sectionTitle, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>
                        Detected Condition
                    </Text>
                    <Text style={[styles.diseaseTitle, {fontFamily: customText.fontFamily, color: confColor(confidence)}]}>
                        {analysis}
                    </Text>
                    <View style={styles.confidenceRow}>
                        <Text style={[styles.confLabel, {fontFamily: customText.fontFamily, color: isDark ? '#9CA3AF' : '#6B7280'}]}>
                            AI Confidence
                        </Text>
                        <Text style={[styles.confValue, {fontFamily: customText.fontFamily, color: confColor(confidence)}]}>
                            {confidence}%
                        </Text>
                    </View>
                    <View style={[styles.confBarBg, {backgroundColor: isDark ? '#374151' : '#E5E7EB'}]}>
                        <View style={[styles.confBarFill, {width: `${confidence}%` as any, backgroundColor: confColor(confidence)}]}/>
                    </View>
                </View>

                {/* About This Condition */}
                {description ? (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: isDark ? '#9CA3AF' : '#6B7280'}]}>
                            About this condition
                        </Text>
                        <Text style={[styles.descriptionText, {fontFamily: customText.fontFamily, color: isDark ? '#D1D5DB' : '#374151'}]}>
                            {description}
                        </Text>
                    </View>
                ) : null}

                {/* Tips */}
                {tips.length > 0 && (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: '#00A3A3'}]}>
                            Tips &amp; Recommendations
                        </Text>
                        {tips.map((tip: string, idx: number) => (
                            <View key={idx} style={styles.bulletRow}>
                                <Ionicons name="checkmark-circle" size={18} color="#00A3A3" style={{marginTop: 2}}/>
                                <Text style={[styles.bulletText, {fontFamily: customText.fontFamily, color: isDark ? '#D1D5DB' : '#374151'}]}>
                                    {tip}
                                </Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* Precautions */}
                {precautions.length > 0 && (
                    <View style={[styles.card, {
                        backgroundColor: isDark ? '#2a1111' : '#FEF2F2',
                        borderColor: '#EF444455', borderWidth: 1,
                    }]}>
                        <Text style={[styles.sectionTitle, {color: '#EF4444'}]}>
                            When to see a doctor
                        </Text>
                        {precautions.map((pre: string, idx: number) => (
                            <View key={idx} style={styles.bulletRow}>
                                <Ionicons name="medical" size={16} color="#EF4444" style={{marginTop: 3}}/>
                                <Text style={[styles.bulletText, {fontFamily: customText.fontFamily, color: isDark ? '#FECACA' : '#991B1B'}]}>
                                    {pre}
                                </Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* Medical References */}
                {sources.length > 0 && (
                    <View style={[styles.card, {backgroundColor: colors.card, borderColor: colors.border}]}>
                        <Text style={[styles.sectionTitle, {color: '#6B7280'}]}>
                            Medical References
                        </Text>
                        {sources.map((src: string, idx: number) => {
                            const match = src.match(/(.*?)\s*\((https?:\/\/[^)]+)\)/);
                            let displayName = src;
                            let linkUrl = `https://www.google.com/search?q=${encodeURIComponent(src + ' skin condition')}`;
                            if (match) {
                                displayName = match[1].trim();
                                linkUrl     = match[2].trim();
                            } else if (src.startsWith('http')) {
                                displayName = "View Medical Reference";
                                linkUrl     = src.trim();
                            }
                            return (
                                <TouchableOpacity
                                    key={idx}
                                    style={styles.bulletRow}
                                    activeOpacity={0.7}
                                    onPress={() => Linking.openURL(linkUrl)}
                                >
                                    <Ionicons name="link-outline" size={18} color="#00A3A3" style={{marginTop: 2}}/>
                                    <Text style={[
                                        styles.bulletText,
                                        {fontFamily: customText.fontFamily, fontStyle: 'italic', textDecorationLine: 'underline', color: '#00A3A3'},
                                    ]}>
                                        {displayName}
                                    </Text>
                                </TouchableOpacity>
                            );
                        })}
                    </View>
                )}

                {/* Download Button */}
                <TouchableOpacity
                    style={[styles.downloadButton, {
                        backgroundColor: colors.primary,
                        flexDirection:   isArabic ? "row-reverse" : "row",
                        opacity:         (isDownloading || !isReady) ? 0.7 : 1,
                    }]}
                    onPress={downloadReport}
                    activeOpacity={0.8}
                    disabled={isDownloading || !isReady}
                >
                    {isDownloading ? (
                        <View style={{flexDirection: "row", alignItems: "center", gap: 10}}>
                            <ActivityIndicator size="small" color="#fff"/>
                            <Text style={[styles.downloadButtonText, {fontFamily: FONT_FAMILY_MAP[settings.fontFamily]}]}>
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

                {/* Disclaimer - في الـ UI بس، مش في الـ PDF */}
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
    container:            {flex: 1},
    header:               {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15},
    backButton:           {width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center'},
    downloadHeaderButton: {width: 40, height: 40, borderRadius: 12, alignItems: 'center', justifyContent: 'center'},
    headerTitle:          {fontSize: 20},
    scrollView:           {flex: 1},
    scrollContent:        {padding: 16, paddingBottom: 40},
    imagePairContainer:   {flexDirection: 'row', justifyContent: 'center', gap: 16, marginBottom: 20},
    imageBox:             {flex: 1, alignItems: 'center'},
    imageLabel:           {fontSize: 12, marginBottom: 6, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5},
    scanImage:            {width: '100%', aspectRatio: 1, borderRadius: 16, overflow: 'hidden'},
    card:                 {borderRadius: 16, padding: 20, marginBottom: 16, borderWidth: 1, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2},
    sectionTitle:         {fontSize: 12, fontWeight: '700', textTransform: 'uppercase', marginBottom: 8, letterSpacing: 0.5},
    diseaseTitle:         {fontSize: 24, fontWeight: '800', marginBottom: 16},
    confidenceRow:        {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8},
    confLabel:            {fontSize: 14, fontWeight: '600'},
    confValue:            {fontSize: 18, fontWeight: '800'},
    confBarBg:            {height: 8, borderRadius: 4, overflow: 'hidden'},
    confBarFill:          {height: '100%', borderRadius: 4},
    descriptionText:      {fontSize: 15, lineHeight: 24},
    bulletRow:            {flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12},
    bulletText:           {fontSize: 15, lineHeight: 22, flex: 1},
    infoCard:             {borderRadius: 16, padding: 20, marginBottom: 16, shadowColor: '#000', shadowOffset: {width: 0, height: 2}, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3},
    infoHeader:           {flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 16, paddingBottom: 12, borderBottomWidth: 1},
    patientGrid:          {flexDirection: 'row', flexWrap: 'wrap', gap: 10},
    patientItem:          {width: '45%', flexGrow: 1, borderRadius: 10, padding: 10, borderWidth: 1},
    infoGrid:             {gap: 16},
    infoItem:             {flexDirection: 'row', alignItems: 'center', gap: 12},
    infoTextContainer:    {flex: 1},
    infoLabel:            {fontSize: 12, marginBottom: 2},
    infoValue:            {fontSize: 15, fontWeight: '600'},
    downloadButton:       {flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, borderRadius: 16, marginBottom: 16, gap: 10},
    downloadButtonText:   {fontSize: 16, color: '#FFFFFF'},
    warningCard:          {flexDirection: 'row', alignItems: 'center', padding: 16, borderRadius: 12, gap: 12, borderWidth: 1},
    warningText:          {flex: 1, fontSize: 13, lineHeight: 18},
    imageBadge:           {position: 'absolute', bottom: 8, right: 8, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8},
    imageBadgeText:       {color: '#FFFFFF', fontSize: 11, fontWeight: 'bold'},
});