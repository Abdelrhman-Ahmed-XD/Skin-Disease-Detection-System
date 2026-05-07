import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';
import { useFocusEffect, useRouter } from 'expo-router';
import { shareAsync } from 'expo-sharing';
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator, Alert, Dimensions, Image, Platform, ScrollView,
    StatusBar, StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';
import { loadAllScansFromFirestore } from '../../Firebase/firestoreService';

// ── Custom Icon Images ─────────────────────────────────────────
const Icons = {
  home:       require('../../assets/Icons/home.png'),
  reports:    require('../../assets/Icons/Reports.png'),
  history:    require('../../assets/Icons/history.png'),
  settings:   require('../../assets/Icons/setting.png'),
  smartphone: require('../../assets/Icons/smartphone.png'),
  monitor:    require('../../assets/Icons/monitor.png'),
};

const { width } = Dimensions.get('window');

type Mole = {
    id: string; x: number; y: number; timestamp: number;
    photoUri?: string; bodyView: 'front' | 'back' | 'N/A' | string;
    analysis?: string; source?: string; description?: string;
};

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
            } catch { return uri; }
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

const isWeb = (source?: string): boolean => {
    if (!source) return false;
    return source.toLowerCase().includes('web');
};

const getPlatform = (source?: string): string =>
    isWeb(source) ? 'Web' : 'Mobile App';

// ── Single Report HTML ────────────────────────────────────────
const buildReportHTML = (params: {
    reportIndex: number; date: string; bodyView: string;
    moleId: string; analysis: string; imageBase64: string;
    frontBody: string; backBody: string; platform: string; description: string;
    patientName: string; age: string; gender: string;
    hairColor: string; eyeColor: string; skinColor: string;
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
    <div class="abox"><p class="atxt">${params.analysis}</p></div>
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

// ── All Reports HTML ──────────────────────────────────────────
const buildAllReportsHTML = (params: {
    rows: Array<{
        index: number; date: string; bodyView: string; analysis: string;
        imageBase64: string; frontBody: string; backBody: string;
        description: string; platform: string;
    }>;
    patientName: string; age: string; gender: string;
    hairColor: string; eyeColor: string; skinColor: string;
    generatedDate: string;
}) => `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,'Times New Roman',serif;background:#D8E9F0;padding:18px 12px}
.page{max-width:960px;margin:0 auto;background:#D8E9F0;border-radius:14px;overflow:hidden}
.header{background:#004F7F;padding:22px 22px 16px;text-align:center}
.brand{font-size:38px;font-weight:bold;color:#fff;letter-spacing:2px}
.brand-s{color:#00A3A3;font-size:46px}
.tagline{color:#C5E3ED;font-size:11px;margin-top:3px;font-style:italic;letter-spacing:3px}
.hdiv{width:46px;height:3px;background:#00A3A3;margin:8px auto 0;border-radius:10px}
.banner{background:#00A3A3;padding:7px 18px;text-align:center}
.banner p{color:#fff;font-size:11px;font-style:italic;letter-spacing:.5px}
.meta{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:12px 22px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:5px}
.mtitle{font-size:17px;font-weight:bold;color:#004F7F}
.mdate{font-size:10px;color:#6B7280;font-family:system-ui,sans-serif}
.psec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:10px 22px;border-top:1px solid #E5F0F6}
.ptitle{font-size:11px;font-weight:bold;color:#004F7F;margin-bottom:7px;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.5px}
.pgrid{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}
.pi{background:#F4FBFF;border-radius:6px;padding:6px 8px;border:1px solid #C5E3ED}
.pl{font-size:7px;color:#9CA3AF;font-family:system-ui,sans-serif;margin-bottom:2px;text-transform:uppercase;letter-spacing:.3px}
.pv{font-size:10px;font-weight:bold;color:#1F2937;font-family:system-ui,sans-serif}
.stats{background:#004F7F;padding:11px 22px;display:flex;justify-content:space-around;flex-wrap:wrap;gap:6px}
.si{text-align:center}
.sv{font-size:19px;font-weight:bold;color:#00A3A3}
.sl{font-size:8px;color:#C5E3ED;font-family:system-ui,sans-serif;margin-top:1px;text-transform:uppercase;letter-spacing:.4px}
.tsec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:14px 18px}
.stitle{font-size:13px;font-weight:bold;color:#004F7F;margin-bottom:10px;padding-bottom:7px;border-bottom:2px solid #C5E3ED;font-family:system-ui,sans-serif}
table{width:100%;border-collapse:collapse;font-family:system-ui,sans-serif}
thead tr{background:#004F7F}
thead th{color:#fff;font-size:10px;font-weight:600;padding:9px 7px;text-align:left;letter-spacing:.3px}
thead th:first-child{border-radius:5px 0 0 0}
thead th:last-child{border-radius:0 5px 0 0}
tbody tr:nth-child(even){background:#F4FBFF}
tbody tr:nth-child(odd){background:#fff}
tbody tr{border-bottom:1px solid #E5F0F6}
tbody tr:last-child{border-bottom:none}
td{padding:8px 7px;vertical-align:middle}
.tnum{font-size:12px;font-weight:bold;color:#004F7F;text-align:center}
.timg{text-align:center}
.timg img{width:50px;height:50px;border-radius:6px;border:2px solid #C5E3ED;object-fit:cover;display:block;margin:0 auto}
.timg-ph{width:50px;height:50px;border-radius:6px;border:2px dashed #C5E3ED;display:flex;align-items:center;justify-content:center;color:#9CA3AF;font-size:8px;margin:0 auto;background:#F4FBFF;text-align:center;padding:3px;line-height:1.3}
.tdate{font-size:10px;color:#374151;white-space:nowrap}
.loc-badge{display:inline-block;background:#E8F4F8;color:#004F7F;border:1px solid #C5E3ED;border-radius:4px;padding:2px 6px;font-size:9px;font-weight:600}
.plat-badge{display:inline-block;border-radius:4px;padding:2px 6px;font-size:9px;font-weight:600}
.plat-app{background:#E8F4F8;color:#004F7F;border:1px solid #C5E3ED}
.plat-web{background:#E6F4EA;color:#1A6B35;border:1px solid #A8D5B5}
.tdesc{font-size:9px;color:#374151;max-width:110px}
.tanal{font-size:9px;color:#374151;line-height:1.5;max-width:210px}
.wsec{background:#fff;border-left:1px solid #C5E3ED;border-right:1px solid #C5E3ED;padding:0 18px 12px}
.wbox{background:#fff3cd;border-left:4px solid #ffc107;border-radius:6px;padding:8px 12px}
.wtxt{font-size:9px;color:#856404;line-height:1.5;font-family:system-ui,sans-serif}
.footer{background:#004F7F;padding:15px 18px;text-align:center}
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
  <div class="banner"><p>Complete Skin Analysis Summary</p></div>
  <div class="meta">
    <div class="mtitle">All Reports — Full History</div>
    <div class="mdate">Generated: ${params.generatedDate}</div>
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
  <div class="stats">
    <div class="si"><div class="sv">${params.rows.length}</div><div class="sl">Total Reports</div></div>
    <div class="si"><div class="sv">${params.rows.filter(r => r.bodyView === 'front').length}</div><div class="sl">Front Body</div></div>
    <div class="si"><div class="sv">${params.rows.filter(r => r.bodyView === 'back').length}</div><div class="sl">Back Body</div></div>
    <div class="si"><div class="sv">${params.rows.filter(r => r.platform === 'Mobile App').length}</div><div class="sl">App Scans</div></div>
    <div class="si"><div class="sv">${params.rows.filter(r => r.platform === 'Web').length}</div><div class="sl">Web Scans</div></div>
  </div>
  <div class="tsec">
    <div class="stitle">📋 Scan History Table</div>
    <table>
      <thead>
        <tr>
          <th style="width:34px">#</th>
          <th style="width:60px">Image</th>
          <th style="width:78px">Date</th>
          <th style="width:68px">Location</th>
          <th style="width:70px">Platform</th>
          <th style="width:105px">Description</th>
          <th>Analysis Result</th>
        </tr>
      </thead>
      <tbody>
        ${params.rows.map(row => `
        <tr>
          <td class="tnum">${row.index + 1}</td>
          <td class="timg">
            ${row.imageBase64
              ? `<img src="${row.imageBase64}" alt="scan"/>`
              : `<div class="timg-ph">No Image</div>`}
          </td>
          <td class="tdate">${row.date}</td>
          <td><span class="loc-badge">${row.bodyView === 'front' ? row.frontBody : row.bodyView === 'back' ? row.backBody : 'N/A'}</span></td>
          <td><span class="plat-badge ${row.platform === 'Web' ? 'plat-web' : 'plat-app'}">${row.platform}</span></td>
          <td class="tdesc">${row.description || 'N/A'}</td>
          <td class="tanal">${row.analysis || 'Analysis in progress...'}</td>
        </tr>`).join('')}
      </tbody>
    </table>
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

export default function ReportsPage() {
    const router = useRouter();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const customText = {
        fontSize:   settings.fontSize,
        color:      settings.textColor,
        fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
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
            const skinMap: Record<string,string> = {
                '#F5E0D3':'Very Light','#EACAA7':'Light','#D1A67A':'Medium',
                '#B57D50':'Tan','#A05C38':'Brown','#8B4513':'Dark Brown',
                '#7A3E11':'Deep','#603311':'Ebony',
            };
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
            // ✅ FIX: فلتر صارم — بس moles عندها photoUri حقيقي (مش فاضي أو spaces)
            const filtered = data.filter(
                (m: Mole) => m.photoUri && m.photoUri.trim() !== ''
            );
            // الأقدم = index 0 = Report #1
            const sorted = filtered.sort((a: Mole, b: Mole) => a.timestamp - b.timestamp);
            setMoles(sorted);
        } catch (err) {
            console.log('Error loading moles:', err);
        } finally {
            setLoading(false);
        }
    };

    // ── Download Single PDF ──────────────────────────────────
    const downloadSingleReport = async (mole: Mole, reportNumber: number) => {
        if (downloadingId || downloadingAll) return;
        try {
            setDownloadingId(mole.id);
            if (!mole.photoUri) return;
            const imageBase64 = await getImageBase64(mole.photoUri);
            const html = buildReportHTML({
                reportIndex: reportNumber - 1,
                date: new Date(mole.timestamp).toLocaleDateString(
                    isArabic ? 'ar-EG' : 'en-US',
                    { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }
                ),
                bodyView:    mole.bodyView,
                moleId:      mole.id,
                analysis:    mole.analysis || t('analysisInProgress'),
                imageBase64,
                frontBody:   t('frontBody'),
                backBody:    t('backBody'),
                patientName, age, gender, hairColor, eyeColor, skinColor,
                platform:    getPlatform(mole.source),
                description: mole.description || 'N/A',
            });
            const { uri } = await Print.printToFileAsync({ html, base64: false });
            await savePDF(uri, `SkinSight_Report_${reportNumber}.pdf`);
        } catch (error: any) {
            if (!String(error?.message || '').includes('Another share request')) {
                Alert.alert(t('error'), 'Failed to download report.');
            }
        } finally {
            setDownloadingId(null);
        }
    };

    // ── Download All PDF ─────────────────────────────────────
    const downloadAllReports = async () => {
        if (downloadingId || downloadingAll) return;
        try {
            setDownloadingAll(true);
            if (moles.length === 0) {
                Alert.alert(t('noReportsYet'), t('noReportsToDownload'));
                return;
            }
            const rows: Array<{
                index: number; date: string; bodyView: string; analysis: string;
                imageBase64: string; frontBody: string; backBody: string;
                description: string; platform: string;
            }> = [];
            for (let i = 0; i < moles.length; i++) {
                const mole = moles[i];
                const imageBase64 = await getImageBase64(mole.photoUri || '');
                rows.push({
                    index:       i,
                    date:        new Date(mole.timestamp).toLocaleDateString(
                                     isArabic ? 'ar-EG' : 'en-US',
                                     { month: 'short', day: 'numeric', year: 'numeric' }
                                 ),
                    bodyView:    mole.bodyView,
                    analysis:    mole.analysis || t('analysisInProgress'),
                    imageBase64,
                    frontBody:   t('frontBody'),
                    backBody:    t('backBody'),
                    description: mole.description || 'N/A',
                    platform:    getPlatform(mole.source),
                });
            }
            const generatedDate = new Date().toLocaleDateString(
                isArabic ? 'ar-EG' : 'en-US',
                { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }
            );
            const html = buildAllReportsHTML({
                rows, patientName, age, gender, hairColor, eyeColor, skinColor, generatedDate,
            });
            const { uri } = await Print.printToFileAsync({ html, base64: false });
            await savePDF(uri, `SkinSight_All_Reports.pdf`);
        } catch (error: any) {
            if (!String(error?.message || '').includes('Another share request')) {
                Alert.alert(t('error'), 'Failed to download reports.');
            }
        } finally {
            setDownloadingAll(false);
        }
    };

    const formatDate = (timestamp: number) =>
        new Date(timestamp).toLocaleDateString(
            isArabic ? 'ar-EG' : 'en-US',
            { month: 'short', day: 'numeric', year: 'numeric' }
        );

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

    // عكس للعرض فقط: الأحدث يظهر فوق، الأقدم في الأسفل
    const displayMoles = [...moles].reverse();

    return (
      <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
        <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

        {/* ── Header ── */}
        <View style={[styles.header, { backgroundColor: colors.card }]}>
          <TouchableOpacity
            style={[styles.backButton, { borderColor: colors.border }]}
            onPress={() => router.back()}
          >
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },,styles.headerTitle, customText, { color: isDark ? "#fff" : "#374151" }]}>
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
              <Text style={[styles.loadingText, customText]}>{t("loadingReports")}</Text>
            </View>
          ) : moles.length === 0 ? (
            <View style={styles.emptyContainer}>
              <Image source={Icons.reports} style={styles.emptyIcon} resizeMode="contain" />
              <Text style={[styles.emptyTitle, customText, { color: isDark ? "#fff" : "#374151" }]}>
                {t("noReportsYet")}
              </Text>
              <Text style={[styles.emptyText, customText, { color: colors.subText }]}>
                {t("noReportsSubtitle")}
              </Text>
            </View>
          ) : (
            <>
              {/*
                ── منطق الترقيم ──
                moles مرتبة: index 0 = الأقدم = Report #1
                displayMoles = معكوسة للعرض (الأحدث فوق)
                displayIndex 0 = الأحدث → reportNumber = moles.length
                displayIndex last = الأقدم → reportNumber = 1
              */}
              {displayMoles.map((mole, displayIndex) => {
                const reportNumber = moles.length - displayIndex;
                const webScan     = isWeb(mole.source);

                return (
                  <View key={mole.id} style={[styles.reportCard, { backgroundColor: colors.card }]}>

                    {/* ── صورة الـ report ── */}
                    <TouchableOpacity
                      style={styles.imageContainer}
                      onPress={() => router.push({
                        pathname: "/Screensbar/Reportdetails",
                        params: {
                          moleId:      mole.id,
                          photoUri:    mole.photoUri,
                          timestamp:   mole.timestamp.toString(),
                          bodyView:    mole.bodyView,
                          x:           mole.x.toString(),
                          y:           mole.y.toString(),
                          analysis:    mole.analysis || "",
                          reportIndex: (reportNumber - 1).toString(),
                        },
                      })}
                      activeOpacity={0.9}
                    >
                      <Image
                        source={{ uri: mole.photoUri }}
                        style={styles.reportImage}
                        resizeMode="cover"
                      />

                      {/* ── Badge يمين: Front / Back — أزرق داكن ── */}
                      <View style={styles.imageBadgeRight}>
                        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },,styles.imageBadgeText]}>
                          {mole.bodyView === "front"
                            ? t("frontBody")
                            : mole.bodyView === "back"
                            ? t("backBody")
                            : "N/A"}
                        </Text>
                      </View>

                      {/* ── Badge شمال: icon الجهاز ── */}
                      <View style={[
                        styles.imageBadgeLeft,
                        {
                          backgroundColor: webScan
                            ? 'rgba(0,163,163,0.88)'
                            : 'rgba(0,79,127,0.88)',
                        },
                      ]}>
                        <Image
                          source={webScan ? Icons.monitor : Icons.smartphone}
                          style={styles.platformIcon}
                          resizeMode="contain"
                        />
                      </View>

                      {/* ── زرار التكبير ── */}
                      <View style={styles.expandIcon}>
                        <Ionicons name="expand-outline" size={20} color="#FFFFFF" />
                      </View>
                    </TouchableOpacity>

                    {/* ── محتوى الكارد ── */}
                    <View style={styles.reportContent}>
                      <View style={[
                        styles.reportHeader,
                        { flexDirection: isArabic ? "row-reverse" : "row" },
                      ]}>
                        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },styles.reportTitle, customText, { color: '#00E5FF' }]}>
                          {t("reportNum")}{reportNumber}
                        </Text>
                        <Text style={[
                          styles.reportDate,
                          customText,
                          { color: colors.subText, fontSize: Math.max(11, settings.fontSize - 3) },
                        ]}>
                          {formatDate(mole.timestamp)}
                        </Text>
                      </View>

                      <Text style={[
                        { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                        styles.reportText,
                        customText,
                        { color: colors.subText, textAlign: isArabic ? "right" : "left" },
                      ]}>
                        {mole.analysis || t("analysisInProgress")}
                      </Text>

                      <TouchableOpacity
                        style={[styles.downloadButton, {
                          backgroundColor: isDark ? "#004F7F" : "#E8F4F8",
                          borderColor:     isDark ? "#374151" : "#C5E3ED",
                          flexDirection:   isArabic ? "row-reverse" : "row",
                          alignSelf:       isArabic ? "flex-start" : "flex-end",
                          opacity:         downloadingId || downloadingAll ? 0.5 : 1,
                        }]}
                        onPress={() => downloadSingleReport(mole, reportNumber)}
                        activeOpacity={0.8}
                        disabled={!!downloadingId || downloadingAll}
                      >
                        {downloadingId === mole.id ? (
                          <ActivityIndicator size="small" color={colors.primary} />
                        ) : (
                          <>
                            <Ionicons
                              name="download-outline"
                              size={18}
                              color={isDark ? "#E8F4F8" : "#374151"}
                            />
                            <Text style={[
                              { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                              styles.downloadButtonText,
                              { color: isDark ? "#E8F4F8" : "#374151" },
                            ]}>
                              {t("downloadPDF")}
                            </Text>
                          </>
                        )}
                      </TouchableOpacity>
                    </View>
                  </View>
                );
              })}

              {/* ── Download All Button ── */}
              <TouchableOpacity
                style={[,styles.downloadAllButton, {
                  backgroundColor: colors.primary,
                  flexDirection:   isArabic ? "row-reverse" : "row",
                }]}
                onPress={downloadAllReports}
                disabled={downloadingAll}
                activeOpacity={0.8}
              >
                {downloadingAll ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <>
                    <Ionicons name="cloud-download-outline" size={22} color="#FFFFFF" />
                    <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.downloadAllText]}>{t("downloadAll")}</Text>
                  </>
                )}
              </TouchableOpacity>
            </>
          )}
        </ScrollView>

        {/* ── Bottom Nav ── */}
        <View style={[styles.bottomNavContainer]}>
          <View style={[styles.bottomNav, { backgroundColor: colors.navBg, borderTopColor: colors.border }]}>
            {["Home", "Reports"].map((tabName) => {
              const tab      = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity
                  key={tab.name}
                  style={styles.navItem}
                  onPress={() => handleTabPress(tab.name)}
                >
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                    isActive && {
                      backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                      borderWidth: 2,
                      borderColor: isDark ? "#00A3A3" : "#C5E3ED",
                    },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                    styles.navText,
                    { color: isActive ? colors.navActive : colors.navText },
                    isActive && { fontWeight: "700" },
                  ]}>
                    {tabLabels[tabName]}
                  </Text>
                </TouchableOpacity>
              );
            })}
            <View style={styles.navCenterSpacer} />
            {["History", "Settings"].map((tabName) => {
              const tab      = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity
                  key={tab.name}
                  style={styles.navItem}
                  onPress={() => handleTabPress(tab.name)}
                >
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                    isActive && {
                      backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                      borderWidth: 2,
                      borderColor: isDark ? "#00A3A3" : "#C5E3ED",
                    },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[
                    { fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                    styles.navText,
                    { color: isActive ? colors.navActive : colors.navText },
                    isActive && { fontWeight: "700" },
                  ]}>
                    {tabLabels[tabName]}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          <TouchableOpacity
            style={[
              styles.cameraButton,
              { backgroundColor: colors.navBg, borderColor: isDark ? "#374151" : "#C5E3ED" },
              activeTab === "Camera" && {
                borderColor:     colors.navActive,
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
    headerTitle:        { fontSize: 22 },
    scrollView:         { flex: 1 },
    scrollContent:      { padding: 16, paddingBottom: 100 },
    loadingContainer:   { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 100 },
    loadingText:        { marginTop: 12, fontSize: 16 },
    emptyContainer:     { alignItems: 'center', justifyContent: 'center', paddingVertical: 80 },
    emptyIcon:          { width: 90, height: 90 },
    emptyTitle:         { fontSize: 20, marginTop: 16 },
    emptyText:          { fontSize: 14, marginTop: 8, textAlign: 'center', paddingHorizontal: 40 },
    reportCard:         { borderRadius: 16, marginBottom: 16, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
    imageContainer:     { position: 'relative', width: '100%', height: 200 },
    reportImage:        { width: '100%', height: '100%' },
    reportNumberBadge:     { position: 'absolute', bottom: 12, left: 12, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
    reportNumberBadgeText: { color: '#FFFFFF', fontSize: 14 },
    imageBadgeRight:    { position: 'absolute', top: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
    imageBadgeText:     { color: '#FFFFFF', fontSize: 12 },
    imageBadgeLeft:     { position: 'absolute', top: 12, left: 12, width: 38, height: 38, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
    platformIcon:       { width: 22, height: 22, tintColor: '#FFFFFF' },
    expandIcon:         { position: 'absolute', bottom: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.8)', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
    reportContent:      { padding: 16 },
    reportHeader:       { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
    reportTitle:        { fontSize: 18},
    reportDate:         { fontSize: 12 },
    reportText:         { fontSize: 14, lineHeight: 20, marginBottom: 16 },
    downloadButton:     { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 10, paddingHorizontal: 16, borderRadius: 12, borderWidth: 1 },
    downloadButtonText: { fontSize: 14, marginLeft: 6 },
    downloadAllButton:  { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, borderRadius: 16, marginTop: 8, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 4, marginBottom: 25 },
    downloadAllText:    { fontSize: 16, color: '#FFFFFF', marginLeft: 8 },
    bottomNavContainer: { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
    bottomNav:          { flexDirection: 'row', paddingVertical: 10, borderTopWidth: 1, width: '100%', paddingBottom: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20 },
    navCenterSpacer:    { flex: 1 },
    navItem:            { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:            { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:         { width: 44, height: 44 },
    navText:            { fontSize: 11 },
    cameraButton:       { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 6 },
});