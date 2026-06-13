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

const Icons = {
  home:       require('../../assets/Icons/home.png'),
  reports:    require('../../assets/Icons/Reports.png'),
  history:    require('../../assets/Icons/history.png'),
  settings:   require('../../assets/Icons/setting.png'),
  smartphone: require('../../assets/Icons/smartphone.png'),
  monitor:    require('../../assets/Icons/monitor.png'),
};

const { width } = Dimensions.get('window');

export type MoleResult = {
    status?: string;
    disease?: string;
    confidence?: number;
    segmentedUrl?: string;
    segmented_url?: string;
    description?: string;
    tips?: string[];
    precautions?: string[];
    sources?: string[];
    message?: string;
};

type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back' | 'N/A' | string;
    analysis?: string;
    source?: string;
    description?: string;
    result?: MoleResult;
};

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
                    FileSystem.cacheDirectory + 'tmp_img_' + Date.now() + '.jpg'
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

const confColor = (c: number) =>
    c >= 80 ? '#22C55E' : c >= 60 ? '#F59E0B' : '#EF4444';

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
      <td style="width:33.3%;padding:2px 3px;vertical-align:top">
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
            ? `<img src="${params.imageBase64}" style="width:100%;max-height:150px;border-radius:10px;border:3px solid #C5E3ED;object-fit:cover;display:block"/>`
            : `<div style="width:100%;height:110px;border-radius:10px;border:2px dashed #C5E3ED;background:#F4FBFF;display:flex;align-items:center;justify-content:center;color:#9CA3AF;font-size:9px;font-family:system-ui,sans-serif">No Image</div>`}
        </td>
        ${params.maskBase64 ? `
        <td style="width:50%;padding:0 0 0 6px;vertical-align:top;text-align:center">
          <div style="font-size:9px;font-weight:700;color:#00E5FF;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.4px;margin-bottom:5px">U-Net Mask</div>
          <img src="${params.maskBase64}" style="width:100%;max-height:150px;border-radius:10px;border:3px solid #00E5FF;object-fit:contain;background:#000000;display:block"/>
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

const buildAllReportsHTML = (params: {
    rows: Array<{
        index:       number;
        date:        string;
        bodyView:    string;
        analysis:    string;
        imageBase64: string;
        maskBase64:  string;
        frontBody:   string;
        backBody:    string;
        platform:    string;
        confidence:  number;
    }>;
    patientName:   string;
    age:           string;
    gender:        string;
    hairColor:     string;
    eyeColor:      string;
    skinColor:     string;
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
tbody tr:nth-child(even){background:#F4FBFF}
tbody tr:nth-child(odd){background:#fff}
tbody tr{border-bottom:1px solid #E5F0F6}
tbody tr:last-child{border-bottom:none}
td{padding:8px 7px;vertical-align:middle}
.tnum{font-size:12px;font-weight:bold;color:#004F7F;text-align:center}
.timg img{width:50px;height:50px;border-radius:6px;border:2px solid #C5E3ED;object-fit:cover;display:block;margin:0 auto}
.timg-mask img{width:50px;height:50px;border-radius:6px;border:2px solid #00E5FF;object-fit:contain;background:#000;display:block;margin:0 auto}
.timg-ph{width:50px;height:50px;border-radius:6px;border:2px dashed #C5E3ED;display:flex;align-items:center;justify-content:center;color:#9CA3AF;font-size:8px;margin:0 auto;background:#F4FBFF;text-align:center;padding:3px;line-height:1.3}
.tdate{font-size:10px;color:#374151;white-space:nowrap}
.loc-badge{display:inline-block;background:#E8F4F8;color:#004F7F;border:1px solid #C5E3ED;border-radius:4px;padding:2px 6px;font-size:9px;font-weight:600}
.plat-badge{display:inline-block;border-radius:4px;padding:2px 6px;font-size:9px;font-weight:600}
.plat-app{background:#E8F4F8;color:#004F7F;border:1px solid #C5E3ED}
.plat-web{background:#E6F4EA;color:#1A6B35;border:1px solid #A8D5B5}
.tanal{font-size:9px;color:#374151;line-height:1.5;max-width:160px}
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
    <div class="mtitle">All Reports. Full History</div>
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
          <th style="width:28px">#</th>
          <th style="width:58px">Image</th>
          <th style="width:58px">U-Net Mask</th>
          <th style="width:72px">Date</th>
          <th style="width:62px">Location</th>
          <th style="width:62px">Platform</th>
          <th style="width:80px">Confidence</th>
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
          <td class="timg-mask">
            ${row.maskBase64
              ? `<img src="${row.maskBase64}" alt="mask"/>`
              : `<div class="timg-ph">No Mask</div>`}
          </td>
          <td class="tdate">${row.date}</td>
          <td><span class="loc-badge">${row.bodyView === 'front' ? row.frontBody : row.bodyView === 'back' ? row.backBody : 'N/A'}</span></td>
          <td><span class="plat-badge ${row.platform === 'Web' ? 'plat-web' : 'plat-app'}">${row.platform}</span></td>
          <td style="text-align:center;min-width:72px">
            ${row.confidence > 0 ? `
              <div style="font-size:11px;font-weight:bold;color:${confColor(row.confidence)};font-family:system-ui,sans-serif;margin-bottom:3px">${row.confidence}%</div>
              <div style="width:100%;height:5px;background:#E5E7EB;border-radius:3px;overflow:hidden">
                <div style="height:100%;width:${row.confidence}%;background:${confColor(row.confidence)};border-radius:3px"></div>
              </div>
            ` : '<span style="font-size:9px;color:#9CA3AF">N/A</span>'}
          </td>
          <td class="tanal">${row.analysis || 'Analysis in progress...'}</td>
        </tr>`).join('')}
      </tbody>
    </table>
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
        const loadPatientData = async () => {
            const keys = ['userProfile', 'profileData', 'signupDraft'];
            let d: any = null;
            for (const key of keys) {
                try {
                    const saved = await AsyncStorage.getItem(key);
                    if (saved) {
                        const parsed = JSON.parse(saved);
                        if (parsed && (parsed.firstName || parsed.name || parsed.hairColor)) {
                            d = parsed;
                            break;
                        }
                    }
                } catch (_) {}
            }
            if (!d) return;
            const firstName = d.firstName || d.name?.split(' ')[0] || '';
            const lastName  = d.lastName  || d.name?.split(' ').slice(1).join(' ') || '';
            setPatientName(`${firstName} ${lastName}`.trim() || 'N/A');
            setGender(d.gender ? d.gender.charAt(0).toUpperCase() + d.gender.slice(1) : 'N/A');
            setHairColor(d.hairColor || 'N/A');
            setEyeColor(d.eyeColor   || 'N/A');
            const skinMap: Record<string, string> = {
                '#F5E0D3': 'Very Light', '#EACAA7': 'Light',      '#D1A67A': 'Medium',
                '#B57D50': 'Tan',        '#A05C38': 'Brown',      '#8B4513': 'Dark Brown',
                '#7A3E11': 'Deep',       '#603311': 'Ebony',
            };
            setSkinColor(d.skinColor ? (skinMap[d.skinColor] || d.skinColor) : 'N/A');
            if (d.birthYear && d.birthMonth && d.birthDay) {
                const dob = new Date(d.birthYear, d.birthMonth - 1, d.birthDay);
                setAge(`${Math.floor((Date.now() - dob.getTime()) / (1000 * 60 * 60 * 24 * 365.25))} years`);
            } else if (d.age) {
                setAge(typeof d.age === 'number' ? `${d.age} years` : d.age);
            }
        };
        loadPatientData().catch(() => {});
    }, []);

    useFocusEffect(
        React.useCallback(() => {
            setActiveTab('Reports');
            loadMoles();
        }, [])
    );

    const loadMoles = async () => {
        try {
            const data = await loadAllScansFromFirestore();
            const filtered = data.filter((m: Mole) => m.photoUri && m.photoUri.trim() !== '');
            const sorted = filtered.sort((a: Mole, b: Mole) => a.timestamp - b.timestamp);
            setMoles(sorted);
        } catch (err) {
            console.log('Error loading moles:', err);
        } finally {
            setLoading(false);
        }
    };

    const downloadSingleReport = async (mole: Mole, reportNumber: number) => {
        if (downloadingId || downloadingAll) return;
        try {
            setDownloadingId(mole.id);
            if (!mole.photoUri) return;
            const maskUrl = mole.result?.segmentedUrl || mole.result?.segmented_url || '';
            const [imageBase64, maskBase64] = await Promise.all([
                getImageBase64(mole.photoUri),
                maskUrl ? getImageBase64(maskUrl) : Promise.resolve(''),
            ]);
            const html = buildReportHTML({
                reportIndex:        reportNumber - 1,
                date:               new Date(mole.timestamp).toLocaleDateString(
                                        isArabic ? 'ar-EG' : 'en-US',
                                        { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }
                                    ),
                bodyView:           mole.bodyView,
                moleId:             mole.id,
                analysis:           mole.result?.disease || mole.analysis || t('analysisInProgress'),
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
                platform:           getPlatform(mole.source),
                confidence:         mole.result?.confidence ?? 0,
                diseaseDescription: mole.result?.description || '',
                tips:               mole.result?.tips        || [],
                precautions:        mole.result?.precautions || [],
                sources:            mole.result?.sources     || [],
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
                imageBase64: string; maskBase64: string; frontBody: string;
                backBody: string; platform: string; confidence: number;
            }> = [];
            for (let i = 0; i < moles.length; i++) {
                const mole    = moles[i];
                const maskUrl = mole.result?.segmentedUrl || mole.result?.segmented_url || '';
                const [imageBase64, maskBase64] = await Promise.all([
                    getImageBase64(mole.photoUri || ''),
                    maskUrl ? getImageBase64(maskUrl) : Promise.resolve(''),
                ]);
                rows.push({
                    index:      i,
                    date:       new Date(mole.timestamp).toLocaleDateString(
                                    isArabic ? 'ar-EG' : 'en-US',
                                    { month: 'short', day: 'numeric', year: 'numeric' }
                                ),
                    bodyView:   mole.bodyView,
                    analysis:   mole.result?.disease || mole.analysis || t('analysisInProgress'),
                    imageBase64,
                    maskBase64,
                    frontBody:  t('frontBody'),
                    backBody:   t('backBody'),
                    platform:   getPlatform(mole.source),
                    confidence: mole.result?.confidence ?? 0,
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

    const displayMoles = [...moles].reverse();

    return (
      <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
        <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

        <View style={[styles.header, { backgroundColor: colors.card }]}>
          <TouchableOpacity
            style={[styles.backButton, { borderColor: colors.border }]}
            onPress={() => router.back()}
          >
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.headerTitle, customText, { color: isDark ? "#fff" : "#374151" }]}>
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
              {/* Summary stats bar */}
              <View style={[styles.statsBar, { backgroundColor: isDark ? '#0D2030' : '#004F7F' }]}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{moles.length}</Text>
                  <Text style={styles.statLabel}>{t('reportsTab') || 'Reports'}</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{moles.filter(m => !isWeb(m.source)).length}</Text>
                  <Text style={styles.statLabel}>App Scans</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{moles.filter(m => isWeb(m.source)).length}</Text>
                  <Text style={styles.statLabel}>Web Scans</Text>
                </View>
              </View>

              {displayMoles.map((mole, displayIndex) => {
                const reportNumber = moles.length - displayIndex;
                const webScan = isWeb(mole.source);
                const diseaseName = mole.result?.disease || mole.analysis || t("analysisInProgress");
                const confidence = mole.result?.confidence || 0;

                return (
                  <View key={mole.id} style={[styles.reportCard, { backgroundColor: colors.card, borderTopColor: '#00A3A3', borderTopWidth: 3 }]}>
                    <TouchableOpacity
                      style={styles.imageContainer}
                      onPress={() => router.push({
                        pathname: "/Screensbar/Reportdetails",
                        params: {
                          moleId:      mole.id,
                          photoUri:    mole.photoUri || "",
                          timestamp:   mole.timestamp.toString(),
                          bodyView:    mole.bodyView,
                          x:           mole.x.toString(),
                          y:           mole.y.toString(),
                          analysis:    diseaseName,
                          reportIndex: (reportNumber - 1).toString(),
                          result:      JSON.stringify(mole.result || {}),
                        },
                      })}
                      activeOpacity={0.9}
                    >
                      <Image
                        source={{ uri: mole.photoUri }}
                        style={styles.reportImage}
                        resizeMode="cover"
                      />
                      <View style={styles.imageBadgeRight}>
                        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.imageBadgeText]}>
                          {mole.bodyView === "front" ? t("frontBody") : mole.bodyView === "back" ? t("backBody") : "N/A"}
                        </Text>
                      </View>
                      <View style={[styles.imageBadgeLeft, { backgroundColor: webScan ? 'rgba(0,163,163,0.88)' : 'rgba(0,79,127,0.88)' }]}>
                        <Image source={webScan ? Icons.monitor : Icons.smartphone} style={styles.platformIcon} resizeMode="contain" />
                      </View>
                      <View style={styles.expandIcon}>
                        <Ionicons name="expand-outline" size={20} color="#FFFFFF" />
                      </View>
                    </TouchableOpacity>

                    <View style={styles.reportContent}>
                      <View style={[styles.reportHeader, { flexDirection: isArabic ? "row-reverse" : "row" }]}>
                        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.reportTitle, customText, { color: isDark ? "#00A3A3" : "#004F7F", fontWeight: 'bold' }]}>
                          {t("reportNum")} {reportNumber}
                        </Text>
                        <Text style={[styles.reportDate, customText, { color: colors.subText, fontSize: Math.max(11, settings.fontSize - 3) }]}>
                          {formatDate(mole.timestamp)}
                        </Text>
                      </View>
                      <View style={[styles.diseaseNameBox, { backgroundColor: isDark ? '#0D2030' : '#EBF5FB', borderColor: isDark ? '#1E3A4A' : '#C5E3ED' }]}>
                        <Ionicons name="scan-outline" size={14} color="#00A3A3" style={{ marginRight: 6 }} />
                        <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.diseaseNameText, { color: isDark ? '#00C8C8' : '#004F7F' }]} numberOfLines={2}>
                          {diseaseName}
                        </Text>
                      </View>
                      {confidence > 0 && (
                        <View style={{ marginVertical: 8 }}>
                          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 }}>
                            <Text style={{ fontSize: 12, color: colors.subText }}>Confidence</Text>
                            <Text style={{ fontSize: 12, fontWeight: '700', color: colors.text }}>{confidence}%</Text>
                          </View>
                          <View style={{ height: 6, backgroundColor: isDark ? '#374151' : '#E5E7EB', borderRadius: 3, overflow: 'hidden' }}>
                            <View style={{ height: '100%', width: `${confidence}%`, backgroundColor: confidence >= 80 ? '#22C55E' : confidence >= 60 ? '#F59E0B' : '#EF4444' }} />
                          </View>
                        </View>
                      )}
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
                            <Ionicons name="download-outline" size={18} color={isDark ? "#E8F4F8" : "#374151"} />
                            <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.downloadButtonText, { color: isDark ? "#E8F4F8" : "#374151" }]}>
                              {t("downloadPDF")}
                            </Text>
                          </>
                        )}
                      </TouchableOpacity>
                    </View>
                  </View>
                );
              })}

              <TouchableOpacity
                style={[styles.downloadAllButton, { backgroundColor: colors.primary, flexDirection: isArabic ? "row-reverse" : "row" }]}
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

        {/* ── Floating Bottom Nav ── */}
        <View style={styles.bottomNavContainer}>
          <View style={[styles.bottomNav, {
            backgroundColor: colors.navBg,
            borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
          }]}>
            {["Home", "Reports"].map((tabName) => {
              const tab      = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: colors.navBg },
                    isActive && { backgroundColor: isDark ? "#1E3A4A" : pageBg, borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#2A7DA0" },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.navText, { color: isActive ? colors.navActive : colors.navText }, isActive && { fontWeight: "700" }]}>
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
                <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: colors.navBg },
                    isActive && { backgroundColor: isDark ? "#1E3A4A" : pageBg, borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#2A7DA0" },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.navText, { color: isActive ? colors.navActive : colors.navText }, isActive && { fontWeight: "700" }]}>
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
              activeTab === "Camera" && { borderColor: colors.navActive, backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8" },
            ]}
            onPress={() => handleTabPress("Camera")}
            activeOpacity={0.85}
          >
            <Ionicons name="camera-outline" size={30} color={activeTab === "Camera" ? colors.navActive : colors.navText} />
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
    scrollContent:      { padding: 16, paddingBottom: 130 },
    loadingContainer:   { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 100 },
    loadingText:        { marginTop: 12, fontSize: 16 },
    emptyContainer:     { alignItems: 'center', justifyContent: 'center', paddingVertical: 80 },
    emptyIcon:          { width: 90, height: 90 },
    emptyTitle:         { fontSize: 20, marginTop: 16 },
    emptyText:          { fontSize: 14, marginTop: 8, textAlign: 'center', paddingHorizontal: 40 },
    statsBar:           { borderRadius: 14, padding: 16, marginBottom: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 6, elevation: 3 },
    statItem:           { alignItems: 'center', flex: 1 },
    statValue:          { fontSize: 22, fontWeight: '800', color: '#FFFFFF' },
    statLabel:          { fontSize: 10, color: '#C5E3ED', marginTop: 2, textTransform: 'uppercase', letterSpacing: 0.5 },
    statDivider:        { width: 1, height: 36, backgroundColor: 'rgba(255,255,255,0.2)' },
    reportCard:         { borderRadius: 16, marginBottom: 16, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
    diseaseNameBox:     { flexDirection: 'row', alignItems: 'center', borderRadius: 10, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 8, marginBottom: 12 },
    diseaseNameText:    { flex: 1, fontSize: 14, fontWeight: '700', lineHeight: 20 },
    imageContainer:     { position: 'relative', width: '100%', height: 200 },
    reportImage:        { width: '100%', height: '100%' },
    imageBadgeRight:    { position: 'absolute', top: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.9)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
    imageBadgeText:     { color: '#FFFFFF', fontSize: 12 },
    imageBadgeLeft:     { position: 'absolute', top: 12, left: 12, width: 30, height: 30, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
    platformIcon:       { width: 16, height: 16, tintColor: '#FFFFFF' },
    expandIcon:         { position: 'absolute', bottom: 12, right: 12, backgroundColor: 'rgba(0,79,127,0.8)', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
    reportContent:      { padding: 16 },
    reportHeader:       { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
    reportTitle:        { fontSize: 18 },
    reportDate:         { fontSize: 12 },
    reportText:         { fontSize: 14, lineHeight: 20, marginBottom: 16 },
    downloadButton:     { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 10, paddingHorizontal: 16, borderRadius: 12, borderWidth: 1 },
    downloadButtonText: { fontSize: 14, marginLeft: 6 },
    downloadAllButton:  { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 16, borderRadius: 16, marginTop: 8, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 4, marginBottom: 25 },
    downloadAllText:    { fontSize: 16, color: '#FFFFFF', marginLeft: 8 },
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
    navCenterSpacer:    { flex: 1 },
    navItem:            { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:            { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:         { width: 32, height: 32 },
    navText:            { fontSize: 11 },
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