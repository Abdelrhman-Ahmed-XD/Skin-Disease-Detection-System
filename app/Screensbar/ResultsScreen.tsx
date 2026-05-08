import React from 'react';
import { View, Text, StyleSheet, Image, ScrollView, TouchableOpacity, StatusBar, Linking } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useTheme } from '../ThemeContext';
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';

export default function ResultsScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t } = useTranslation(settings.language);

    const result = params.result ? JSON.parse(params.result as string) : null;
    const originalUri = params.originalUri as string;

    const customText = {
        fontSize: settings.fontSize,
        fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
    };

    const pageBg = isDark ? colors.background : settings.backgroundColor;
    const textColor = isDark ? '#fff' : '#004F7F';

    if (!result) {
        return (
            <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
                <Text style={{ color: textColor }}>Error loading results.</Text>
                <TouchableOpacity onPress={() => router.back()}>
                    <Text style={{ color: '#00A3A3', marginTop: 20 }}>Go Back</Text>
                </TouchableOpacity>
            </SafeAreaView>
        );
    }

    const isRejected = result.status === 'no_lesion' || result.status === 'bad_photo';
    const isWarning = result.status === 'healthy' || result.status === 'unknown' || result.status === 'low_confidence';
    const isKnown = result.status === 'known';
    const maskUrl = result.segmented_url || result.segmentedUrl;

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={['top', 'bottom']}>
            <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

            {/* Header */}
            <View style={[styles.header, { borderBottomColor: colors.border, backgroundColor: colors.card }]}>
                <TouchableOpacity onPress={() => router.navigate('/Screensbar/FirstHomePage')} style={styles.backBtn}>
                    <Ionicons name="close" size={28} color={textColor} />
                </TouchableOpacity>
                <Text style={[{ fontFamily: customText.fontFamily }, styles.headerTitle, { color: textColor }]}>
                    Scan Results
                </Text>
                <View style={{ width: 40 }} />
            </View>

            <ScrollView style={styles.scrollContent} contentContainerStyle={{ paddingBottom: 40 }} showsVerticalScrollIndicator={false}>

                {/* Images Section */}
                <View style={styles.imagePairContainer}>
                    <View style={styles.imageBox}>
                        <Text style={[styles.imageLabel, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>Original</Text>
                        <Image source={{ uri: originalUri }} style={[styles.scanImage, { backgroundColor: colors.card }]} />
                    </View>
                    {maskUrl && (
                        <View style={styles.imageBox}>
                            <Text style={[styles.imageLabel, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>U-Net Mask</Text>
                            <Image source={{ uri: maskUrl }} style={[styles.scanImage, { borderColor: '#00A3A3', borderWidth: 2, backgroundColor: '#000' }]} />
                        </View>
                    )}
                </View>

                {/* Status Badges */}
                {isRejected && (
                    <View style={[styles.card, { backgroundColor: isDark ? '#3F1A1A' : '#FEF2F2', borderColor: '#EF4444' }]}>
                        <Ionicons name="alert-circle" size={24} color="#EF4444" />
                        <Text style={[{ fontFamily: customText.fontFamily }, styles.diseaseTitle, { color: '#EF4444' }]}>Analysis Rejected</Text>
                        <Text style={[{ fontFamily: customText.fontFamily }, styles.descriptionText, { color: isDark ? '#D1D5DB' : '#7F1D1D' }]}>{result.message}</Text>
                    </View>
                )}

                {isWarning && (
                    <View style={[styles.card, { backgroundColor: isDark ? '#3A2E15' : '#FFFBEB', borderColor: '#F59E0B' }]}>
                        <Ionicons name="warning" size={24} color="#F59E0B" />
                        <Text style={[{ fontFamily: customText.fontFamily }, styles.diseaseTitle, { color: '#F59E0B' }]}>
                            {result.status === 'healthy' ? 'Healthy Skin' : 'Uncertain Result'}
                        </Text>
                        <Text style={[{ fontFamily: customText.fontFamily }, styles.descriptionText, { color: isDark ? '#D1D5DB' : '#92400E' }]}>{result.message || result.description}</Text>
                        {result.status === 'low_confidence' && (
                            <Text style={[{ fontFamily: customText.fontFamily }, { color: isDark ? '#fff' : '#004F7F', marginTop: 10, fontWeight: 'bold' }]}>Possible: {result.disease}</Text>
                        )}
                    </View>
                )}

                {/* Known Disease Info */}
                {isKnown && (
                    <>
                        <View style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                            <Text style={[styles.sectionTitle, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>Detected Condition</Text>
                            <Text style={[{ fontFamily: customText.fontFamily }, styles.diseaseTitle, { color: textColor }]}>{result.disease}</Text>

                            <View style={styles.confidenceRow}>
                                <Text style={[{ fontFamily: customText.fontFamily }, styles.confLabel, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>AI Confidence</Text>
                                <Text style={[{ fontFamily: customText.fontFamily }, styles.confValue, { color: textColor }]}>{result.confidence}%</Text>
                            </View>
                            <View style={[styles.confBarBg, { backgroundColor: isDark ? '#374151' : '#E5E7EB' }]}>
                                <View style={[styles.confBarFill, { width: `${result.confidence}%`, backgroundColor: result.confidence >= 80 ? '#22C55E' : result.confidence >= 60 ? '#F59E0B' : '#EF4444' }]} />
                            </View>
                        </View>

                        {result.description && (
                            <View style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                                <Text style={[styles.sectionTitle, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>About this condition</Text>
                                <Text style={[{ fontFamily: customText.fontFamily }, styles.descriptionText, { color: isDark ? '#D1D5DB' : '#374151' }]}>{result.description}</Text>
                            </View>
                        )}

                        {result.tips && result.tips.length > 0 && (
                            <View style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                                <Text style={[styles.sectionTitle, { color: '#00A3A3' }]}>Tips & Recommendations</Text>
                                {result.tips.map((tip: string, idx: number) => (
                                    <View key={idx} style={styles.bulletRow}>
                                        <Ionicons name="checkmark-circle" size={18} color="#00A3A3" style={{ marginTop: 2 }} />
                                        <Text style={[{ fontFamily: customText.fontFamily }, styles.bulletText, { color: isDark ? '#D1D5DB' : '#374151' }]}>{tip}</Text>
                                    </View>
                                ))}
                            </View>
                        )}

                        {result.precautions && result.precautions.length > 0 && (
                            <View style={[styles.card, { backgroundColor: isDark ? '#2a1111' : '#FEF2F2', borderColor: '#EF444455', borderWidth: 1 }]}>
                                <Text style={[styles.sectionTitle, { color: '#EF4444' }]}>When to see a doctor</Text>
                                {result.precautions.map((pre: string, idx: number) => (
                                    <View key={idx} style={styles.bulletRow}>
                                        <Ionicons name="medical" size={16} color="#EF4444" style={{ marginTop: 3 }} />
                                        <Text style={[{ fontFamily: customText.fontFamily }, styles.bulletText, { color: isDark ? '#FECACA' : '#991B1B' }]}>{pre}</Text>
                                    </View>
                                ))}
                            </View>
                        )}

                        {/* ── CLICKABLE MEDICAL REFERENCES (SMART PARSING) ── */}
                        {result?.sources && result.sources.length > 0 && (
                            <View style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                                <Text style={[styles.sectionTitle, { color: '#6B7280' }]}>Medical References</Text>

                                {result.sources.map((source: string, idx: number) => {
                                    // Safer Regex to catch the URL perfectly
                                    const match = source.match(/(.*?)\s*\((https?:\/\/[^)]+)\)/);

                                    let displayName = source;
                                    let linkUrl = '';

                                    if (match) {
                                        // Trim removes any accidental invisible spaces!
                                        displayName = match[1].trim();
                                        linkUrl = match[2].trim();
                                    } else if (source.startsWith('http')) {
                                        displayName = "View Medical Reference";
                                        linkUrl = source.trim();
                                    } else {
                                        displayName = source;
                                        linkUrl = `https://www.google.com/search?q=${encodeURIComponent(source + ' skin condition')}`;
                                    }

                                    return (
                                        <TouchableOpacity
                                            key={idx}
                                            style={styles.bulletRow}
                                            activeOpacity={0.7}
                                            onPress={() => Linking.openURL(linkUrl)}
                                        >
                                            <Ionicons name="link-outline" size={18} color="#00A3A3" style={{ marginTop: 2 }} />
                                            <Text style={[{ fontFamily: customText.fontFamily, fontStyle: 'italic', textDecorationLine: 'underline' }, styles.bulletText, { color: '#00A3A3' }]}>
                                                {displayName}
                                            </Text>
                                        </TouchableOpacity>
                                    );
                                })}
                            </View>
                        )}
                    </>
                )}

                {/* ── BOTTOM BUTTONS (Appears at the end of the scroll) ── */}
                <View style={{ flexDirection: 'row', marginTop: 10, marginBottom: 20 }}>
                    <TouchableOpacity
                        style={[styles.actionButton, { backgroundColor: isDark ? '#004F7F' : '#E8F4F8', marginRight: 6 }]}
                        onPress={() => router.push('/Screensbar/Reports')}
                    >
                        <Ionicons name="document-text-outline" size={20} color={isDark ? '#fff' : '#004F7F'} style={{ marginRight: 6 }} />
                        <Text style={[styles.actionButtonText, { color: isDark ? '#fff' : '#004F7F', fontFamily: customText.fontFamily }]}>Go to Reports</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[styles.actionButton, { backgroundColor: '#00A3A3', marginLeft: 6 }]}
                        onPress={() => router.replace('/Screensbar/Camera')}
                    >
                        <Ionicons name="camera-outline" size={20} color="#fff" style={{ marginRight: 6 }} />
                        <Text style={[styles.actionButtonText, { color: '#fff', fontFamily: customText.fontFamily }]}>Take Another Scan</Text>
                    </TouchableOpacity>
                </View>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1 },
    header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1 },
    backBtn: { padding: 4 },
    headerTitle: { fontSize: 18, fontWeight: '700' },
    scrollContent: { flex: 1, padding: 16 },
    imagePairContainer: { flexDirection: 'row', justifyContent: 'center', gap: 16, marginBottom: 20 },
    imageBox: { flex: 1, alignItems: 'center' },
    imageLabel: { fontSize: 12, marginBottom: 6, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
    scanImage: { width: '100%', aspectRatio: 1, borderRadius: 16, overflow: 'hidden' },
    card: { borderRadius: 16, padding: 20, marginBottom: 16, borderWidth: 1, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2 },
    sectionTitle: { fontSize: 12, fontWeight: '700', textTransform: 'uppercase', marginBottom: 8, letterSpacing: 0.5 },
    diseaseTitle: { fontSize: 24, fontWeight: '800', marginBottom: 16 },
    confidenceRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
    confLabel: { fontSize: 14, fontWeight: '600' },
    confValue: { fontSize: 18, fontWeight: '800' },
    confBarBg: { height: 8, borderRadius: 4, overflow: 'hidden' },
    confBarFill: { height: '100%', borderRadius: 4 },
    descriptionText: { fontSize: 15, lineHeight: 24 },
    bulletRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12 },
    bulletText: { fontSize: 15, lineHeight: 22, flex: 1 },
// ── BOTTOM BUTTON STYLES ──
    fixedBottomContainer: {
        flexDirection: 'row',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderTopWidth: 1,
    },
    actionButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 14,
        borderRadius: 12,
    },
    actionButtonText: {
        fontSize: 15,
        fontWeight: '700',
    },
});