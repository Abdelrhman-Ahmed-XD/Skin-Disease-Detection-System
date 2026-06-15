import React from 'react';
import { View, Text, StyleSheet, Image, ScrollView, TouchableOpacity, StatusBar, Linking } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';

export default function GuestResultsScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();

    const result = params.result ? JSON.parse(params.result as string) : null;
    const originalUri = params.originalUri as string;

    if (!result) {
        return (
            <SafeAreaView style={styles.container}>
                <Text style={{ color: '#004F7F', padding: 20 }}>Error loading results.</Text>
                <TouchableOpacity onPress={() => router.back()} style={{ paddingHorizontal: 20 }}>
                    <Text style={{ color: '#00A3A3' }}>Go Back</Text>
                </TouchableOpacity>
            </SafeAreaView>
        );
    }

    const isRejected = result.status === 'no_lesion' || result.status === 'bad_photo';
    const isWarning  = result.status === 'healthy' || result.status === 'unknown' || result.status === 'low_confidence';
    const isKnown    = result.status === 'known';
    const maskUrl    = result.segmented_url || result.segmentedUrl;

    return (
        <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
            <StatusBar barStyle="dark-content" backgroundColor="#D8E9F0" />

            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.replace('/Guest/Guest')} style={styles.backBtn}>
                    <Ionicons name="close" size={28} color="#004F7F" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Scan Results</Text>
                <View style={{ width: 40 }} />
            </View>

            <ScrollView style={styles.scrollContent} contentContainerStyle={{ paddingBottom: 40 }} showsVerticalScrollIndicator={false}>

                {/* Images */}
                <View style={styles.imagePairContainer}>
                    <View style={styles.imageBox}>
                        <Text style={styles.imageLabel}>Original</Text>
                        <Image source={{ uri: originalUri }} style={[styles.scanImage, { backgroundColor: '#fff' }]} />
                    </View>
                    {maskUrl && (
                        <View style={styles.imageBox}>
                            <Text style={styles.imageLabel}>U-Net Mask</Text>
                            <Image source={{ uri: maskUrl }} style={[styles.scanImage, { borderColor: '#00A3A3', borderWidth: 2, backgroundColor: '#000' }]} />
                        </View>
                    )}
                </View>

                {/* Rejected */}
                {isRejected && (
                    <View style={[styles.card, { backgroundColor: '#FEF2F2', borderColor: '#EF4444' }]}>
                        <Ionicons name="alert-circle" size={24} color="#EF4444" />
                        <Text style={[styles.diseaseTitle, { color: '#EF4444' }]}>Analysis Rejected</Text>
                        <Text style={[styles.descriptionText, { color: '#7F1D1D' }]}>{result.message}</Text>
                    </View>
                )}

                {/* Warning */}
                {isWarning && (
                    <View style={[styles.card, { backgroundColor: '#FFFBEB', borderColor: '#F59E0B' }]}>
                        <Ionicons name="warning" size={24} color="#F59E0B" />
                        <Text style={[styles.diseaseTitle, { color: '#F59E0B' }]}>
                            {result.status === 'healthy' ? 'Healthy Skin' : 'Uncertain Result'}
                        </Text>
                        <Text style={[styles.descriptionText, { color: '#92400E' }]}>{result.message || result.description}</Text>
                        {result.status === 'low_confidence' && (
                            <Text style={{ color: '#004F7F', marginTop: 10, fontWeight: 'bold' }}>Possible: {result.disease}</Text>
                        )}
                    </View>
                )}

                {/* Known disease */}
                {isKnown && (
                    <>
                        <View style={[styles.card, { backgroundColor: '#fff', borderColor: '#E5E7EB' }]}>
                            <Text style={styles.sectionTitle}>Detected Condition</Text>
                            <Text style={[styles.diseaseTitle, { color: '#004F7F' }]}>{result.disease}</Text>
                            <View style={styles.confidenceRow}>
                                <Text style={[styles.confLabel, { color: '#6B7280' }]}>AI Confidence</Text>
                                <Text style={[styles.confValue, { color: '#004F7F' }]}>{result.confidence}%</Text>
                            </View>
                            <View style={[styles.confBarBg, { backgroundColor: '#E5E7EB' }]}>
                                <View style={[styles.confBarFill, { width: `${result.confidence}%`, backgroundColor: result.confidence >= 80 ? '#22C55E' : result.confidence >= 60 ? '#F59E0B' : '#EF4444' }]} />
                            </View>
                        </View>

                        {result.description && (
                            <View style={[styles.card, { backgroundColor: '#fff', borderColor: '#E5E7EB' }]}>
                                <Text style={styles.sectionTitle}>About this condition</Text>
                                <Text style={[styles.descriptionText, { color: '#374151' }]}>{result.description}</Text>
                            </View>
                        )}

                        {result.tips && result.tips.length > 0 && (
                            <View style={[styles.card, { backgroundColor: '#fff', borderColor: '#E5E7EB' }]}>
                                <Text style={[styles.sectionTitle, { color: '#00A3A3' }]}>Tips & Recommendations</Text>
                                {result.tips.map((tip: string, idx: number) => (
                                    <View key={idx} style={styles.bulletRow}>
                                        <Ionicons name="checkmark-circle" size={18} color="#00A3A3" style={{ marginTop: 2 }} />
                                        <Text style={[styles.bulletText, { color: '#374151' }]}>{tip}</Text>
                                    </View>
                                ))}
                            </View>
                        )}

                        {result.precautions && result.precautions.length > 0 && (
                            <View style={[styles.card, { backgroundColor: '#FEF2F2', borderColor: '#EF444455', borderWidth: 1 }]}>
                                <Text style={[styles.sectionTitle, { color: '#EF4444' }]}>When to see a doctor</Text>
                                {result.precautions.map((pre: string, idx: number) => (
                                    <View key={idx} style={styles.bulletRow}>
                                        <Ionicons name="medical" size={16} color="#EF4444" style={{ marginTop: 3 }} />
                                        <Text style={[styles.bulletText, { color: '#991B1B' }]}>{pre}</Text>
                                    </View>
                                ))}
                            </View>
                        )}

                        {result?.sources && result.sources.length > 0 && (
                            <View style={[styles.card, { backgroundColor: '#fff', borderColor: '#E5E7EB' }]}>
                                <Text style={[styles.sectionTitle, { color: '#6B7280' }]}>Medical References</Text>
                                {result.sources.map((source: string, idx: number) => {
                                    const match = source.match(/(.*?)\s*\((https?:\/\/[^)]+)\)/);
                                    let displayName = source;
                                    let linkUrl = '';
                                    if (match) {
                                        displayName = match[1].trim();
                                        linkUrl = match[2].trim();
                                    } else if (source.startsWith('http')) {
                                        displayName = 'View Medical Reference';
                                        linkUrl = source.trim();
                                    } else {
                                        displayName = source;
                                        linkUrl = `https://www.google.com/search?q=${encodeURIComponent(source + ' skin condition')}`;
                                    }
                                    return (
                                        <TouchableOpacity key={idx} style={styles.bulletRow} activeOpacity={0.7} onPress={() => Linking.openURL(linkUrl)}>
                                            <Ionicons name="link-outline" size={18} color="#00A3A3" style={{ marginTop: 2 }} />
                                            <Text style={[{ fontStyle: 'italic', textDecorationLine: 'underline' }, styles.bulletText, { color: '#00A3A3' }]}>{displayName}</Text>
                                        </TouchableOpacity>
                                    );
                                })}
                            </View>
                        )}
                    </>
                )}

                {/* Upsell Card */}
                <View style={styles.upsellCard}>
                    <View style={styles.upsellIconRing}>
                        <Ionicons name="lock-closed" size={28} color="#fff" />
                    </View>
                    <Text style={styles.upsellTitle}>Want more scans?</Text>
                    <Text style={styles.upsellSub}>
                        Create a free account to unlock unlimited AI skin analyses, detailed scan history, and full PDF reports.
                    </Text>
                    <View style={styles.upsellFeatures}>
                        {[
                            { icon: 'scan-circle-outline',   label: 'Unlimited AI Scans' },
                            { icon: 'document-text-outline', label: 'Full Detailed Reports' },
                            { icon: 'time-outline',          label: 'Scan History & Tracking' },
                        ].map((f) => (
                            <View key={f.label} style={styles.upsellFeatureRow}>
                                <Ionicons name={f.icon as any} size={16} color="#00A3A3" />
                                <Text style={styles.upsellFeatureText}>{f.label}</Text>
                            </View>
                        ))}
                    </View>
                    <TouchableOpacity style={styles.signUpBtn} onPress={() => router.push('/SignUp')} activeOpacity={0.85}>
                        <Ionicons name="person-add-outline" size={18} color="#fff" />
                        <Text style={styles.signUpBtnText}>Create Free Account</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.loginBtn} onPress={() => router.push('/Login1')} activeOpacity={0.85}>
                        <Ionicons name="log-in-outline" size={18} color="#004F7F" />
                        <Text style={styles.loginBtnText}>I Already Have an Account</Text>
                    </TouchableOpacity>
                </View>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:            { flex: 1, backgroundColor: '#D8E9F0' },
    header:               { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#C5D9E3', backgroundColor: '#fff' },
    backBtn:              { padding: 4 },
    headerTitle:          { fontSize: 18, fontWeight: '700', color: '#004F7F' },
    scrollContent:        { flex: 1, padding: 16 },
    imagePairContainer:   { flexDirection: 'row', justifyContent: 'center', gap: 16, marginBottom: 20 },
    imageBox:             { flex: 1, alignItems: 'center' },
    imageLabel:           { fontSize: 12, marginBottom: 6, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5, color: '#6B7280' },
    scanImage:            { width: '100%', aspectRatio: 1, borderRadius: 16, overflow: 'hidden' },
    card:                 { borderRadius: 16, padding: 20, marginBottom: 16, borderWidth: 1, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, shadowRadius: 6, elevation: 2 },
    sectionTitle:         { fontSize: 12, fontWeight: '700', textTransform: 'uppercase', marginBottom: 8, letterSpacing: 0.5, color: '#6B7280' },
    diseaseTitle:         { fontSize: 24, fontWeight: '800', marginBottom: 16 },
    confidenceRow:        { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
    confLabel:            { fontSize: 14, fontWeight: '600' },
    confValue:            { fontSize: 18, fontWeight: '800' },
    confBarBg:            { height: 8, borderRadius: 4, overflow: 'hidden' },
    confBarFill:          { height: '100%', borderRadius: 4 },
    descriptionText:      { fontSize: 15, lineHeight: 24 },
    bulletRow:            { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12 },
    bulletText:           { fontSize: 15, lineHeight: 22, flex: 1 },
    upsellCard:           { backgroundColor: '#003057', borderRadius: 24, padding: 24, alignItems: 'center', marginTop: 8, marginBottom: 20, shadowColor: '#00A3A3', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.15, shadowRadius: 20, elevation: 10 },
    upsellIconRing:       { width: 64, height: 64, borderRadius: 32, backgroundColor: '#00A3A3', justifyContent: 'center', alignItems: 'center', marginBottom: 16 },
    upsellTitle:          { fontSize: 20, fontWeight: '800', color: '#FFFFFF', marginBottom: 8, textAlign: 'center' },
    upsellSub:            { fontSize: 13, color: '#93C5CE', textAlign: 'center', lineHeight: 20, marginBottom: 16 },
    upsellFeatures:       { width: '100%', backgroundColor: 'rgba(0,163,163,0.1)', borderRadius: 12, padding: 12, marginBottom: 20, gap: 8 },
    upsellFeatureRow:     { flexDirection: 'row', alignItems: 'center', gap: 10 },
    upsellFeatureText:    { color: '#E0F2F7', fontSize: 13, fontWeight: '500' },
    signUpBtn:            { width: '100%', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#00A3A3', paddingVertical: 15, borderRadius: 16, marginBottom: 10 },
    signUpBtnText:        { color: '#fff', fontSize: 15, fontWeight: '700' },
    loginBtn:             { width: '100%', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#C5E3ED', paddingVertical: 14, borderRadius: 16 },
    loginBtnText:         { color: '#004F7F', fontSize: 14, fontWeight: '700' },
});
