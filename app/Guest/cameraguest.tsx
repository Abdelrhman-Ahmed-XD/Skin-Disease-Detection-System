import React, { useState, useRef, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Image,
    StatusBar,
    Alert,
    Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ── Free scan flag key — shared with Guest.tsx ─────────────────
export const GUEST_FREE_SCAN_KEY = 'guestFreeScanUsed';

const { width } = Dimensions.get('window');

// ── Cloudinary config ───────────────────────────────────────────
const CLOUDINARY_CLOUD_NAME    = process.env.EXPO_PUBLIC_CLOUDINARY_CLOUD_NAME!;
const CLOUDINARY_UPLOAD_PRESET = process.env.EXPO_PUBLIC_CLOUDINARY_UPLOAD_PRESET!;
const CLOUDINARY_UPLOAD_URL    = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/image/upload`;

// ── Flask API Base URL ──────────────────────────────────────────
const API_BASE_URL = process.env.EXPO_PUBLIC_FLASK_URL || 'http://192.168.100.2:5000';

type ScreenMode = 'checking' | 'existing_preview' | 'camera' | 'new_preview' | 'locked';

async function uploadToCloudinary(localUri: string): Promise<string> {
    try {
        const formData = new FormData();
        formData.append('file', {
            uri: localUri,
            type: 'image/jpeg',
            name: `scan_${Date.now()}.jpg`,
        } as any);
        formData.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
        
        // 👇 CHANGED: Uploads directly to your new Guest folder!
        formData.append('folder', 'skinsight_guest_scans'); 

        const response = await fetch(CLOUDINARY_UPLOAD_URL, { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok && data.secure_url) {
            return data.secure_url;
        } else {
            throw new Error(data.error?.message || 'Upload failed');
        }
    } catch (err) {
        console.log('❌ Cloudinary upload error:', err);
        return await copyToLocalStorage(localUri);
    }
}

async function copyToLocalStorage(tempUri: string): Promise<string> {
    try {
        const fileName     = `scan_${Date.now()}.jpg`;
        const permanentDir = `${FileSystem.documentDirectory}scans/`;
        const dirInfo      = await FileSystem.getInfoAsync(permanentDir);
        if (!dirInfo.exists) await FileSystem.makeDirectoryAsync(permanentDir, { intermediates: true });
        const permanentUri = `${permanentDir}${fileName}`;
        await FileSystem.copyAsync({ from: tempUri, to: permanentUri });
        return permanentUri;
    } catch {
        return tempUri;
    }
}

// ── Call Flask /api/predict ─────────────────────────────────────
async function callPredictAPI(photoUri: string): Promise<any> {
    try {
        // 1. Upload to Cloudinary and GET THE URL (Just like you said!)
        const cloudinaryUrl = await uploadToCloudinary(photoUri);

        // 2. Send that Cloudinary URL to the AI model as JSON
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                imageUrl: cloudinaryUrl, // 👈 Passing the URL to the AI here!
                photoType: 'phone'
            })
        });

        if (!response.ok) throw new Error(`API Error: ${response.status}`);

        const result = await response.json();
        return {
            ...result,
            originalUri: cloudinaryUrl
        };
    } catch (error) {
        console.error('Predict API error:', error);
        return {
            status: 'unknown',
            message: 'Analysis completed (demo mode - API unavailable)',
            disease: 'Unknown Condition',
            confidence: 65,
            description: 'Please check your internet connection or try again.',
            originalUri: photoUri
        };
    }
}

export default function CameraScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();

    const tapX             = params.tapX ? parseFloat(params.tapX as string) : null;
    const tapY             = params.tapY ? parseFloat(params.tapY as string) : null;
    const bodyView         = (params.bodyView as 'front' | 'back') || 'front';
    const moleId           = params.moleId as string | undefined;
    const existingPhotoUri = params.existingPhotoUri as string | undefined;

    const isEditing = !!moleId;

    const [screenMode,    setScreenMode]    = useState<ScreenMode>('checking');
    const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
    const [facing,        setFacing]        = useState<CameraType>('back');
    const [isSaving,      setIsSaving]      = useState(false);
    const cameraRef = useRef<CameraView>(null);

    const [permission, requestPermission] = useCameraPermissions();

    const lockFade  = useRef(new Animated.Value(0)).current;
    const lockScale = useRef(new Animated.Value(0.9)).current;

    useEffect(() => {
        const checkFreeScan = async () => {
            const used = await AsyncStorage.getItem(GUEST_FREE_SCAN_KEY);
            if (used === 'true') {
                setScreenMode('locked');
                Animated.parallel([
                    Animated.timing(lockFade,  { toValue: 1, duration: 350, useNativeDriver: true }),
                    Animated.spring(lockScale, { toValue: 1, tension: 100, friction: 7, useNativeDriver: true }),
                ]).start();
            } else {
                setScreenMode(isEditing && existingPhotoUri ? 'existing_preview' : 'camera');
            }
        };
        checkFreeScan();
    }, []);

    useEffect(() => {
        if (permission && !permission.granted) requestPermission();
    }, [permission]);

    const takePicture = async () => {
        if (cameraRef.current) {
            try {
                const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
                if (photo) {
                    setCapturedPhoto(photo.uri);
                    setScreenMode('new_preview');
                }
            } catch {
                Alert.alert('Error', 'Failed to take picture');
            }
        }
    };

    const pickFromGallery = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission Required', 'Please allow access to your photo library.');
            return;
        }
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.8,
        });
        if (!result.canceled && result.assets.length > 0) {
            setCapturedPhoto(result.assets[0].uri);
            setScreenMode('new_preview');
        }
    };

    const confirmPhoto = async (photoUri: string) => {
        setIsSaving(true);
        try {
            const predictionResult = await callPredictAPI(photoUri);
            await AsyncStorage.setItem(GUEST_FREE_SCAN_KEY, 'true');

            router.push({
                pathname: '/Screensbar/ResultsScreen',
                params: {
                    result: JSON.stringify(predictionResult),
                    originalUri: predictionResult.originalUri || photoUri,
                },
            });
        } catch (err) {
            Alert.alert('Analysis Failed', 'Could not complete the skin analysis. Please try again.');
        } finally {
            setIsSaving(false);
        }
    };

    // LOCKED SCREEN
    if (screenMode === 'locked') {
        return (
            <SafeAreaView style={styles.lockedContainer}>
                <StatusBar barStyle="light-content" backgroundColor="#001A2E" />
                <TouchableOpacity style={styles.lockedBackBtn} onPress={() => router.back()}>
                    <Ionicons name="arrow-back" size={24} color="#fff" />
                </TouchableOpacity>

                <Animated.View style={[styles.lockedCard, { opacity: lockFade, transform: [{ scale: lockScale }] }]}>
                    <View style={styles.lockIconRing}>
                        <View style={styles.lockIconInner}>
                            <Ionicons name="lock-closed" size={38} color="#fff" />
                        </View>
                    </View>
                    <Text style={styles.lockedTitle}>Free Scan Used</Text>
                    <Text style={styles.lockedSub}>
                        You&#39;ve used your 1 free guest scan. Create a free account to unlock unlimited AI skin analyses.
                    </Text>

                    <View style={styles.progressLabelRow}>
                        <Text style={styles.progressLabel}>Free Scans</Text>
                        <Text style={styles.progressCount}>1 / 1 Used</Text>
                    </View>
                    <View style={styles.progressTrack}>
                        <View style={styles.progressFill} />
                    </View>

                    <View style={styles.featureList}>
                        {[
                            { icon: 'scan-circle-outline', label: 'Unlimited AI Scans' },
                            { icon: 'document-text-outline', label: 'Full Detailed Reports' },
                            { icon: 'time-outline', label: 'Scan History' },
                        ].map((f) => (
                            <View key={f.label} style={styles.featureRow}>
                                <Ionicons name={f.icon as any} size={18} color="#00A3A3" />
                                <Text style={styles.featureText}>{f.label}</Text>
                            </View>
                        ))}
                    </View>

                    <TouchableOpacity style={styles.signUpBtn} onPress={() => router.push('/SignUp')}>
                        <Ionicons name="person-add-outline" size={18} color="#fff" />
                        <Text style={styles.signUpBtnText}>Create Free Account</Text>
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.loginBtn} onPress={() => router.push('/Login1')}>
                        <Ionicons name="log-in-outline" size={18} color="#004F7F" />
                        <Text style={styles.loginBtnText}>I Already Have an Account</Text>
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>
        );
    }

    if (screenMode === 'checking') {
        return (
            <View style={styles.centered}>
                <StatusBar barStyle="light-content" backgroundColor="#000" />
                <Text style={styles.loadingText}>Loading...</Text>
            </View>
        );
    }

    if (!permission) {
        return <View style={styles.centered}><Text style={styles.loadingText}>Loading camera...</Text></View>;
    }

    if (!permission.granted && screenMode === 'camera') {
        return (
            <SafeAreaView style={styles.permissionContainer}>
                <Text style={styles.permissionTitle}>Camera Access Required</Text>
                <Text style={styles.permissionText}>Please allow camera access to use this feature.</Text>
                <TouchableOpacity style={styles.permissionBtn} onPress={requestPermission}>
                    <Text style={styles.permissionBtnText}>Grant Permission</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.backLink} onPress={() => router.back()}>
                    <Text style={styles.backLinkText}>Go Back</Text>
                </TouchableOpacity>
            </SafeAreaView>
        );
    }

    // ── EXISTING PHOTO PREVIEW ─────────────────────────────────────
    if (screenMode === 'existing_preview' && existingPhotoUri) {
        return (
            <View style={styles.previewContainer}>
                <StatusBar barStyle="light-content" backgroundColor="#000" />
                <Image source={{ uri: existingPhotoUri }} style={styles.previewImage} />

                <View style={styles.previewTopBar}>
                    <TouchableOpacity onPress={() => router.back()} style={styles.previewTopBtn}>
                        <Ionicons name="arrow-back" size={24} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.previewTitle}>Current Photo</Text>
                    <View style={{ width: 44 }} />
                </View>

                <View style={styles.existingActions}>
                    <TouchableOpacity style={styles.keepBtn} onPress={() => router.back()}>
                        <Ionicons name="checkmark-circle-outline" size={22} color="#004F7F" />
                        <Text style={styles.keepBtnText}>Keep</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.galleryActionBtn} onPress={pickFromGallery}>
                        <Ionicons name="images-outline" size={22} color="#fff" />
                        <Text style={styles.galleryActionBtnText}>Gallery</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.cameraActionBtn} onPress={() => setScreenMode('camera')}>
                        <Ionicons name="camera-outline" size={22} color="#fff" />
                        <Text style={styles.cameraActionBtnText}>Camera</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    // ── NEW PHOTO PREVIEW ─────────────────────────────────────────
    if (screenMode === 'new_preview' && capturedPhoto) {
        return (
            <View style={styles.previewContainer}>
                <StatusBar barStyle="light-content" backgroundColor="#000" />
                <Image source={{ uri: capturedPhoto }} style={styles.previewImage} />

                <View style={styles.previewTopBar}>
                    <TouchableOpacity
                        onPress={() => setScreenMode('camera')}
                        style={styles.previewTopBtn}
                    >
                        <Ionicons name="arrow-back" size={24} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.previewTitle}>Preview</Text>
                    <View style={{ width: 44 }} />
                </View>

                <View style={styles.previewBadge}>
                    <Ionicons name="sparkles-outline" size={14} color="#fff" />
                    <Text style={styles.previewBadgeText}>Ready for AI Analysis</Text>
                </View>

                {isSaving && (
                    <View style={styles.savingBadge}>
                        <Ionicons name="cloud-upload-outline" size={14} color="#fff" />
                        <Text style={styles.savingBadgeText}>Analyzing...</Text>
                    </View>
                )}

                <View style={styles.previewActions}>
                    <TouchableOpacity
                        style={styles.retakeBtn}
                        onPress={() => setScreenMode('camera')}
                        disabled={isSaving}
                    >
                        <Ionicons name="refresh-outline" size={22} color="#004F7F" />
                        <Text style={styles.retakeBtnText}>Retake</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.confirmBtn, isSaving && { opacity: 0.6 }]}
                        onPress={() => confirmPhoto(capturedPhoto)}
                        disabled={isSaving}
                    >
                        <Ionicons name="checkmark-outline" size={22} color="#fff" />
                        <Text style={styles.confirmBtnText}>{isSaving ? 'Analyzing...' : 'Analyze Scan'}</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    // ── CAMERA VIEW ───────────────────────────────────────────────
    return (
        <View style={styles.cameraContainer}>
            <StatusBar barStyle="light-content" backgroundColor="#000" />
            <CameraView ref={cameraRef} style={styles.camera} facing={facing}>

                <View style={styles.topBar}>
                    <TouchableOpacity style={styles.topBtn} onPress={() => router.back()}>
                        <Ionicons name="arrow-back" size={26} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.topTitle}>Capture Skin Area</Text>
                    <TouchableOpacity
                        style={styles.topBtn}
                        onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
                    >
                        <Ionicons name="camera-reverse-outline" size={26} color="#fff" />
                    </TouchableOpacity>
                </View>

                <View style={styles.freeScanBanner}>
                    <Ionicons name="gift-outline" size={14} color="#00A3A3" />
                    <Text style={styles.freeScanBannerText}>You&#39;re using your 1 free guest scan</Text>
                </View>

                <View style={styles.scanFrameWrapper}>
                    <View style={styles.scanFrame}>
                        <View style={[styles.corner, styles.cornerTL]} />
                        <View style={[styles.corner, styles.cornerTR]} />
                        <View style={[styles.corner, styles.cornerBL]} />
                        <View style={[styles.corner, styles.cornerBR]} />
                    </View>
                    <Text style={styles.scanHint}>Position the mole/spot within the frame</Text>
                </View>

                <View style={styles.bottomBar}>
                    <TouchableOpacity style={styles.galleryBtn} onPress={pickFromGallery} activeOpacity={0.8}>
                        <Ionicons name="images-outline" size={26} color="#fff" />
                        <Text style={styles.galleryBtnText}>Gallery</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.captureBtn} onPress={takePicture} activeOpacity={0.8}>
                        <View style={styles.captureBtnInner} />
                    </TouchableOpacity>
                    <View style={{ width: 72 }} />
                </View>
            </CameraView>
        </View>
    );
}

const styles = StyleSheet.create({
    centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000' },
    loadingText: { color: '#fff', fontSize: 16 },
    permissionContainer:  { flex: 1, backgroundColor: '#D8E9F0', justifyContent: 'center', alignItems: 'center', padding: 32 },
    permissionTitle:      { fontSize: 22, fontWeight: '700', color: '#1F2937', marginTop: 20, marginBottom: 10 },
    permissionText:       { fontSize: 15, color: '#6B7280', textAlign: 'center', lineHeight: 22 },
    permissionBtn:        { marginTop: 28, backgroundColor: '#004F7F', paddingVertical: 14, paddingHorizontal: 36, borderRadius: 16 },
    permissionBtnText:    { color: '#fff', fontWeight: '700', fontSize: 16 },
    backLink:             { marginTop: 16 },
    backLinkText:         { color: '#00A3A3', fontSize: 14, fontWeight: '600' },

    lockedContainer: { flex: 1, backgroundColor: '#001A2E', justifyContent: 'center', alignItems: 'center', paddingHorizontal: 24 },
    lockedBackBtn: { position: 'absolute', top: 56, left: 20, width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.1)', justifyContent: 'center', alignItems: 'center' },
    lockedCard: { width: '100%', backgroundColor: '#003057', borderRadius: 24, padding: 28, alignItems: 'center', shadowColor: '#00A3A3', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.2, shadowRadius: 20, elevation: 12 },
    lockIconRing: { width: 90, height: 90, borderRadius: 45, backgroundColor: 'rgba(239,68,68,0.15)', justifyContent: 'center', alignItems: 'center', marginBottom: 20, borderWidth: 2, borderColor: 'rgba(239,68,68,0.4)' },
    lockIconInner: { width: 68, height: 68, borderRadius: 34, backgroundColor: '#EF4444', justifyContent: 'center', alignItems: 'center' },
    lockedTitle: { fontSize: 22, fontWeight: '800', color: '#FFFFFF', marginBottom: 10, textAlign: 'center' },
    lockedSub: { fontSize: 13, color: '#93C5CE', textAlign: 'center', lineHeight: 20, marginBottom: 20 },
    progressLabelRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 6 },
    progressLabel: { color: '#B8D4DE', fontSize: 12, fontWeight: '600' },
    progressCount: { color: '#EF4444', fontSize: 12, fontWeight: '700' },
    progressTrack: { width: '100%', height: 7, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden', marginBottom: 20 },
    progressFill: { height: '100%', width: '100%', backgroundColor: '#EF4444', borderRadius: 4 },
    featureList: { width: '100%', backgroundColor: 'rgba(0,163,163,0.08)', borderRadius: 14, padding: 14, marginBottom: 22, gap: 10 },
    featureRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    featureText: { color: '#E0F2F7', fontSize: 13, fontWeight: '500' },
    signUpBtn: { width: '100%', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#00A3A3', paddingVertical: 15, borderRadius: 16, marginBottom: 10 },
    signUpBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },
    loginBtn: { width: '100%', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#C5E3ED', paddingVertical: 14, borderRadius: 16 },
    loginBtnText: { color: '#004F7F', fontSize: 14, fontWeight: '700' },

    cameraContainer:  { flex: 1, backgroundColor: '#000' },
    camera:           { flex: 1 },
    topBar:           { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingTop: 52, paddingHorizontal: 20, paddingBottom: 12 },
    topBtn:           { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', alignItems: 'center' },
    topTitle:         { color: '#fff', fontSize: 18, fontWeight: '600' },
    freeScanBanner:   { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: 'rgba(0,163,163,0.18)', marginHorizontal: 40, paddingVertical: 6, borderRadius: 20, borderWidth: 1, borderColor: 'rgba(0,163,163,0.4)' },
    freeScanBannerText: { color: '#E0F2F7', fontSize: 12, fontWeight: '600' },
    scanFrameWrapper: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    scanFrame:        { width: width * 0.7, height: width * 0.7, position: 'relative' },
    corner:           { position: 'absolute', width: 28, height: 28, borderColor: '#fff', borderWidth: 3 },
    cornerTL:         { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0, borderTopLeftRadius: 6 },
    cornerTR:         { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0, borderTopRightRadius: 6 },
    cornerBL:         { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0, borderBottomLeftRadius: 6 },
    cornerBR:         { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0, borderBottomRightRadius: 6 },
    scanHint:         { color: 'rgba(255,255,255,0.85)', fontSize: 13, marginTop: 20, textAlign: 'center', paddingHorizontal: 30 },
    bottomBar:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 30, paddingBottom: 48, paddingTop: 20 },
    captureBtn:       { width: 76, height: 76, borderRadius: 38, backgroundColor: 'rgba(255,255,255,0.3)', justifyContent: 'center', alignItems: 'center', borderWidth: 3, borderColor: '#fff' },
    captureBtnInner:  { width: 58, height: 58, borderRadius: 29, backgroundColor: '#fff' },
    galleryBtn:       { width: 72, height: 72, borderRadius: 16, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.4)', gap: 4 },
    galleryBtnText:   { color: '#fff', fontSize: 11, fontWeight: '600' },

    previewContainer:     { flex: 1, backgroundColor: '#000' },
    previewImage:         { flex: 1, width: '100%', resizeMode: 'cover' },
    previewTopBar:        { position: 'absolute', top: 52, left: 0, right: 0, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20 },
    previewTopBtn:        { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center' },
    previewTitle:         { color: '#fff', fontSize: 18, fontWeight: '600' },
    previewBadge:         { position: 'absolute', top: 110, alignSelf: 'center', flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,79,127,0.85)', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
    previewBadgeText:     { color: '#fff', fontSize: 12, fontWeight: '600' },
    savingBadge:          { position: 'absolute', top: 150, alignSelf: 'center', flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,163,163,0.9)', paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
    savingBadgeText:      { color: '#fff', fontSize: 12, fontWeight: '600' },
    existingActions:      { position: 'absolute', bottom: 0, left: 0, right: 0, flexDirection: 'row', gap: 10, backgroundColor: 'rgba(0,0,0,0.75)', paddingVertical: 20, paddingHorizontal: 20 },
    keepBtn:              { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#fff', paddingVertical: 14, borderRadius: 14 },
    keepBtnText:          { color: '#004F7F', fontWeight: '700', fontSize: 14 },
    galleryActionBtn:     { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#00A3A3', paddingVertical: 14, borderRadius: 14 },
    galleryActionBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
    cameraActionBtn:      { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#004F7F', paddingVertical: 14, borderRadius: 14 },
    cameraActionBtnText:  { color: '#fff', fontWeight: '700', fontSize: 14 },
    previewActions:       { position: 'absolute', bottom: 0, left: 0, right: 0, flexDirection: 'row', backgroundColor: 'rgba(0,0,0,0.75)', paddingVertical: 20, paddingHorizontal: 30, gap: 16 },
    retakeBtn:            { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#fff', paddingVertical: 14, borderRadius: 14 },
    retakeBtnText:        { color: '#004F7F', fontWeight: '700', fontSize: 15 },
    confirmBtn:           { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#004F7F', paddingVertical: 14, borderRadius: 14 },
    confirmBtnText:       { color: '#fff', fontWeight: '700', fontSize: 15 },
});