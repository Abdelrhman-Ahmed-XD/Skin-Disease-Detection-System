import React, { useState, useRef, useEffect } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, Dimensions,
    Image, StatusBar, Alert, ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { scheduleScanNotification } from '../services/notificationService';
import { Ionicons } from '@expo/vector-icons';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { db, auth } from '../../Firebase/firebaseConfig';
import {
    collection, addDoc, updateDoc, doc, serverTimestamp,
} from 'firebase/firestore';

const MOLES_STORAGE_KEY = 'savedMoles';
const { width } = Dimensions.get('window');

const CLOUDINARY_CLOUD_NAME = process.env.EXPO_PUBLIC_CLOUDINARY_CLOUD_NAME!;
const CLOUDINARY_UPLOAD_PRESET = process.env.EXPO_PUBLIC_CLOUDINARY_UPLOAD_PRESET!;
const CLOUDINARY_UPLOAD_URL = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/image/upload`;

type MoleResult = {
    status?: string | null;
    disease?: string | null;
    confidence?: number | null;
    segmentedUrl?: string | null;
    description?: string | null;
    tips?: string[];
    precautions?: string[];
    message?: string | null;
};

type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back' | 'N/A' | string;
    firestoreId?: string;
    source?: string;
    result?: MoleResult | null;
};

type ScreenMode = 'existing_preview' | 'camera' | 'new_preview';

async function uploadToCloudinary(localUri: string): Promise<string> {
    try {
        const formData = new FormData();
        formData.append('file', {
            uri: localUri,
            type: 'image/jpeg',
            name: `scan_${Date.now()}.jpg`,
        } as any);
        formData.append('upload_preset', CLOUDINARY_UPLOAD_PRESET);
        formData.append('folder', 'skinsight_scans');

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
        const fileName = `scan_${Date.now()}.jpg`;
        const permanentDir = `${FileSystem.documentDirectory}scans/`;
        const dirInfo = await FileSystem.getInfoAsync(permanentDir);
        if (!dirInfo.exists) {
            await FileSystem.makeDirectoryAsync(permanentDir, { intermediates: true });
        }
        const permanentUri = `${permanentDir}${fileName}`;
        await FileSystem.copyAsync({ from: tempUri, to: permanentUri });
        return permanentUri;
    } catch {
        return tempUri;
    }
}

async function updateFirestoreScan(firestoreId: string, photoUri: string): Promise<void> {
    try {
        const user = auth.currentUser;
        if (!user) return;
        const scanRef = doc(db, 'users', user.uid, 'scans', firestoreId);
        await updateDoc(scanRef, { photoUri, updatedAt: serverTimestamp() });
    } catch (err) {}
}

export default function CameraScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();

    const tapX = params.tapX ? parseFloat(params.tapX as string) : null;
    const tapY = params.tapY ? parseFloat(params.tapY as string) : null;
    const bodyView = (params.bodyView as string) || 'front';
    const moleId = params.moleId as string | undefined;
    const existingPhotoUri = params.existingPhotoUri as string | undefined;
    const existingFirestoreId = params.firestoreId as string | undefined;

    const hasPosition = tapX !== null && tapY !== null;
    const isEditing = !!moleId;

    const [screenMode, setScreenMode] = useState<ScreenMode>(
        isEditing && existingPhotoUri ? 'existing_preview' : 'camera'
    );
    const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
    const [photoSource, setPhotoSource] = useState<'camera' | 'gallery'>('camera');
    const [facing, setFacing] = useState<CameraType>('back');
    const [isSaving, setIsSaving] = useState(false);
    const cameraRef = useRef<CameraView>(null);

    const [permission, requestPermission] = useCameraPermissions();

    useEffect(() => {
        if (permission && !permission.granted) requestPermission();
    }, [permission]);

    const takePicture = async () => {
        if (cameraRef.current) {
            try {
                const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
                if (photo) {
                    setCapturedPhoto(photo.uri);
                    setPhotoSource('camera');
                    setScreenMode('new_preview');
                }
            } catch (err) {
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
            allowsEditing: false,
        });
        if (!result.canceled && result.assets.length > 0) {
            setCapturedPhoto(result.assets[0].uri);
            setPhotoSource('gallery');
            setScreenMode('new_preview');
        }
    };

    const confirmPhoto = async (photoUri: string) => {
        setIsSaving(true);
        try {
            const permanentUri = await uploadToCloudinary(photoUri);

            let predictionData = null;
            try {
                const flaskUrl = process.env.EXPO_PUBLIC_FLASK_URL;
                const response = await fetch(`${flaskUrl}/api/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        imageUrl: permanentUri,
                        photoType: photoSource === 'camera' ? 'phone' : 'dermo'
                    })
                });
                predictionData = await response.json();
            } catch (e) {
                console.log("Prediction API Error:", e);
            }

            let firestoreId: string | null = null;
            const finalBodyView = hasPosition ? bodyView : 'N/A';

            const resultPayload = predictionData ? {
                status: predictionData.status ?? null,
                disease: predictionData.disease ?? null,
                confidence: predictionData.confidence ?? null,
                segmentedUrl: predictionData.segmented_url ?? predictionData.segmentedUrl ?? null,
                description: predictionData.description ?? null,
                tips: predictionData.tips || [],
                precautions: predictionData.precautions || [],
                sources: predictionData.sources || [], // 👈 THIS IS THE MISSING LINK!
                message: predictionData.message ?? null,
            } : null;

            if (isEditing && moleId && existingFirestoreId) {
                await updateFirestoreScan(existingFirestoreId, permanentUri);
                firestoreId = existingFirestoreId;
            } else {
                const user = auth.currentUser;
                if (user) {
                    const scansRef = collection(db, 'users', user.uid, 'scans');
                    const docRef = await addDoc(scansRef, {
                        photoUri: permanentUri,
                        bodyView: finalBodyView,
                        x: tapX ?? 0,
                        y: tapY ?? 0,
                        result: resultPayload,
                        createdAt: serverTimestamp(),
                        source: 'mobile'
                    });
                    firestoreId = docRef.id;
                }
            }

            const existing = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
            const currentMoles: Mole[] = existing ? JSON.parse(existing) : [];

            if (isEditing && moleId) {
                const updated = currentMoles.map((m) =>
                    m.id === moleId ? { ...m, photoUri: permanentUri, firestoreId, result: resultPayload } : m
                );
                await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(updated));
            } else {
                const newMole: Mole = {
                    id: `mole_${Date.now()}`,
                    x: tapX ?? 0,
                    y: tapY ?? 0,
                    timestamp: Date.now(),
                    photoUri: permanentUri,
                    bodyView: finalBodyView,
                    firestoreId: firestoreId ?? undefined,
                    source: 'mobile',
                    result: resultPayload || undefined,
                };
                await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify([...currentMoles, newMole]));
            }

     if (predictionData) {

                // ── FIREBASE NOTIFICATION BLOCK ──
                if (auth.currentUser) {
                    try {
                        const notifRef = collection(db, 'users', auth.currentUser.uid, 'notifications');
                        await addDoc(notifRef, {
                            type: 'scan_result',
                            title: 'New Scan Analysis Ready',
                            message: `Your AI scan results for ${predictionData.disease || 'your recent photo'} are ready to view.`,
                            disease: predictionData.disease || 'Unknown',
                            confidence: predictionData.confidence || 0,
                            isRead: false,
                            timestamp: serverTimestamp(),
                            imageUri: permanentUri
                        });
                    } catch (e) {
                        console.error("Failed to save notification:", e);
                    }
                }
                // ─────────────────────────────────

                scheduleScanNotification(
                    predictionData.disease || 'Unknown condition',
                    predictionData.confidence || 0,
                ).catch(() => {});

                router.push({
                    pathname: '/Screensbar/ResultsScreen',
                    params: {
                        result: JSON.stringify(predictionData),
                        originalUri: photoUri,
                    }
                });
            } else {
                router.back();
            }

        } catch (err) {
            Alert.alert('Error', 'Failed to save photo');
            console.log('confirmPhoto error:', err);
        } finally {
            setIsSaving(false);
        }
    };

    if (!permission) {
        return (
            <View style={styles.centered}>
                <Text style={styles.loadingText}>Loading camera...</Text>
            </View>
        );
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

    if (screenMode === 'existing_preview' && existingPhotoUri) {
        return (
            <SafeAreaView style={styles.previewContainer} edges={['top', 'bottom']}>
                <StatusBar barStyle="light-content" backgroundColor="#111" />

                <View style={styles.previewTopBar}>
                    <TouchableOpacity onPress={() => router.back()} style={styles.previewTopBtn}>
                        <Ionicons name="arrow-back" size={24} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.previewTitle}>Current Photo</Text>
                    <View style={{ width: 44 }} />
                </View>

                <View style={styles.previewImageArea}>
                    <Image source={{ uri: existingPhotoUri }} style={styles.previewImage} />
                </View>

                <View style={styles.previewInfoPanel}>
                    <View style={styles.previewInfoRow}>
                        <View style={styles.previewInfoIconWrap}>
                            <Ionicons name="eye-outline" size={20} color="#00A3A3" />
                        </View>
                        <View style={{ flex: 1 }}>
                            <Text style={styles.previewInfoLabel}>Current saved photo</Text>
                            <Text style={styles.previewInfoSub}>Keep it, pick a new one from your gallery, or retake</Text>
                        </View>
                    </View>
                </View>

                <View style={styles.existingActions}>
                    <TouchableOpacity style={styles.keepBtn} onPress={() => router.back()} activeOpacity={0.8}>
                        <Ionicons name="checkmark-circle-outline" size={22} color="#004F7F" />
                        <Text style={styles.keepBtnText}>Keep</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.galleryActionBtn} onPress={pickFromGallery} activeOpacity={0.8}>
                        <Ionicons name="images-outline" size={22} color="#fff" />
                        <Text style={styles.galleryActionBtnText}>Gallery</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.cameraActionBtn} onPress={() => setScreenMode('camera')} activeOpacity={0.8}>
                        <Ionicons name="camera-outline" size={22} color="#fff" />
                        <Text style={styles.cameraActionBtnText}>Camera</Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    if (screenMode === 'new_preview' && capturedPhoto) {
        return (
            <SafeAreaView style={styles.previewContainer} edges={['top', 'bottom']}>
                <StatusBar barStyle="light-content" backgroundColor="#111" />

                <View style={styles.previewTopBar}>
                    <TouchableOpacity
                        onPress={() => setScreenMode(isEditing && existingPhotoUri ? 'existing_preview' : 'camera')}
                        style={styles.previewTopBtn}
                    >
                        <Ionicons name="arrow-back" size={24} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.previewTitle}>Preview</Text>
                    <View style={{ width: 44 }} />
                </View>

                <View style={styles.previewImageArea}>
                    <Image source={{ uri: capturedPhoto }} style={styles.previewImage} />
                </View>

                <View style={styles.previewInfoPanel}>
                    <View style={styles.previewInfoRow}>
                        <View style={styles.previewInfoIconWrap}>
                            <Ionicons
                                name={isSaving ? 'cloud-upload-outline' : isEditing ? 'pencil-outline' : 'sparkles-outline'}
                                size={20}
                                color="#00A3A3"
                            />
                        </View>
                        <View style={{ flex: 1 }}>
                            <Text style={styles.previewInfoLabel}>
                                {isSaving ? 'Analyzing your scan...' : 'Ready for AI Analysis'}
                            </Text>
                            <Text style={styles.previewInfoSub}>
                                {isEditing
                                    ? 'This will replace the existing photo'
                                    : hasPosition
                                        ? 'Will be saved on body map'
                                        : 'Will be saved to History'}
                            </Text>
                        </View>
                        {isSaving && <ActivityIndicator size="small" color="#00A3A3" />}
                    </View>
                </View>

                <View style={styles.previewActions}>
                    <TouchableOpacity
                        style={styles.retakeBtn}
                        onPress={() => setScreenMode(isEditing && existingPhotoUri ? 'existing_preview' : 'camera')}
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
                        <Text style={styles.confirmBtnText}>
                            {isSaving ? 'Analyzing...' : isEditing ? 'Update Photo' : 'Analyze Image'}
                        </Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    return (
        <View style={styles.cameraContainer}>
            <StatusBar barStyle="light-content" backgroundColor="#000" />

            <CameraView ref={cameraRef} style={StyleSheet.absoluteFillObject} facing={facing} />

            {/* ✅ FIX: UI Overlay is now placed ON TOP of CameraView instead of inside it */}
            <View style={StyleSheet.absoluteFillObject}>
                <View style={styles.topBar}>
                    <TouchableOpacity
                        style={styles.topBtn}
                        onPress={() => {
                            if (isEditing && existingPhotoUri) {
                                setScreenMode('existing_preview');
                            } else {
                                router.back();
                            }
                        }}
                    >
                        <Ionicons name="arrow-back" size={26} color="#fff" />
                    </TouchableOpacity>
                    <Text style={styles.topTitle}>
                        {isEditing ? 'Take New Photo' : 'Capture Skin Area'}
                    </Text>
                    <TouchableOpacity
                        style={styles.topBtn}
                        onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
                    >
                        <Ionicons name="camera-reverse-outline" size={26} color="#fff" />
                    </TouchableOpacity>
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
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000' },
    loadingText: { color: '#fff', fontSize: 16 },
    permissionContainer: { flex: 1, backgroundColor: '#D8E9F0', justifyContent: 'center', alignItems: 'center', padding: 32 },
    permissionTitle: { fontSize: 22, fontWeight: '700', color: '#1F2937', marginTop: 20, marginBottom: 10 },
    permissionText: { fontSize: 15, color: '#6B7280', textAlign: 'center', lineHeight: 22 },
    permissionBtn: { marginTop: 28, backgroundColor: '#004F7F', paddingVertical: 14, paddingHorizontal: 36, borderRadius: 16 },
    permissionBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },
    backLink: { marginTop: 16 },
    backLinkText: { color: '#00A3A3', fontSize: 14, fontWeight: '600' },
    cameraContainer: { flex: 1, backgroundColor: '#000' },
    topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingTop: 52, paddingHorizontal: 20, paddingBottom: 12 },
    topBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', alignItems: 'center' },
    topTitle: { color: '#fff', fontSize: 18, fontWeight: '600' },
    scanFrameWrapper: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    scanFrame: { width: width * 0.7, height: width * 0.7, position: 'relative' },
    corner: { position: 'absolute', width: 28, height: 28, borderColor: '#fff', borderWidth: 3 },
    cornerTL: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0, borderTopLeftRadius: 6 },
    cornerTR: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0, borderTopRightRadius: 6 },
    cornerBL: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0, borderBottomLeftRadius: 6 },
    cornerBR: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0, borderBottomRightRadius: 6 },
    scanHint: { color: 'rgba(255,255,255,0.85)', fontSize: 13, marginTop: 20, textAlign: 'center', paddingHorizontal: 30 },
    bottomBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 30, paddingBottom: 48, paddingTop: 20 },
    captureBtn: { width: 76, height: 76, borderRadius: 38, backgroundColor: 'rgba(255,255,255,0.3)', justifyContent: 'center', alignItems: 'center', borderWidth: 3, borderColor: '#fff' },
    captureBtnInner: { width: 58, height: 58, borderRadius: 29, backgroundColor: '#fff' },
    galleryBtn: { width: 72, height: 72, borderRadius: 16, backgroundColor: 'rgba(255,255,255,0.15)', justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.4)', gap: 4 },
    galleryBtnText: { color: '#fff', fontSize: 11, fontWeight: '600' },
    previewContainer:     { flex: 1, backgroundColor: '#111' },
    previewTopBar:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 14, backgroundColor: 'rgba(0,0,0,0.7)' },
    previewTopBtn:        { width: 44, height: 44, borderRadius: 22, backgroundColor: 'rgba(255,255,255,0.1)', justifyContent: 'center', alignItems: 'center' },
    previewTitle:         { color: '#fff', fontSize: 18, fontWeight: '600' },
    previewImageArea:     { flex: 1 },
    previewImage:         { flex: 1, width: '100%', resizeMode: 'contain' },
    previewInfoPanel:     { backgroundColor: 'rgba(0,0,0,0.85)', paddingHorizontal: 20, paddingVertical: 14 },
    previewInfoRow:       { flexDirection: 'row', alignItems: 'center', gap: 14 },
    previewInfoIconWrap:  { width: 42, height: 42, borderRadius: 21, backgroundColor: 'rgba(0,163,163,0.15)', justifyContent: 'center', alignItems: 'center', flexShrink: 0 },
    previewInfoLabel:     { color: '#fff', fontSize: 15, fontWeight: '700', marginBottom: 3 },
    previewInfoSub:       { color: 'rgba(255,255,255,0.55)', fontSize: 12, lineHeight: 18 },
    existingActions:      { flexDirection: 'row', gap: 10, backgroundColor: 'rgba(0,0,0,0.85)', paddingVertical: 16, paddingHorizontal: 20 },
    keepBtn:              { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#fff', paddingVertical: 14, borderRadius: 14 },
    keepBtnText:          { color: '#004F7F', fontWeight: '700', fontSize: 14 },
    galleryActionBtn:     { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#00A3A3', paddingVertical: 14, borderRadius: 14 },
    galleryActionBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
    cameraActionBtn:      { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: '#004F7F', paddingVertical: 14, borderRadius: 14 },
    cameraActionBtnText:  { color: '#fff', fontWeight: '700', fontSize: 14 },
    previewActions:       { flexDirection: 'row', backgroundColor: 'rgba(0,0,0,0.85)', paddingVertical: 16, paddingHorizontal: 20, gap: 16 },
    retakeBtn:            { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#fff', paddingVertical: 14, borderRadius: 14 },
    retakeBtnText:        { color: '#004F7F', fontWeight: '700', fontSize: 15 },
    confirmBtn:           { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#004F7F', paddingVertical: 14, borderRadius: 14 },
    confirmBtnText:       { color: '#fff', fontWeight: '700', fontSize: 15 },
});