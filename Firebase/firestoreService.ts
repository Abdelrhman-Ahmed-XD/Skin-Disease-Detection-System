// ====================== 1. firestoreService.ts ======================
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
    collection, deleteDoc, doc, getDocs,
    query, setDoc, orderBy
} from 'firebase/firestore';
import { db, auth } from '../Firebase/firebaseConfig';

const MOLES_STORAGE_KEY = 'savedMoles';

export type MoleResult = {
    status?: string;
    disease?: string;
    confidence?: number;
    segmentedUrl?: string;
    segmented_url?: string;
    description?: string;
    tips?: string[];
    precautions?: string[];
    message?: string;
    sources?: string[];
};

export type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back' | 'N/A' | string;
    firestoreId?: string;
    source?: string;
    result?: MoleResult;           // ← NEW NESTED STRUCTURE
    // Legacy flat fields (for backward compatibility)
    analysis?: string;
    confidence?: number;
    segmentedUrl?: string;
};

const fetchAllScansFromFirestore = async (userId: string): Promise<Mole[]> => {
    const scansRef = collection(db, 'users', userId, 'scans');
    const q = query(scansRef, orderBy('createdAt', 'desc'));
    const snapshot = await getDocs(q);
    if (snapshot.empty) return [];

    return snapshot.docs.map(docSnap => {
        const data = docSnap.data();

        // Normalize to new nested structure
        let result: MoleResult = data.result || {};

        // Legacy flat field migration
        if (!data.result && (data.analysis || data.status)) {
            result = {
                status: data.status,
                disease: data.analysis || data.disease,
                confidence: data.confidence,
                segmentedUrl: data.segmentedUrl || data.segmented_url,
                description: data.description,
                tips: data.tips || [],
                precautions: data.precautions || [],
                message: data.message,
            };
        }

        return {
            id: docSnap.id,
            x: data.x ?? 0,
            y: data.y ?? 0,
            timestamp: data.createdAt?.toMillis?.() ?? (typeof data.createdAt === 'string' ? new Date(data.createdAt).getTime() : Date.now()),
            photoUri: data.photoUri ?? '',
            bodyView: data.bodyView ?? 'N/A',
            firestoreId: docSnap.id,
            source: data.source ?? 'mobile',
            result: result,
            // Keep legacy for safety
            analysis: data.analysis,
            confidence: data.confidence,
            segmentedUrl: data.segmentedUrl,
        } as Mole;
    });
};

export const saveMoleToFirestore = async (mole: Mole): Promise<void> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) return;

        const moleRef = doc(db, 'users', userId, 'scans', mole.id);
        await setDoc(moleRef, {
            ...mole,
            userId,
            updatedAt: Date.now(),
            source: mole.source || 'mobile',
        }, { merge: true });
    } catch (error) {
        console.log('❌ Error saving mole to Firestore:', error);
    }
};

export const loadMolesFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) return loadMolesFromLocal();

        const allScans = await fetchAllScansFromFirestore(userId);
        if (allScans.length === 0) return loadMolesFromLocal();

        const moles = allScans.filter(m => m.source !== 'web' && m.bodyView !== 'N/A');
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(moles));
        return moles;
    } catch (error) {
        console.log('❌ Error loading from Firestore:', error);
        return loadMolesFromLocal();
    }
};

export const loadAllScansFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) return [];

        const allScans = await fetchAllScansFromFirestore(userId);
        if (allScans.length === 0) return loadMolesFromLocal();

        return allScans;
    } catch (error) {
        console.log('❌ Error loading all scans:', error);
        return loadMolesFromLocal();
    }
};

export const loadMolesFromLocal = async (): Promise<Mole[]> => {
    try {
        const saved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (error) {
        console.log('❌ Error loading from AsyncStorage:', error);
        return [];
    }
};

export const saveMole = async (mole: Mole): Promise<void> => {
    try {
        const existing = await loadMolesFromLocal();
        const index = existing.findIndex(m => m.id === mole.id);
        if (index >= 0) {
            existing[index] = mole;
        } else {
            existing.unshift(mole);
        }
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(existing));
        await saveMoleToFirestore(mole);
    } catch (error) {
        console.log('❌ Error saving mole:', error);
    }
};

export const deleteMole = async (moleId: string): Promise<Mole[]> => {
    try {
        const existing = await loadMolesFromLocal();
        const updated = existing.filter(m => m.id !== moleId);
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(updated));

        const userId = auth.currentUser?.uid;
        if (userId) {
            await deleteDoc(doc(db, 'users', userId, 'scans', moleId));
        }
        return updated;
    } catch (error) {
        console.log('❌ Error deleting mole:', error);
        return [];
    }
};

export const syncLocalMolesToFirestore = async (): Promise<void> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) return;
        const localMoles = await loadMolesFromLocal();
        for (const mole of localMoles) {
            await saveMoleToFirestore(mole);
        }
    } catch (error) {
        console.log('❌ Error syncing moles:', error);
    }
};