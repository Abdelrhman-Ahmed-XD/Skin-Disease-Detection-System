import AsyncStorage from '@react-native-async-storage/async-storage';
import {
    collection, doc, getDocs,
    query, setDoc, updateDoc, orderBy
} from 'firebase/firestore';
import { db, auth } from '../Firebase/firebaseConfig';

const MOLES_STORAGE_KEY   = 'savedMoles';
const ALL_SCANS_CACHE_KEY = 'savedAllScans';

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
    isDeleted?: boolean;
    result?: MoleResult;
    // Legacy flat fields (for backward compatibility)
    analysis?: string;
    confidence?: number;
    segmentedUrl?: string;
};

// Fetches ALL documents from Firestore including soft-deleted ones.
// Storing the full state (with isDeleted flags) in cache means Device B
// will correctly hide deleted items even offline, once it has synced at least once after a deletion.
const fetchAllScansRaw = async (userId: string): Promise<Mole[]> => {
    const scansRef = collection(db, 'users', userId, 'scans');
    const q = query(scansRef, orderBy('createdAt', 'desc'));
    const snapshot = await getDocs(q);
    if (snapshot.empty) return [];

    return snapshot.docs.map(docSnap => {
        const data = docSnap.data();

        let result: MoleResult = data.result || {};

        // Legacy flat field migration
        if (!data.result && (data.analysis || data.status)) {
            result = {
                status:       data.status,
                disease:      data.analysis || data.disease,
                confidence:   data.confidence,
                segmentedUrl: data.segmentedUrl || data.segmented_url,
                description:  data.description,
                tips:         data.tips        || [],
                precautions:  data.precautions || [],
                message:      data.message,
            };
        }

        return {
            id:          docSnap.id,
            x:           data.x ?? 0,
            y:           data.y ?? 0,
            timestamp:   data.createdAt?.toMillis?.() ?? (typeof data.createdAt === 'string' ? new Date(data.createdAt).getTime() : Date.now()),
            photoUri:    data.photoUri  ?? '',
            bodyView:    data.bodyView  ?? 'N/A',
            firestoreId: docSnap.id,
            source:      data.source    ?? 'mobile',
            isDeleted:   data.isDeleted ?? false,
            result,
            // Keep legacy for safety
            analysis:    data.analysis,
            confidence:  data.confidence,
            segmentedUrl: data.segmentedUrl,
        } as Mole;
    });
};

const activeOnly = (scans: Mole[]): Mole[] => scans.filter(m => !m.isDeleted);

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

        const allRaw = await fetchAllScansRaw(userId);
        if (allRaw.length === 0) return loadMolesFromLocal();

        // Save full list (with isDeleted flags) so offline reads stay accurate
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(allRaw));

        // MoleMap only shows active mobile scans with a body view
        return activeOnly(allRaw).filter(m => m.source !== 'web' && m.bodyView !== 'N/A');
    } catch (error) {
        console.log('❌ Error loading from Firestore:', error);
        return loadMolesFromLocal();
    }
};

// Reads cache and filters out soft-deleted items
const loadAllScansFromCache = async (): Promise<Mole[]> => {
    try {
        const saved = await AsyncStorage.getItem(ALL_SCANS_CACHE_KEY);
        const scans: Mole[] = saved ? JSON.parse(saved) : [];
        return activeOnly(scans);
    } catch {
        return [];
    }
};

export const loadAllScansFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) return [];

        const allRaw = await fetchAllScansRaw(userId);

        // Always overwrite cache with full Firestore state (including isDeleted flags).
        // This ensures Device B, after syncing, has accurate flags for offline reads.
        await AsyncStorage.setItem(ALL_SCANS_CACHE_KEY, JSON.stringify(allRaw));

        return activeOnly(allRaw);
    } catch (error) {
        console.log('❌ Error loading all scans, using cache:', error);
        return loadAllScansFromCache();
    }
};

export const loadMolesFromLocal = async (): Promise<Mole[]> => {
    try {
        const saved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
        const moles: Mole[] = saved ? JSON.parse(saved) : [];
        return activeOnly(moles);
    } catch (error) {
        console.log('❌ Error loading from AsyncStorage:', error);
        return [];
    }
};

export const saveMole = async (mole: Mole): Promise<void> => {
    try {
        const saved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
        const existing: Mole[] = saved ? JSON.parse(saved) : [];
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

// Soft-delete: marks isDeleted: true in Firestore and both local caches.
// The document is preserved in Firestore for data recovery.
// Any device that syncs after this call will correctly hide the scan even while offline.
export const deleteMole = async (moleId: string): Promise<void> => {
    try {
        // Mark as deleted in ALL_SCANS_CACHE_KEY
        const allSaved = await AsyncStorage.getItem(ALL_SCANS_CACHE_KEY);
        if (allSaved) {
            const allScans: Mole[] = JSON.parse(allSaved);
            await AsyncStorage.setItem(
                ALL_SCANS_CACHE_KEY,
                JSON.stringify(allScans.map(m => m.id === moleId ? { ...m, isDeleted: true } : m))
            );
        }

        // Mark as deleted in MOLES_STORAGE_KEY
        const molesSaved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
        if (molesSaved) {
            const moles: Mole[] = JSON.parse(molesSaved);
            await AsyncStorage.setItem(
                MOLES_STORAGE_KEY,
                JSON.stringify(moles.map(m => (m.id === moleId || m.firestoreId === moleId) ? { ...m, isDeleted: true } : m))
            );
        }

        // Soft-delete in Firestore
        const userId = auth.currentUser?.uid;
        if (userId) {
            await updateDoc(doc(db, 'users', userId, 'scans', moleId), {
                isDeleted: true,
                deletedAt: Date.now(),
            });
        }
    } catch (error) {
        console.log('❌ Error deleting mole:', error);
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
