import AsyncStorage from '@react-native-async-storage/async-storage';
import {
    collection, deleteDoc, doc, getDocs,
    query, setDoc, orderBy
} from 'firebase/firestore';
import { db, auth } from '../Firebase/firebaseConfig';

const MOLES_STORAGE_KEY = 'savedMoles';

export type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back' | 'N/A' | string;
    firestoreId?: string;
    analysis?: string;
    source?: string;
};

// ── Shared helper: fetch ALL scans from Firestore ─────────────────────────────
const fetchAllScansFromFirestore = async (userId: string): Promise<Mole[]> => {
    const scansRef = collection(db, 'users', userId, 'scans');
    const q = query(scansRef, orderBy('createdAt', 'desc'));
    const snapshot = await getDocs(q);
    if (snapshot.empty) return [];
    return snapshot.docs.map(docSnap => {
        const data = docSnap.data();
        return {
            id:          docSnap.id,
            x:           data.x           ?? 0,
            y:           data.y           ?? 0,
            timestamp:   data.createdAt?.toMillis?.() ?? (typeof data.createdAt === 'string' ? new Date(data.createdAt).getTime() : Date.now()),
            photoUri:    data.photoUri    ?? '',
            bodyView:    data.bodyView    ?? 'N/A',
            firestoreId: docSnap.id,
            analysis:    data.analysis    ?? '',
            source:      data.source      ?? 'mobile',
        } as Mole;
    });
};

export const saveMoleToFirestore = async (mole: Mole): Promise<void> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) {
            console.log('⚠️ No user logged in - saving to AsyncStorage only');
            return;
        }
        const moleRef = doc(db, 'users', userId, 'scans', mole.id);
        await setDoc(moleRef, {
            ...mole,
            userId,
            updatedAt: Date.now(),
            source: mole.source || 'mobile',
        });
        console.log(`✅ Mole ${mole.id} saved to Firestore`);
    } catch (error) {
        console.log('❌ Error saving mole to Firestore:', error);
    }
};

// ── For the BODY MAP — mobile scans only (source !== 'web' and != 'N/A') ──────
export const loadMolesFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) {
            console.log('⚠️ No user logged in - loading from AsyncStorage');
            return loadMolesFromLocal();
        }

        const allScans = await fetchAllScansFromFirestore(userId);

        if (allScans.length === 0) {
            console.log('📭 No moles in Firestore, loading from local');
            return loadMolesFromLocal();
        }

        // Body map: filter out web scans and scans without a body location
        const moles = allScans.filter(m => m.source !== 'web' && m.bodyView !== 'N/A');
        console.log(`✅ Loaded ${moles.length} mobile scans for body map`);

        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(moles));
        return moles;

    } catch (error) {
        console.log('❌ Error loading from Firestore, falling back to local:', error);
        return loadMolesFromLocal();
    }
};

// ── For HISTORY & REPORTS — ALL scans (mobile + web) ─────────────────────────
export const loadAllScansFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) {
            console.log('⚠️ No user logged in');
            return [];
        }

        const allScans = await fetchAllScansFromFirestore(userId);

        if (allScans.length === 0) {
            console.log('📭 No scans in Firestore');
            return loadMolesFromLocal();
        }

        console.log(`✅ Loaded ${allScans.length} total scans for History/Reports`);
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
        console.log('✅ Mole saved to AsyncStorage');
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
        console.log('✅ Mole deleted from AsyncStorage');

        const userId = auth.currentUser?.uid;
        if (userId) {
            await deleteDoc(doc(db, 'users', userId, 'scans', moleId));
            console.log('✅ Scan deleted from Firestore');
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
        if (localMoles.length === 0) return;
        console.log(`🔄 Syncing ${localMoles.length} local moles to Firestore...`);
        for (const mole of localMoles) {
            await saveMoleToFirestore(mole);
        }
        console.log('✅ All local moles synced to Firestore');
    } catch (error) {
        console.log('❌ Error syncing moles:', error);
    }
};