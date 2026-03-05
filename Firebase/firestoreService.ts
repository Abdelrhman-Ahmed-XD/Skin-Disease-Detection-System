import AsyncStorage from '@react-native-async-storage/async-storage';
import {
    collection, deleteDoc, doc, getDocs,
    query, setDoc, where, orderBy
} from 'firebase/firestore';
import { db, auth } from '../Firebase/firebaseConfig';

const MOLES_STORAGE_KEY = 'savedMoles';

export type Mole = {
    id: string;
    x: number;
    y: number;
    timestamp: number;
    photoUri?: string;
    bodyView: 'front' | 'back';
    firestoreId?: string;
    analysis?: string;
};

// ─── Save a single mole to Firestore ────────────────────────────────────────
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
        });
        console.log(`✅ Mole ${mole.id} saved to Firestore`);
    } catch (error) {
        console.log('❌ Error saving mole to Firestore:', error);
    }
};

// ─── Load all moles (Firestore first, fallback to AsyncStorage) ──────────────
export const loadMolesFromFirestore = async (): Promise<Mole[]> => {
    try {
        const userId = auth.currentUser?.uid;
        if (!userId) {
            console.log('⚠️ No user logged in - loading from AsyncStorage');
            return loadMolesFromLocal();
        }

        // Camera saves to 'scans' collection, load from there
        const scansRef = collection(db, 'users', userId, 'scans');
        const q = query(scansRef, orderBy('createdAt', 'desc'));
        const snapshot = await getDocs(q);

        if (snapshot.empty) {
            console.log('📭 No moles in Firestore, loading from local');
            return loadMolesFromLocal();
        }

        // Map Firestore scan docs to Mole type
        const moles: Mole[] = snapshot.docs.map(docSnap => {
            const data = docSnap.data();
            return {
                id:          docSnap.id,
                x:           data.x           ?? 0,
                y:           data.y           ?? 0,
                timestamp:   data.createdAt?.toMillis?.() ?? Date.now(),
                photoUri:    data.photoUri    ?? '',
                bodyView:    data.bodyView    ?? 'front',
                firestoreId: docSnap.id,
                analysis:    data.analysis    ?? '',
            } as Mole;
        });
        console.log(`✅ Loaded ${moles.length} scans from Firestore`);

        // Sync to AsyncStorage as local cache
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(moles));
        return moles;

    } catch (error) {
        console.log('❌ Error loading from Firestore, falling back to local:', error);
        return loadMolesFromLocal();
    }
};

// ─── Load from AsyncStorage (local cache / offline fallback) ─────────────────
export const loadMolesFromLocal = async (): Promise<Mole[]> => {
    try {
        const saved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (error) {
        console.log('❌ Error loading from AsyncStorage:', error);
        return [];
    }
};

// ─── Save mole locally AND to Firestore ──────────────────────────────────────
export const saveMole = async (mole: Mole): Promise<void> => {
    try {
        // 1. Save locally first (instant)
        const existing = await loadMolesFromLocal();
        const index = existing.findIndex(m => m.id === mole.id);
        if (index >= 0) {
            existing[index] = mole; // update
        } else {
            existing.unshift(mole); // add to top
        }
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(existing));
        console.log('✅ Mole saved to AsyncStorage');

        // 2. Sync to Firestore
        await saveMoleToFirestore(mole);
    } catch (error) {
        console.log('❌ Error saving mole:', error);
    }
};

// ─── Delete mole from both AsyncStorage and Firestore ────────────────────────
export const deleteMole = async (moleId: string): Promise<Mole[]> => {
    try {
        // 1. Delete from local
        const existing = await loadMolesFromLocal();
        const updated = existing.filter(m => m.id !== moleId);
        await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(updated));
        console.log('✅ Mole deleted from AsyncStorage');

        // 2. Delete from Firestore
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

// ─── Sync all local moles to Firestore (call on login) ───────────────────────
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