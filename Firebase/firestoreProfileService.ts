// app/services/firestoreProfileService.ts
// Handles saving and loading user profile data to/from Firestore

import AsyncStorage from '@react-native-async-storage/async-storage';
import { getApp } from 'firebase/app';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { doc, getDoc, getFirestore, setDoc, updateDoc } from 'firebase/firestore';

const STORAGE_KEY = 'signupDraft';

export type UserProfile = {
    firstName:    string;
    lastName:     string;
    email:        string;
    gender:       string | null;
    birthDay:     number | null;
    birthMonth:   number | null;
    birthYear:    number | null;
    skinColor:    string | null;
    eyeColor:     string | null;
    hairColor:    string | null;
    photoUri:     string | null;
    isEmailVerified: boolean;
    updatedAt?:   string;
    createdAt?:   string;
};

// ── Get current Firebase user ID ──────────────────────────────────────────────
const getUserId = (): string | null => {
    try {
        const auth = getAuth(getApp());
        return auth.currentUser?.uid ?? null;
    } catch {
        return null;
    }
};

// ── Wait for Firebase Auth to restore session (fixes blank profile after login) ──
const waitForAuth = (): Promise<string | null> => {
    return new Promise((resolve) => {
        const auth = getAuth(getApp());
        // If already signed in, resolve immediately
        if (auth.currentUser) {
            resolve(auth.currentUser.uid);
            return;
        }
        // Otherwise wait for the auth state to restore (happens after app restart/login)
        const unsubscribe = auth.onAuthStateChanged((user) => {
            unsubscribe();
            resolve(user?.uid ?? null);
        });
        // Safety timeout — don't wait forever
        setTimeout(() => { unsubscribe(); resolve(null); }, 5000);
    });
};

// ── Save full profile to Firestore (called at end of onboarding) ──────────────
export const saveProfileToFirestore = async (profile: Partial<UserProfile>): Promise<void> => {
    const uid = getUserId();
    if (!uid) {
        console.warn('saveProfileToFirestore: No authenticated user');
        return;
    }

    try {
        const db  = getFirestore(getApp());
        const ref = doc(db, 'users', uid);

        // Check if document already exists to decide set vs update
        // ⚠️ Strip base64 image data — Firestore has a 1MB field limit.
        // photoUri should be a local file path (short string), not base64.
        const safeProfile = { ...profile };
        // Never store password in Firestore — Firebase Auth handles it securely
        delete (safeProfile as any).password;
        // Strip base64 image data — Firestore has a 1MB field limit
        if (safeProfile.photoUri && safeProfile.photoUri.startsWith('data:')) {
            delete safeProfile.photoUri;
        }

        const snap = await getDoc(ref);
        const now  = new Date().toISOString();

        if (snap.exists()) {
            await updateDoc(ref, { ...safeProfile, updatedAt: now });
        } else {
            await setDoc(ref, { ...safeProfile, createdAt: now, updatedAt: now });
        }

        console.log('✅ Profile saved to Firestore');
    } catch (err) {
        console.error('❌ saveProfileToFirestore error:', err);
        throw err;
    }
};

// ── Load profile from Firestore, fallback to AsyncStorage ────────────────────
export const loadProfileFromFirestore = async (): Promise<UserProfile | null> => {
    // Wait for auth to restore — fixes blank profile after logout/login
    const uid = await waitForAuth();

    // Try Firestore first
    if (uid) {
        try {
            const db   = getFirestore(getApp());
            const ref  = doc(db, 'users', uid);
            const snap = await getDoc(ref);

            if (snap.exists()) {
                const data = snap.data() as UserProfile;
                // Also keep AsyncStorage in sync as offline cache
                await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
                console.log('✅ Profile loaded from Firestore');
                return data;
            }
        } catch (err) {
            console.warn('Firestore load failed, falling back to AsyncStorage:', err);
        }
    }

    // Fallback to AsyncStorage
    try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (saved) {
            console.log('✅ Profile loaded from AsyncStorage (offline)');
            return JSON.parse(saved);
        }
    } catch (err) {
        console.error('AsyncStorage load error:', err);
    }

    return null;
};

// ── Save onboarding data to BOTH AsyncStorage and Firestore ──────────────────
export const saveOnboardingData = async (data: Partial<UserProfile>): Promise<void> => {
    // 1. Merge into AsyncStorage
    try {
        const saved  = await AsyncStorage.getItem(STORAGE_KEY);
        const existing = saved ? JSON.parse(saved) : {};
        const merged   = { ...existing, ...data };
        await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
    } catch (err) {
        console.error('AsyncStorage merge error:', err);
    }

    // 2. Save to Firestore
    await saveProfileToFirestore(data);
};

// ── Save full signupDraft to Firestore (called at onboarding completion) ──────
export const flushOnboardingToFirestore = async (): Promise<void> => {
    try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (!saved) return;
        const data: UserProfile = JSON.parse(saved);
        await saveProfileToFirestore(data);
        console.log('✅ Full onboarding data flushed to Firestore');
    } catch (err) {
        console.error('flushOnboardingToFirestore error:', err);
    }
};