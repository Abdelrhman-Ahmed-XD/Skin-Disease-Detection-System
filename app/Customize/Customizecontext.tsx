import AsyncStorage from '@react-native-async-storage/async-storage';
import { onAuthStateChanged } from 'firebase/auth';
import { doc, getDoc, setDoc, updateDoc } from 'firebase/firestore';
import React, {
  createContext, useCallback, useContext, useEffect, useRef, useState,
} from 'react';
import { auth, db } from '../../Firebase/firebaseConfig';

// ─── Font Imports ─────────────────────────────────────────────────────────────
// Run these installs:
// npx expo install @expo-google-fonts/edu-au-vic-wa-nt-hand
// npx expo install @expo-google-fonts/supermercado-one
// npx expo install @expo-google-fonts/playwrite-nz-guides
// npx expo install @expo-google-fonts/playwrite-de-sas
// npx expo install @expo-google-fonts/bricolage-grotesque

import {
  EduAUVICWANTHand_400Regular,
  useFonts as useEduAuVicWaNtHandFonts,
} from '@expo-google-fonts/edu-au-vic-wa-nt-hand';

import {
  SupermercadoOne_400Regular,
  useFonts as useSupermercadoOneFonts,
} from '@expo-google-fonts/supermercado-one';

import {
  PlaywriteNZGuides_400Regular,
  useFonts as usePlaywriteNZFonts,
} from '@expo-google-fonts/playwrite-nz-guides';

import {
  PlaywriteDESAS_100Thin,
  PlaywriteDESAS_200ExtraLight,
  PlaywriteDESAS_300Light,
  PlaywriteDESAS_400Regular,
  useFonts as usePlaywriteDESASFonts,
} from '@expo-google-fonts/playwrite-de-sas';

import {
  BricolageGrotesque_400Regular,
  BricolageGrotesque_500Medium,
  BricolageGrotesque_600SemiBold,
  BricolageGrotesque_700Bold,
  useFonts as useBricolageGrotesqueFonts,
} from '@expo-google-fonts/bricolage-grotesque';

// ─── Types ────────────────────────────────────────────────────────────────────

export type Language = 'English' | 'Arabic';
export type FontFamily =
  | 'System'
  | 'EduAuVicWaNtHand'
  | 'SupermercadoOne'
  | 'PlaywriteNZ'
  | 'PlaywriteDESAS'
  | 'BricolageGrotesque';

export interface CustomizeSettings {
  fontFamilyFONT_FAMILY_MAP: any;
  language:            Language;
  fontFamily:          FontFamily;
  fontSize:            number;
  textColor:           string;
  backgroundColor:     string;
  textColorCustomized: boolean;
}

export const DEFAULT_SETTINGS: CustomizeSettings = {
  language: 'English',
  fontFamily: 'System',
  fontSize: 16,
  textColor: '#1F2937',
  backgroundColor: '#D8E9F0',
  textColorCustomized: false,
  fontFamilyFONT_FAMILY_MAP: undefined
};

// ─── Font name map ────────────────────────────────────────────────────────────
// Usage in any screen:
// <Text style={{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }}>...</Text>

export const FONT_FAMILY_MAP: Record<FontFamily, string | undefined> = {
  System:             undefined,                        // React Native default
  EduAuVicWaNtHand:   'EduAUVICWANTHand_400Regular',
  SupermercadoOne:    'SupermercadoOne_400Regular',
  PlaywriteNZ:        'PlaywriteNZGuides_400Regular',
  PlaywriteDESAS:     'PlaywriteDESAS_400Regular',
  BricolageGrotesque: 'BricolageGrotesque_400Regular',
};

// ─── Context ──────────────────────────────────────────────────────────────────

interface CustomizeContextValue {
  settings:           CustomizeSettings;
  saveSettings:       (s: CustomizeSettings) => Promise<void>;
  effectiveTextColor: (isDark: boolean) => string;
  fontsLoaded:        boolean;
}

const CustomizeContext = createContext<CustomizeContextValue>({
  settings:           DEFAULT_SETTINGS,
  saveSettings:       async () => {},
  effectiveTextColor: () => DEFAULT_SETTINGS.textColor,
  fontsLoaded:        false,
});

// ─── Provider ─────────────────────────────────────────────────────────────────

export function CustomizeProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<CustomizeSettings>(DEFAULT_SETTINGS);
  const uidRef = useRef<string | null>(null);

  // ── Load all Google Fonts ──
  const [eduLoaded]       = useEduAuVicWaNtHandFonts({ EduAUVICWANTHand_400Regular });
  const [supLoaded]       = useSupermercadoOneFonts({ SupermercadoOne_400Regular });
  const [pwNZLoaded]      = usePlaywriteNZFonts({ PlaywriteNZGuides_400Regular });
  const [pwDESASLoaded]   = usePlaywriteDESASFonts({
    PlaywriteDESAS_100Thin,
    PlaywriteDESAS_200ExtraLight,
    PlaywriteDESAS_300Light,
    PlaywriteDESAS_400Regular,
  });
  const [bricolageLoaded] = useBricolageGrotesqueFonts({
    BricolageGrotesque_400Regular,
    BricolageGrotesque_500Medium,
    BricolageGrotesque_600SemiBold,
    BricolageGrotesque_700Bold,
  });

  const fontsLoaded =
    !!eduLoaded &&
    !!supLoaded &&
    !!pwNZLoaded &&
    !!pwDESASLoaded &&
    !!bricolageLoaded;

  // Per-user AsyncStorage key
  const storageKey = (uid: string | null) =>
    uid ? `appCustomizeSettings_${uid}` : 'appCustomizeSettings_guest';

  // Load from AsyncStorage (instant, no flicker)
  const loadFromStorage = useCallback(async (uid: string | null) => {
    try {
      const raw = await AsyncStorage.getItem(storageKey(uid));
      if (raw) {
        const parsed: CustomizeSettings = { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
        setSettings(parsed);
        return parsed;
      }
    } catch {}
    return null;
  }, []);

  // Load from Firestore (source of truth after login)
  const loadFromFirestore = useCallback(async (uid: string) => {
    try {
      const snap = await getDoc(doc(db, 'users', uid));
      if (snap.exists()) {
        const data = snap.data();
        if (data?.customizeSettings) {
          const merged: CustomizeSettings = { ...DEFAULT_SETTINGS, ...data.customizeSettings };
          setSettings(merged);
          await AsyncStorage.setItem(storageKey(uid), JSON.stringify(merged));
          console.log('✅ Customize settings loaded from Firestore');
          return merged;
        }
      }
    } catch (e) {
      console.log('⚠️ Could not load customize settings from Firestore:', e);
    }
    return null;
  }, []);

  // Watch auth state
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (user) => {
      if (user) {
        uidRef.current = user.uid;
        try { await AsyncStorage.removeItem('appCustomizeSettings'); } catch {}
        await loadFromStorage(user.uid);
        await loadFromFirestore(user.uid);
      } else {
        uidRef.current = null;
        setSettings(DEFAULT_SETTINGS);
      }
    });
    return unsub;
  }, [loadFromStorage, loadFromFirestore]);

  // Save settings
  const saveSettings = useCallback(async (newSettings: CustomizeSettings) => {
    setSettings(newSettings);
    const uid = uidRef.current;

    try {
      await AsyncStorage.setItem(storageKey(uid), JSON.stringify(newSettings));
    } catch (e) {
      console.log('⚠️ AsyncStorage save failed:', e);
    }

    if (uid) {
      try {
        const userRef = doc(db, 'users', uid);
        await updateDoc(userRef, { customizeSettings: newSettings });
        console.log('✅ Customize settings saved to Firestore');
      } catch {
        try {
          await setDoc(doc(db, 'users', uid), { customizeSettings: newSettings }, { merge: true });
        } catch (e2) {
          console.log('⚠️ Firestore save failed:', e2);
        }
      }
    }
  }, []);

  // Dark-mode aware text color
  const effectiveTextColor = useCallback((isDark: boolean): string => {
    if (settings.textColorCustomized) return settings.textColor;
    return isDark ? '#FFFFFF' : '#1F2937';
  }, [settings.textColor, settings.textColorCustomized]);

  return (
    <CustomizeContext.Provider value={{ settings, saveSettings, effectiveTextColor, fontsLoaded }}>
      {children}
    </CustomizeContext.Provider>
  );
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useCustomize() {
  return useContext(CustomizeContext);
}1