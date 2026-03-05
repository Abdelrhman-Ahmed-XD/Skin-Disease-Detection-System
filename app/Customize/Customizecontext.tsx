import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { createContext, useContext, useEffect, useState } from 'react';

// ─── Types ─────────────────────────────────────────────────────────────────────
export type FontFamily = 'System' | 'Inter' | 'SpaceMono' | 'Roboto';
export type Language   = 'English' | 'Arabic';

export type CustomizeSettings = {
  language:        Language;
  fontFamily:      FontFamily;
  fontSize:        number;   // 12–36
  textColor:       string;
  backgroundColor: string;
};

type CustomizeContextType = {
  settings:    CustomizeSettings;
  setSettings: (s: CustomizeSettings) => void;
  saveSettings: (s: CustomizeSettings) => Promise<void>;
};

// ─── Defaults ──────────────────────────────────────────────────────────────────
export const DEFAULT_SETTINGS: CustomizeSettings = {
  language:        'English',
  fontFamily:      'Inter',
  fontSize:        16,
  textColor:       '#1F2937',
  backgroundColor: '#D8E9F0',
};

const CUSTOMIZE_KEY = 'appCustomizeSettings';

// ─── Context ───────────────────────────────────────────────────────────────────
const CustomizeContext = createContext<CustomizeContextType>({
  settings:     DEFAULT_SETTINGS,
  setSettings:  () => {},
  saveSettings: async () => {},
});

// ─── Provider ──────────────────────────────────────────────────────────────────
export function CustomizeProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettingsState] = useState<CustomizeSettings>(DEFAULT_SETTINGS);

  // Load saved settings on mount
  useEffect(() => {
    const load = async () => {
      try {
        const saved = await AsyncStorage.getItem(CUSTOMIZE_KEY);
        if (saved) setSettingsState(JSON.parse(saved));
      } catch (err) {
        console.log('Error loading customize settings:', err);
      }
    };
    load();
  }, []);

  const setSettings = (s: CustomizeSettings) => setSettingsState(s);

  const saveSettings = async (s: CustomizeSettings) => {
    setSettingsState(s);
    try {
      await AsyncStorage.setItem(CUSTOMIZE_KEY, JSON.stringify(s));
    } catch (err) {
      console.log('Error saving customize settings:', err);
    }
  };

  return (
    <CustomizeContext.Provider value={{ settings, setSettings, saveSettings }}>
      {children}
    </CustomizeContext.Provider>
  );
}

// ─── Hook ───────────────────────────────────────────────────────────────────────
export function useCustomize() {
  return useContext(CustomizeContext);
}