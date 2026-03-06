import AsyncStorage from "@react-native-async-storage/async-storage";
import { onAuthStateChanged } from "firebase/auth";
import { doc, getDoc, getFirestore, updateDoc, setDoc } from "firebase/firestore";
import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { auth } from "../Firebase/firebaseConfig";
import { getApp } from "firebase/app";

type ThemeContextType = {
  isDark: boolean;
  toggleTheme: () => void;
  colors: typeof lightColors;
};

export const lightColors = {
  background:  "#D8E9F0",
  card:        "#FFFFFF",
  text:        "#1F2937",
  subText:     "#6B7280",
  border:      "#E5E7EB",
  input:       "#FFFFFF",
  primary:     "#004F7F",
  accent:      "#00A3A3",
  navBg:       "#FFFFFF",
  navText:     "#6B7280",
  navActive:   "#004F7F",
  toggleBg:    "#B8D4DE",
  badgeBg:     "#EF4444",
  statusBar:   "dark-content" as "dark-content" | "light-content",
};

export const darkColors = {
  background:  "#0F1923",
  card:        "#1E2A35",
  text:        "#F3F4F6",
  subText:     "#9CA3AF",
  border:      "#374151",
  input:       "#1E2A35",
  primary:     "#004F7F",
  accent:      "#00A3A3",
  navBg:       "#1E2A35",
  navText:     "#9CA3AF",
  navActive:   "#7DD3FC",
  toggleBg:    "#374151",
  badgeBg:     "#EF4444",
  statusBar:   "light-content" as "dark-content" | "light-content",
};

const ThemeContext = createContext<ThemeContextType>({
  isDark: false,
  toggleTheme: () => {},
  colors: lightColors,
});

const themeKey = (uid: string | null) =>
    uid ? `darkMode_${uid}` : "darkMode_guest";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [isDark, setIsDark] = useState(false);
  const uidRef = useRef<string | null>(null);

  // Save dark mode to Firestore
  const saveDarkModeToFirestore = async (uid: string, value: boolean) => {
    try {
      const db  = getFirestore(getApp());
      const ref = doc(db, "users", uid);
      const snap = await getDoc(ref);
      if (snap.exists()) {
        await updateDoc(ref, { darkMode: value });
      } else {
        await setDoc(ref, { darkMode: value }, { merge: true });
      }
      console.log("✅ Dark mode saved to Firestore:", value);
    } catch (e) {
      console.log("⚠️ Could not save dark mode to Firestore:", e);
    }
  };

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (user) => {
      const uid = user?.uid ?? null;
      uidRef.current = uid;

      // Remove old shared key so it never bleeds between users
      try { await AsyncStorage.removeItem("darkMode"); } catch {}

      if (uid) {
        // 1. Load from AsyncStorage instantly (no flicker)
        try {
          const val = await AsyncStorage.getItem(themeKey(uid));
          if (val !== null) setIsDark(val === "true");
        } catch {}

        // 2. Override with Firestore (source of truth)
        try {
          const db   = getFirestore(getApp());
          const snap = await getDoc(doc(db, "users", uid));
          if (snap.exists() && snap.data()?.darkMode !== undefined) {
            const firestoreDark = snap.data()!.darkMode as boolean;
            setIsDark(firestoreDark);
            await AsyncStorage.setItem(themeKey(uid), String(firestoreDark));
            console.log("✅ Dark mode loaded from Firestore:", firestoreDark);
          }
        } catch (e) {
          console.log("⚠️ Could not load dark mode from Firestore:", e);
        }
      } else {
        // Logged out — reset to light
        setIsDark(false);
      }
    });
    return unsub;
  }, []);

  const toggleTheme = () => {
    setIsDark((prev) => {
      const next = !prev;
      // Save to AsyncStorage immediately
      AsyncStorage.setItem(themeKey(uidRef.current), String(next));
      // Save to Firestore in background
      if (uidRef.current) saveDarkModeToFirestore(uidRef.current, next);
      return next;
    });
  };

  return (
      <ThemeContext.Provider value={{ isDark, toggleTheme, colors: isDark ? darkColors : lightColors }}>
        {children}
      </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);