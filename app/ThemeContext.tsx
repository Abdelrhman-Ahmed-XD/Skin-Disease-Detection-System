import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { createContext, useContext, useEffect, useState } from "react";

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

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem("darkMode").then((val) => {
      if (val === "true") setIsDark(true);
    });
  }, []);

  const toggleTheme = () => {
    setIsDark((prev) => {
      const next = !prev;
      AsyncStorage.setItem("darkMode", String(next));
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