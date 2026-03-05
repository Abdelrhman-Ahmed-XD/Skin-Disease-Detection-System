import { Stack } from "expo-router";
import { CustomizeProvider } from "./Customize/Customizecontext"; // ← adjust path if needed
import { ThemeProvider } from "./ThemeContext";
import React from "react";

export default function RootLayout() {
  return (
    <CustomizeProvider>
      <ThemeProvider>
        <Stack screenOptions={{ headerShown: false }} />
      </ThemeProvider>
    </CustomizeProvider>
  );
}