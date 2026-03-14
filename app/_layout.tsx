import { Stack, router } from "expo-router";
import { onAuthStateChanged } from "firebase/auth";
import React, { useEffect, useState } from "react";
import { auth } from "../Firebase/firebaseConfig";
import { CustomizeProvider } from "./Customize/Customizecontext";
import { ThemeProvider } from "./ThemeContext";

export let isLoggingIn = false;
export const setIsLoggingIn = (val: boolean) => {
  isLoggingIn = val;
};

export default function RootLayout() {
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (user) => {
      if (user && !isLoggingIn) {
        setTimeout(() => {
          router.replace("/Screensbar/FirstHomePage");
        }, 3000); // ← انتظر الـ splash screen تخلص
      }
      setAuthChecked(true);
    });
    return unsub;
  }, []);

  return (
    <CustomizeProvider>
      <ThemeProvider>
        <Stack screenOptions={{ headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen
            name="StartUp"
            listeners={{
              focus: () => {
                if (auth.currentUser && !isLoggingIn) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />
          <Stack.Screen
            name="Login1"
            listeners={{
              focus: () => {
                if (auth.currentUser && !isLoggingIn) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />
          <Stack.Screen
            name="SignUp"
            listeners={{
              focus: () => {
                if (auth.currentUser && !isLoggingIn) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />
        </Stack>
      </ThemeProvider>
    </CustomizeProvider>
  );
}
