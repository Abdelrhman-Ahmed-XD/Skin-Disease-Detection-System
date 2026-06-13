import { Stack, router } from "expo-router";
import { onAuthStateChanged } from "firebase/auth";
import React, { useEffect, useRef, useState } from "react";
import { auth } from "../Firebase/firebaseConfig";
import { CustomizeProvider } from "./Customize/Customizecontext";
import { ThemeProvider } from "./ThemeContext";

export let isLoggingIn = false;
export const setIsLoggingIn = (val: boolean) => {
  isLoggingIn = val;
};

export default function RootLayout() {
  const [authChecked, setAuthChecked] = useState(false);
  const hasRedirected = useRef(false);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (user) => {
      setAuthChecked(true);
      // On first auth check, if the user is already signed in from a previous session,
      // skip the startup/login screens and go straight to home.
      if (!hasRedirected.current && user && !user.isAnonymous && !isLoggingIn) {
        hasRedirected.current = true;
        router.replace("/Screensbar/FirstHomePage");
      }
    });
    return unsub;
  }, []);

  return (
    <CustomizeProvider>
      <ThemeProvider>
        <Stack
          screenOptions={{
            headerShown: false,
            animation: "fade",
            animationDuration: 200,
            gestureEnabled: true,
          }}
        >
          <Stack.Screen name="index" />

          <Stack.Screen
            name="StartUp"
            options={{ animation: "fade", animationDuration: 200 }}
            listeners={{
              focus: () => {
                if (
                  auth.currentUser &&
                  !auth.currentUser.isAnonymous &&
                  !isLoggingIn
                ) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />

          <Stack.Screen
            name="Login1"
            options={{ animation: "fade", animationDuration: 200 }}
            listeners={{
              focus: () => {
                if (
                  auth.currentUser &&
                  !auth.currentUser.isAnonymous &&
                  !isLoggingIn
                ) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />

          <Stack.Screen
            name="SignUp"
            options={{ animation: "fade", animationDuration: 200 }}
            listeners={{
              focus: () => {
                if (
                  auth.currentUser &&
                  !auth.currentUser.isAnonymous &&
                  !isLoggingIn
                ) {
                  router.replace("/Screensbar/FirstHomePage");
                }
              },
            }}
          />

          <Stack.Screen
            name="Screensbar/FirstHomePage"
            options={{
              gestureEnabled: false,
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/Camera"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/ResultsScreen"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/History"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/Reports"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/Setting"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/Notifications"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Screensbar/Reportdetails"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Editprofile"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Customize"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/About"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Help"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Changepassword"
            options={{
              animation: "fade",
              animationDuration: 200,
            }}
          />
        </Stack>
      </ThemeProvider>
    </CustomizeProvider>
  );
}