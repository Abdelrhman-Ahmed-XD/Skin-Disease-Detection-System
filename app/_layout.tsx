import { Stack, router } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { onAuthStateChanged } from "firebase/auth";
import React, { useEffect, useRef, useState } from "react";
import { auth } from "../Firebase/firebaseConfig";
import { CustomizeProvider } from "./Customize/Customizecontext";
import { ThemeProvider } from "./ThemeContext";

// Keep the native splash visible until React is ready to render
SplashScreen.preventAutoHideAsync().catch(() => {});

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
            animation: "none",
            animationDuration: 100,
            gestureEnabled: true,
          }}
        >
          <Stack.Screen name="index" />
          <Stack.Screen
            name="StartUp"
            options={{ animation: "none", animationDuration: 100 }}
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
            options={{ animation: "none", animationDuration: 100 }}
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
            options={{ animation: "none", animationDuration: 100 }}
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
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/Camera"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/ResultsScreen"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/History"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/Reports"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/Setting"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/Notifications"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Screensbar/Reportdetails"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Editprofile"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />
          <Stack.Screen name="Guest/Guest" options={{ animation: "fade" }} />
          <Stack.Screen
            name="Guest/reportguest"
            options={{ animation: "fade" }}
          />
          <Stack.Screen
            name="Guest/histroyguest"
            options={{ animation: "fade" }}
          />
          <Stack.Screen
            name="Guest/settingsguest"
            options={{ animation: "fade" }}
          />
<Stack.Screen
  name="Guest/cameraguest"
  options={{ animation: "fade" }}
/>
<Stack.Screen
  name="Guest/GuestResultsScreen"
  options={{ animation: "fade" }}
/>
<Stack.Screen
  name="Guest/aboutguest"
  options={{ animation: "fade" }}
/>
<Stack.Screen
  name="Guest/helpguest"
  options={{ animation: "fade" }}
/>
<Stack.Screen
  name="Guest/notificationguest"
  options={{ animation: "fade" }}
/>
          <Stack.Screen
            name="Settingsoptions/Customize"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/About"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Help"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />

          <Stack.Screen
            name="Settingsoptions/Changepassword"
            options={{
              animation: "none",
              animationDuration: 100,
            }}
          />
        </Stack>
      </ThemeProvider>
    </CustomizeProvider>
  );
}