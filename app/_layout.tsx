import { Stack, router } from "expo-router";
import { onAuthStateChanged } from "firebase/auth";
import React, { useEffect, useState } from "react";
import { auth } from "../Firebase/firebaseConfig";
import { CustomizeProvider } from "./Customize/Customizecontext";
import { ThemeProvider } from "./ThemeContext";

// Global flag — Login1 sets this to true before signing in,
// so the onAuthStateChanged listener doesn't jump to home mid-loading
export let isLoggingIn = false;
export const setIsLoggingIn = (val: boolean) => { isLoggingIn = val; };

// Screens that logged-in users should never see
const AUTH_SCREENS = ['index', 'StartUp', 'Login1', 'SignUp', 'Verifyemail'];

export default function RootLayout() {
    const [authChecked, setAuthChecked] = useState(false);

    useEffect(() => {
        const unsub = onAuthStateChanged(auth, async (user) => {
            if (user && !isLoggingIn) {
                // Auto-restore session (app reopen) — go straight to home
                router.replace("/Screensbar/FirstHomePage");
            }
            setAuthChecked(true);
        });
        return unsub;
    }, []);

    return (
        <CustomizeProvider>
            <ThemeProvider>
                <Stack
                    screenOptions={{ headerShown: false }}
                    // Block going back to auth screens when logged in
                >
                    <Stack.Screen name="StartUp" listeners={{
                        focus: () => {
                            if (auth.currentUser && !isLoggingIn) {
                                router.replace("/Screensbar/FirstHomePage");
                            }
                        }
                    }} />
                    <Stack.Screen name="Login1" listeners={{
                        focus: () => {
                            if (auth.currentUser && !isLoggingIn) {
                                router.replace("/Screensbar/FirstHomePage");
                            }
                        }
                    }} />
                    <Stack.Screen name="SignUp" listeners={{
                        focus: () => {
                            if (auth.currentUser && !isLoggingIn) {
                                router.replace("/Screensbar/FirstHomePage");
                            }
                        }
                    }} />
                </Stack>
            </ThemeProvider>
        </CustomizeProvider>
    );
}