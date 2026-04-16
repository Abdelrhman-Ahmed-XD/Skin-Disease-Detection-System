import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Google from "expo-auth-session/providers/google";
import * as Facebook from "expo-auth-session/providers/facebook";
import * as WebBrowser from "expo-web-browser";
import { router, useRouter } from "expo-router";
import {
    signInWithEmailAndPassword, updatePassword,
    GoogleAuthProvider, FacebookAuthProvider, signInWithCredential,
} from "firebase/auth";
import { doc, getDoc, setDoc, serverTimestamp } from "firebase/firestore";
import React, { useEffect, useRef, useState } from "react";
import {
    ActivityIndicator, Animated, Image, InteractionManager, KeyboardAvoidingView,
    Modal, Platform, Pressable, ScrollView,
    StyleSheet, Text, TextInput, TouchableOpacity, View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { auth, db } from "../Firebase/firebaseConfig";
import { setIsLoggingIn } from "./_layout";
import { loadMolesFromFirestore } from "../Firebase/firestoreService";

// Required for expo-auth-session to close the browser popup after redirect
WebBrowser.maybeCompleteAuthSession();

const STORAGE_KEY  = "signupDraft";
const TOTAL_STEPS  = 4;

// Forces UI to update before continuing heavy async work
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const flushUI = () => new Promise<void>((r) => {
    InteractionManager.runAfterInteractions(() => setTimeout(r, 50));
});

export default function Login1() {
    const Router = useRouter();

    const [showPassword, setShowPassword] = useState(true);
    const [email, setEmail]               = useState("");
    const [password, setPassword]         = useState("");
    const [emailError, setEmailError]     = useState("");
    const [loading, setLoading]           = useState(false);
    const [loadingStatus, setLoadingStatus] = useState("");
    const [loadingStep, setLoadingStep]   = useState(0);

    const [loginError, setLoginError]     = useState(false);
    const [failCount, setFailCount]       = useState(0);
    const [lockSeconds, setLockSeconds]   = useState(0);

    const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
    const scaleAnim = useRef(new Animated.Value(1)).current;

    const isFormValid = !!email && !!password && !emailError;
    const isLocked    = lockSeconds > 0;

    // ── Google OAuth ─────────────────────────────────────────────
    const [googleRequest, googleResponse, googlePromptAsync] = Google.useAuthRequest({
        iosClientId:     process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
        androidClientId: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
        webClientId:     process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
    });

    // ── Facebook OAuth ────────────────────────────────────────────
    const [fbRequest, fbResponse, fbPromptAsync] = Facebook.useAuthRequest({
        clientId: process.env.EXPO_PUBLIC_FACEBOOK_APP_ID,
    });

    const togglePassword = () => setShowPassword(!showPassword);

    // ── Handle Google response ────────────────────────────────────
    useEffect(() => {
        if (googleResponse?.type === "success") {
            const { id_token } = googleResponse.params;
            const credential = GoogleAuthProvider.credential(id_token);
            handleSocialLogin(credential, "google");
        }
    }, [googleResponse]);

    // ── Handle Facebook response ──────────────────────────────────
    useEffect(() => {
        if (fbResponse?.type === "success") {
            const { access_token } = fbResponse.params;
            const credential = FacebookAuthProvider.credential(access_token);
            handleSocialLogin(credential, "facebook");
        }
    }, [fbResponse]);

    // ── Shared social login handler ───────────────────────────────
    const handleSocialLogin = async (credential: any, provider: "google" | "facebook") => {
        setIsLoggingIn(true);
        setLoading(true);
        await flushUI();
        try {
            const goToStep = async (step: number, status: string) => {
                setLoadingStep(step);
                setLoadingStatus(status);
                await flushUI();
            };

            await goToStep(1, `Signing in with ${provider === "google" ? "Google" : "Facebook"}...`);
            const userCredential = await signInWithCredential(auth, credential);
            const user = userCredential.user;

            await goToStep(2, "Loading your profile...");
            const docRef = doc(db, "users", user.uid);
            const docSnap = await getDoc(docRef);

            if (!docSnap.exists()) {
                // First-time social login — create profile in Firestore
                const displayName = user.displayName || "";
                const [firstName, ...rest] = displayName.split(" ");
                await setDoc(docRef, {
                    uid:        user.uid,
                    email:      user.email,
                    firstName:  firstName || "",
                    lastName:   rest.join(" ") || "",
                    photoUri:   user.photoURL || null,
                    provider,
                    createdAt:  serverTimestamp(),
                });
            }

            const data = docSnap.exists() ? docSnap.data() : {};
            await AsyncStorage.setItem("signupDraft", JSON.stringify({
                uid:            user.uid,
                email:          user.email,
                firstName:      data.firstName  || user.displayName?.split(" ")[0] || "",
                lastName:       data.lastName   || "",
                photoUri:       data.photoUri   || user.photoURL || null,
                isEmailVerified: true,
            }));

            await goToStep(3, "Loading your settings...");
            if (data.customizeSettings) {
                await AsyncStorage.setItem(`appCustomizeSettings_${user.uid}`, JSON.stringify(data.customizeSettings));
            }
            if (data.darkMode !== undefined) {
                await AsyncStorage.setItem(`darkMode_${user.uid}`, String(data.darkMode));
            }

            await goToStep(4, "Loading your scans...");
            try { await loadMolesFromFirestore(); } catch (_) {}

            setIsLoggingIn(false);

            // FIXED: Route to Onboarding if first time, else skip to Home
            const hasSeenOnboarding = await AsyncStorage.getItem("hasSeenOnboarding");
            if (hasSeenOnboarding === "true") {
                Router.replace("/Screensbar/FirstHomePage");
            } else {
                Router.replace("/Starthome");
            }

        } catch (err: any) {
            console.log("Social login error:", err);
            setIsLoggingIn(false);
            setLoading(false);
            setLoadingStep(0);
            setLoadingStatus("");
        }
    };

    useEffect(() => {
        const emailRegex = /^\S+@\S+\.\S+$/;
        if (!email) setEmailError("");
        else if (!emailRegex.test(email)) setEmailError("Please enter a valid email");
        else setEmailError("");
    }, [email]);

    useEffect(() => {
        return () => { if (timerRef.current) clearInterval(timerRef.current); };
    }, []);

    const startLockTimer = () => {
        setLockSeconds(60);
        timerRef.current = setInterval(() => {
            setLockSeconds((prev) => {
                if (prev <= 1) {
                    clearInterval(timerRef.current!);
                    timerRef.current = null;
                    setFailCount(0);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);
    };

    const handlePressIn  = () => Animated.spring(scaleAnim, { toValue: 0.96, useNativeDriver: true }).start();
    const handlePressOut = () => Animated.spring(scaleAnim, { toValue: 1,    useNativeDriver: true }).start();

    const handleLogin = async () => {
        if (isLocked) return;
        setLoginError(false);

        // Helper: set step + status, then yield to UI thread so progress bar repaints
        const goToStep = async (step: number, status: string) => {
            setLoadingStep(step);
            setLoadingStatus(status);
            await flushUI();
        };

        // Tell _layout NOT to auto-redirect while we're loading data
        setIsLoggingIn(true);

        // Show modal first and wait for it to fully render
        setLoading(true);
        await flushUI();
        await sleep(150);

        try {
            // ── Step 1: Auth ──────────────────────────────────────
            await goToStep(1, "Signing you in...");
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            const user = userCredential.user;

            // Clear stale data from previous session
            await AsyncStorage.removeItem(STORAGE_KEY);

            // Handle pending password update from forgot-password flow
            const savedRaw  = await AsyncStorage.getItem(STORAGE_KEY);
            const savedData = savedRaw ? JSON.parse(savedRaw) : {};
            if (savedData.pendingPasswordEmail === email && savedData.pendingPasswordUpdate) {
                try {
                    await updatePassword(user, savedData.pendingPasswordUpdate);
                    delete savedData.pendingPasswordUpdate;
                    delete savedData.pendingPasswordEmail;
                    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(savedData));
                } catch (updateErr) {
                    console.log("Password update error:", updateErr);
                }
            }

            // ── Step 2: Profile ───────────────────────────────────
            await goToStep(2, "Loading your profile...");
            const docSnap = await getDoc(doc(db, "users", user.uid));
            if (docSnap.exists()) {
                const data = docSnap.data();
                await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({
                    uid:            user.uid,
                    email:          user.email,
                    firstName:      data.firstName   || "",
                    lastName:       data.lastName    || "",
                    gender:         data.gender      || null,
                    birthDay:       data.birthDay    || null,
                    birthMonth:     data.birthMonth  || null,
                    birthYear:      data.birthYear   || null,
                    skinColor:      data.skinColor   || null,
                    eyeColor:       data.eyeColor    || null,
                    hairColor:      data.hairColor   || null,
                    photoUri:       data.photoUri    || null,
                    isEmailVerified: true,
                }));

                // ── Step 3: Settings (customize + dark mode) ──────
                await goToStep(3, "Loading your settings...");
                if (data.customizeSettings) {
                    await AsyncStorage.setItem(
                        `appCustomizeSettings_${user.uid}`,
                        JSON.stringify(data.customizeSettings)
                    );
                    console.log("✅ Customize settings loaded");
                }
                if (data.darkMode !== undefined) {
                    await AsyncStorage.setItem(`darkMode_${user.uid}`, String(data.darkMode));
                    console.log("✅ Dark mode loaded:", data.darkMode);
                }
            } else {
                await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({
                    uid: user.uid, email: user.email,
                    firstName: "", lastName: "", isEmailVerified: true,
                }));
                // Step 3 still ticks even if no settings saved yet
                await goToStep(3, "Loading your settings...");
            }

            // ── Step 4: Scans ─────────────────────────────────────
            await goToStep(4, "Loading your scans...");
            try {
                await loadMolesFromFirestore();
                console.log("✅ Scans preloaded");
            } catch (err) {
                console.log("⚠️ Scan preload failed, continuing:", err);
            }

            setFailCount(0);
            setIsLoggingIn(false); // allow _layout auth listener again

            // FIXED: Route to Onboarding if first time, else skip to Home
            const hasSeenOnboarding = await AsyncStorage.getItem("hasSeenOnboarding");
            if (hasSeenOnboarding === "true") {
                Router.replace("/Screensbar/FirstHomePage");
            } else {
                Router.replace("/Starthome");
            }

        } catch (error: any) {
            console.log("Firebase error code:", error.code);
            const isCredentialError =
                error.code === "auth/user-not-found"          ||
                error.code === "auth/wrong-password"          ||
                error.code === "auth/invalid-credential"      ||
                error.code === "auth/invalid-login-credentials";
            if (isCredentialError) {
                setLoginError(true);
                setEmail("");
                setPassword("");
                const newCount = failCount + 1;
                setFailCount(newCount);
                if (newCount >= 5) startLockTimer();
            }
        } finally {
            setIsLoggingIn(false);
            setLoading(false);
            setLoadingStep(0);
            setLoadingStatus("");
        }
    };

    const getButtonColor = () => {
        if (isLocked) return "#aeaeae";
        if (loading)  return "#7BAFC4";
        if (isFormValid) return "#004F7F";
        return "#aeaeae";
    };

    const inputErrorStyle = loginError ? styles.inputError : {};

    const STEPS = [
        { step: 1, label: "Auth"     },
        { step: 2, label: "Profile"  },
        { step: 3, label: "Settings" },
        { step: 4, label: "Scans"    },
    ];

    return (
        <SafeAreaView style={styles.container} edges={["top"]}>

            {/* ── Full-screen loading overlay ───────────────────── */}
            <Modal visible={loading} transparent animationType="fade" statusBarTranslucent>
                <View style={styles.overlayContainer}>
                    <View style={styles.overlayCard}>
                        <ActivityIndicator color="#004F7F" size="large" />
                        <Text style={styles.overlayTitle}>{loadingStatus || "Logging in..."}</Text>

                        {/* Step indicators */}
                        <View style={styles.stepsRow}>
                            {STEPS.map(({ step, label }) => (
                                <View key={step} style={styles.stepItem}>
                                    <View style={[
                                        styles.stepDot,
                                        loadingStep >= step && styles.stepDotActive,
                                        loadingStep === step && styles.stepDotCurrent,
                                    ]}>
                                        {loadingStep > step
                                            ? <Text style={styles.stepCheck}>✓</Text>
                                            : <Text style={[styles.stepNum, loadingStep >= step && { color: "#fff" }]}>{step}</Text>
                                        }
                                    </View>
                                    <Text style={[styles.stepLabel, loadingStep >= step && styles.stepLabelActive]}>
                                        {label}
                                    </Text>
                                    {step < TOTAL_STEPS && (
                                        <View style={[styles.stepLine, loadingStep > step && styles.stepLineActive]} />
                                    )}
                                </View>
                            ))}
                        </View>

                        {/* Progress bar */}
                        <View style={styles.progressBarBg}>
                            <View style={[styles.progressBarFill, { width: `${Math.round((loadingStep / TOTAL_STEPS) * 100)}%` }]} />
                        </View>
                        <Text style={styles.progressText}>
                            {Math.round((loadingStep / TOTAL_STEPS) * 100)}% complete
                        </Text>
                    </View>
                </View>
            </Modal>

            <KeyboardAvoidingView
                style={{ flex: 1 }}
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                >
                    <Pressable onPress={() => router.back()} style={styles.backBtn}>
                        <Ionicons name="arrow-back" size={28} color="black" />
                    </Pressable>

                    <Image source={require("../assets/images/Splashscreen1 .png")} style={styles.logo} resizeMode="contain" />
                    <Text style={styles.title}>Log in</Text>

                    <Text style={styles.label}>Email</Text>
                    <TextInput
                        placeholder="Enter your email"
                        value={email}
                        onChangeText={(text) => { setEmail(text); setLoginError(false); }}
                        keyboardType="email-address"
                        autoCapitalize="none"
                        autoCorrect={false}
                        textAlign="left"
                        style={[styles.input, inputErrorStyle]}
                    />
                    {!!emailError && <Text style={styles.errorText}>{emailError}</Text>}

                    <Text style={styles.label}>Password</Text>
                    <View style={styles.passwordWrapper}>
                        <TextInput
                            placeholder="Enter your password"
                            secureTextEntry={showPassword}
                            value={password}
                            onChangeText={(text) => { setPassword(text); setLoginError(false); }}
                            textAlign="left"
                            autoCorrect={false}
                            autoCapitalize="none"
                            autoComplete="off"
                            textContentType="oneTimeCode"
                            spellCheck={false}
                            style={[styles.passwordInput, inputErrorStyle]}
                        />
                        <TouchableOpacity onPress={togglePassword} style={styles.eyeIcon}>
                            <Ionicons name={showPassword ? "eye-off" : "eye"} size={22} color="black" />
                        </TouchableOpacity>
                    </View>

                    {loginError && (
                        <Text style={styles.loginErrorText}>Incorrect email or password. Please try again.</Text>
                    )}

                    <Pressable style={{ marginTop: 10 }} onPress={() => Router.push("/Forgetpassword")}>
                        <Text style={styles.forgetText}>Forget Password?</Text>
                    </Pressable>

                    <Animated.View style={{ transform: [{ scale: scaleAnim }], marginTop: 25 }}>
                        <TouchableOpacity
                            onPress={handleLogin}
                            onPressIn={handlePressIn}
                            onPressOut={handlePressOut}
                            disabled={loading || !isFormValid || isLocked}
                            style={[styles.loginBtn, { backgroundColor: getButtonColor() }]}
                        >
                            <Text style={styles.loginText}>Login</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    <View style={styles.orContainer}>
                        <View style={styles.line} />
                        <Text style={styles.orText}>Or With</Text>
                        <View style={styles.line} />
                    </View>

                    <View style={styles.socialContainer}>
                        <TouchableOpacity
                            style={[styles.socialBtn, !googleRequest && { opacity: 0.5 }]}
                            onPress={() => googlePromptAsync()}
                            disabled={!googleRequest || loading}
                        >
                            <Ionicons name="logo-google" size={24} color="#DB4437" />
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.socialBtn, !fbRequest && { opacity: 0.5 }]}
                            onPress={() => fbPromptAsync()}
                            disabled={!fbRequest || loading}
                        >
                            <Ionicons name="logo-facebook" size={24} color="#1877F2" />
                        </TouchableOpacity>
                    </View>

                    <View style={styles.signupRow}>
                        <Text style={styles.signupText}>Don&#39;t have an account? </Text>
                        <TouchableOpacity onPress={() => Router.push("/SignUp")}>
                            <Text style={styles.signUpText}>Sign Up</Text>
                        </TouchableOpacity>
                    </View>

                    {isLocked && (
                        <View style={styles.timerContainer}>
                            <Ionicons name="time-outline" size={20} color="#D9534F" />
                            <Text style={styles.timerText}>Too many attempts. Try again in {lockSeconds}s</Text>
                        </View>
                    )}
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:        { flex: 1, backgroundColor: "#D8E9F0" },
    scrollContent:    { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 40, flexGrow: 1 },
    backBtn:          { width: 42, height: 42, backgroundColor: "#D8E9F0", borderRadius: 15, borderColor: "#000", borderWidth: 0.5, justifyContent: "center", alignItems: "center", elevation: 3, alignSelf: "flex-start" },
    logo:             { width: 250, height: 50, alignSelf: "center", marginTop: 20 },
    title:            { fontSize: 32, fontWeight: "bold", textAlign: "center", marginTop: 25 },
    label:            { fontSize: 20, marginTop: 30, alignSelf: "flex-start", textAlign: "left" },
    input:            { borderWidth: 1, borderColor: "#000", backgroundColor: "#fff", borderRadius: 8, padding: 12, marginTop: 10, textAlign: "left" },
    inputError:       { borderColor: "#D9534F", shadowColor: "#D9534F", shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.8, shadowRadius: 6, elevation: 4 },
    passwordWrapper:  { position: "relative", marginTop: 10 },
    passwordInput:    { borderWidth: 1, borderColor: "#000", backgroundColor: "#fff", borderRadius: 8, padding: 12, paddingRight: 45, textAlign: "left" },
    eyeIcon:          { position: "absolute", right: 12, top: "50%", transform: [{ translateY: -11 }] },
    errorText:        { color: "red", marginTop: 6, textAlign: "left" },
    loginErrorText:   { color: "#D9534F", marginTop: 8, fontSize: 13, textAlign: "center", fontWeight: "600" },
    forgetText:       { color: "#004F7F", textAlign: "right", textDecorationLine: "underline" },
    loginBtn:         { padding: 14, borderRadius: 8 },
    loginText:        { color: "#fff", textAlign: "center", fontWeight: "bold", fontSize: 16 },
    orContainer:      { flexDirection: "row", alignItems: "center", marginVertical: 25 },
    line:             { flex: 1, height: 1, backgroundColor: "#969696" },
    orText:           { marginHorizontal: 10, color: "#004F7F", fontWeight: "600" },
    socialContainer:  { flexDirection: "row", justifyContent: "center", gap: 60 },
    socialBtn:        { width: 52, height: 52, borderRadius: 12, backgroundColor: "#fff", justifyContent: "center", alignItems: "center", elevation: 3 },
    signupRow:        { flexDirection: "row", justifyContent: "center", marginTop: 50 },
    signupText:       { color: "#000" },
    signUpText:       { color: "#004F7F", textDecorationLine: "underline" },
    timerContainer:   { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 20, padding: 12, backgroundColor: "#FFE5E5", borderRadius: 10, borderWidth: 1, borderColor: "#D9534F" },
    timerText:        { color: "#D9534F", fontWeight: "600", fontSize: 14 },
    // Overlay
    overlayContainer: { flex: 1, backgroundColor: "rgba(0,0,0,0.55)", justifyContent: "center", alignItems: "center" },
    overlayCard:      { backgroundColor: "#fff", borderRadius: 20, padding: 32, width: "80%", alignItems: "center", shadowColor: "#000", shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.25, shadowRadius: 16, elevation: 10 },
    overlayTitle:     { fontSize: 16, fontWeight: "700", color: "#004F7F", marginTop: 16, marginBottom: 24, textAlign: "center" },
    stepsRow:         { flexDirection: "row", alignItems: "center", marginBottom: 32 },
    stepItem:         { alignItems: "center", flexDirection: "row" },
    stepDot:          { width: 32, height: 32, borderRadius: 16, borderWidth: 2, borderColor: "#c0c0c0", backgroundColor: "#f0f0f0", justifyContent: "center", alignItems: "center" },
    stepDotActive:    { backgroundColor: "#004F7F", borderColor: "#004F7F" },
    stepDotCurrent:   { backgroundColor: "#4A9CC2", borderColor: "#4A9CC2" },
    stepNum:          { fontSize: 13, fontWeight: "700", color: "#aaa" },
    stepCheck:        { fontSize: 13, fontWeight: "900", color: "#fff" },
    stepLabel:        { position: "absolute", bottom: -18, left: "50%", fontSize: 10, color: "#aaa", fontWeight: "600", width: 50, textAlign: "center", marginLeft: -25 },
    stepLabelActive:  { color: "#004F7F" },
    stepLine:         { width: 20, height: 2, backgroundColor: "#ddd", marginHorizontal: 4 },
    stepLineActive:   { backgroundColor: "#004F7F" },
    progressBarBg:    { width: "100%", height: 8, backgroundColor: "#E8F0F5", borderRadius: 4, overflow: "hidden", marginTop: 8 },
    progressBarFill:  { height: "100%", backgroundColor: "#004F7F", borderRadius: 4 },
    progressText:     { marginTop: 8, fontSize: 12, color: "#7BAFC4", fontWeight: "600" },
});