import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router, useRouter } from "expo-router";
import { signInWithEmailAndPassword, updatePassword } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import React, { useEffect, useRef, useState } from "react";
import {
    ActivityIndicator,
    Animated,
    Image,
    KeyboardAvoidingView,
    Linking,
    Platform,
    Pressable,
    ScrollView,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { auth, db } from "../Firebase/firebaseConfig";
import { loadMolesFromFirestore } from "../Firebase/firestoreService";

const STORAGE_KEY = "signupDraft";

export default function Login1() {
    const Router = useRouter();

    const [showPassword, setShowPassword] = useState(true);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [emailError, setEmailError] = useState("");
    const [loading, setLoading]       = useState(false);
    const [loadingStatus, setLoadingStatus] = useState("");

    const [loginError, setLoginError] = useState(false);
    const [failCount, setFailCount] = useState(0);
    const [lockSeconds, setLockSeconds] = useState(0);

    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const scaleAnim = useRef(new Animated.Value(1)).current;

    const isFormValid = !!email && !!password && !emailError;
    const isLocked = lockSeconds > 0;

    const togglePassword = () => setShowPassword(!showPassword);
    const openGoogle = () => Linking.openURL("https://accounts.google.com");
    const openFacebook = () => Linking.openURL("https://www.facebook.com/login/");

    useEffect(() => {
        const emailRegex = /^\S+@\S+\.\S+$/;
        if (!email) setEmailError("");
        else if (!emailRegex.test(email)) setEmailError("Please enter a valid email");
        else setEmailError("");
    }, [email]);

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
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

    const handlePressIn = () => {
        Animated.spring(scaleAnim, { toValue: 0.96, useNativeDriver: true }).start();
    };

    const handlePressOut = () => {
        Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true }).start();
    };

    const handleLogin = async () => {
        if (isLocked) return;
        setLoading(true);
        setLoginError(false);
        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            const user = userCredential.user;

            // ── Clear stale data from any previous user session ──
            await AsyncStorage.removeItem(STORAGE_KEY);

            // ── Check for pending password update from forgot password flow ──
            const savedRaw = await AsyncStorage.getItem(STORAGE_KEY);
            const savedData = savedRaw ? JSON.parse(savedRaw) : {};

            if (
                savedData.pendingPasswordEmail === email &&
                savedData.pendingPasswordUpdate
            ) {
                try {
                    await updatePassword(user, savedData.pendingPasswordUpdate);
                    // Clear pending update
                    delete savedData.pendingPasswordUpdate;
                    delete savedData.pendingPasswordEmail;
                    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(savedData));
                } catch (updateErr) {
                    console.log("Password update error:", updateErr);
                }
            }

            // ── Load user data from Firestore ──
            const docSnap = await getDoc(doc(db, "users", user.uid));
            if (docSnap.exists()) {
                const data = docSnap.data();
                await AsyncStorage.setItem(
                    STORAGE_KEY,
                    JSON.stringify({
                        uid: user.uid,
                        email: user.email,
                        firstName:      data.firstName      || "",
                        lastName:       data.lastName       || "",
                        gender:         data.gender         || null,
                        birthDay:       data.birthDay       || null,
                        birthMonth:     data.birthMonth     || null,
                        birthYear:      data.birthYear      || null,
                        skinColor:      data.skinColor      || null,
                        eyeColor:       data.eyeColor       || null,
                        hairColor:      data.hairColor      || null,
                        photoUri:       data.photoUri       || null,
                        isEmailVerified: true,
                    })
                );
            } else {
                await AsyncStorage.setItem(
                    STORAGE_KEY,
                    JSON.stringify({
                        uid: user.uid,
                        email: user.email,
                        firstName: "",
                        lastName: "",
                        isEmailVerified: true,
                    })
                );
            }

            setFailCount(0);

            // ── Preload scans before navigating ──────────────────
            // This ensures body map shows images immediately on login
            try {
                setLoadingStatus("Loading your scans...");
                await loadMolesFromFirestore();
                console.log("✅ Scans preloaded before navigation");
            } catch (err) {
                console.log("⚠️ Scan preload failed, continuing anyway:", err);
            }

            setLoadingStatus("");
            Router.replace("/Screensbar/FirstHomePage");
        } catch (error: any) {
            console.log("Firebase error code:", error.code);
            const isCredentialError =
                error.code === "auth/user-not-found" ||
                error.code === "auth/wrong-password" ||
                error.code === "auth/invalid-credential" ||
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
            setLoading(false);
        }
    };

    const getButtonColor = () => {
        if (isLocked) return "#aeaeae";
        if (loading) return "#7BAFC4";
        if (isFormValid) return "#004F7F";
        return "#aeaeae";
    };

    const inputErrorStyle = loginError ? styles.inputError : {};

    return (
        <SafeAreaView style={styles.container} edges={["top"]}>
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
                    {/* Back Button */}
                    <Pressable onPress={() => router.back()} style={styles.backBtn}>
                        <Ionicons name="arrow-back" size={28} color="black" />
                    </Pressable>

                    {/* Logo */}
                    <Image
                        source={require("../assets/images/Skinsight.png")}
                        style={styles.logo}
                        resizeMode="contain"
                    />

                    <Text style={styles.title}>Log in</Text>

                    {/* Email */}
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

                    {/* Password */}
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
                            <Ionicons
                                name={showPassword ? "eye-off" : "eye"}
                                size={22}
                                color="black"
                            />
                        </TouchableOpacity>
                    </View>

                    {loginError && (
                        <Text style={styles.loginErrorText}>
                            Incorrect email or password. Please try again.
                        </Text>
                    )}

                    <Pressable
                        style={{ marginTop: 10 }}
                        onPress={() => Router.push("/Forgetpassword")}
                    >
                        <Text style={styles.forgetText}>Forget Password?</Text>
                    </Pressable>

                    {/* Login Button */}
                    <Animated.View style={{ transform: [{ scale: scaleAnim }], marginTop: 25 }}>
                        <TouchableOpacity
                            onPress={handleLogin}
                            onPressIn={handlePressIn}
                            onPressOut={handlePressOut}
                            disabled={loading || !isFormValid || isLocked}
                            style={[styles.loginBtn, { backgroundColor: getButtonColor() }]}
                        >
                            {loading ? (
                                <View style={{ alignItems: "center" }}>
                                    <ActivityIndicator color="#fff" size="small" />
                                    <Text style={[styles.loginText, { fontSize: 12, marginTop: 4 }]}>
                                        {loadingStatus || "Logging in..."}
                                    </Text>
                                </View>
                            ) : (
                                <Text style={styles.loginText}>Login</Text>
                            )}
                        </TouchableOpacity>
                    </Animated.View>

                    {/* OR WITH */}
                    <View style={styles.orContainer}>
                        <View style={styles.line} />
                        <Text style={styles.orText}>Or With</Text>
                        <View style={styles.line} />
                    </View>

                    {/* Social Buttons */}
                    <View style={styles.socialContainer}>
                        <TouchableOpacity style={styles.socialBtn} onPress={openGoogle}>
                            <Ionicons name="logo-google" size={24} color="#DB4437" />
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.socialBtn} onPress={openFacebook}>
                            <Ionicons name="logo-facebook" size={24} color="#1877F2" />
                        </TouchableOpacity>
                    </View>

                    {/* Sign Up Row */}
                    <View style={styles.signupRow}>
                        <Text style={styles.signupText}>Don&#39;t have an account? </Text>
                        <TouchableOpacity onPress={() => Router.push("/SignUp")}>
                            <Text style={styles.signUpText}>Sign Up</Text>
                        </TouchableOpacity>
                    </View>

                    {isLocked && (
                        <View style={styles.timerContainer}>
                            <Ionicons name="time-outline" size={20} color="#D9534F" />
                            <Text style={styles.timerText}>
                                Too many attempts. Try again in {lockSeconds}s
                            </Text>
                        </View>
                    )}
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:       { flex: 1, backgroundColor: "#D8E9F0" },
    scrollContent:   { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 40, flexGrow: 1 },
    backBtn:         { width: 42, height: 42, backgroundColor: "#D8E9F0", borderRadius: 15, borderColor: "#000", borderWidth: 0.5, justifyContent: "center", alignItems: "center", elevation: 3, alignSelf: "flex-start" },
    logo:            { width: 250, height: 50, alignSelf: "center", marginTop: 20 },
    title:           { fontSize: 32, fontWeight: "bold", textAlign: "center", marginTop: 25 },
    label:           { fontSize: 20, marginTop: 30, alignSelf: "flex-start", textAlign: "left" },
    input:           { borderWidth: 1, borderColor: "#000", backgroundColor: "#fff", borderRadius: 8, padding: 12, marginTop: 10, textAlign: "left" },
    inputError:      { borderColor: "#D9534F", shadowColor: "#D9534F", shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.8, shadowRadius: 6, elevation: 4 },
    passwordWrapper: { position: "relative", marginTop: 10 },
    passwordInput:   { borderWidth: 1, borderColor: "#000", backgroundColor: "#fff", borderRadius: 8, padding: 12, paddingRight: 45, textAlign: "left" },
    eyeIcon:         { position: "absolute", right: 12, top: "50%", transform: [{ translateY: -11 }] },
    errorText:       { color: "red", marginTop: 6, textAlign: "left" },
    loginErrorText:  { color: "#D9534F", marginTop: 8, fontSize: 13, textAlign: "center", fontWeight: "600" },
    forgetText:      { color: "#004F7F", textAlign: "right", textDecorationLine: "underline" },
    loginBtn:        { padding: 14, borderRadius: 8 },
    loginText:       { color: "#fff", textAlign: "center", fontWeight: "bold", fontSize: 16 },
    orContainer:     { flexDirection: "row", alignItems: "center", marginVertical: 25 },
    line:            { flex: 1, height: 1, backgroundColor: "#969696" },
    orText:          { marginHorizontal: 10, color: "#004F7F", fontWeight: "600" },
    socialContainer: { flexDirection: "row", justifyContent: "center", gap: 60 },
    socialBtn:       { width: 52, height: 52, borderRadius: 12, backgroundColor: "#fff", justifyContent: "center", alignItems: "center", elevation: 3 },
    signupRow:       { flexDirection: "row", justifyContent: "center", marginTop: 50 },
    signupText:      { color: "#000" },
    signUpText:      { color: "#004F7F", textDecorationLine: "underline" },
    timerContainer:  { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 20, padding: 12, backgroundColor: "#FFE5E5", borderRadius: 10, borderWidth: 1, borderColor: "#D9534F" },
    timerText:       { color: "#D9534F", fontWeight: "600", fontSize: 14 },
});