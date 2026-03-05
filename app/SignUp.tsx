import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Label } from "@react-navigation/elements";
import { router, useRouter } from "expo-router";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import React, { useEffect, useState } from "react";
import {
    Alert,
    Image,
    KeyboardAvoidingView,
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





const STORAGE_KEY = "signupDraft";

export default function SignUp() {
    const Router = useRouter();

    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");

    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const [emailError, setEmailError] = useState("");
    const [passwordError, setPasswordError] = useState("");
    const [confirmError, setConfirmError] = useState("");

    const [isEmailVerified, setIsEmailVerified] = useState(false);
    const [verifiedEmail, setVerifiedEmail] = useState("");

    // ── Load saved data ───────────────────────────────────────
    useEffect(() => {
        const loadSavedData = async () => {
            try {
                const saved = await AsyncStorage.getItem(STORAGE_KEY);
                if (saved) {
                    const data = JSON.parse(saved);
                    setFirstName(data.firstName || "");
                    setLastName(data.lastName || "");
                    setEmail(data.email || "");
                    // password is never stored in AsyncStorage for security
                    setIsEmailVerified(data.isEmailVerified || false);
                    setVerifiedEmail(data.isEmailVerified ? data.email : "");
                }
            } catch (err) {
                console.log("Load Error:", err);
            }
        };
        loadSavedData();
    }, []);

    // ── Email verified check ──────────────────────────────────
    useEffect(() => {
        if (email === verifiedEmail && verifiedEmail !== "") {
            setIsEmailVerified(true);
        } else {
            setIsEmailVerified(false);
        }
    }, [email, verifiedEmail]);

    // ── Email validation ──────────────────────────────────────
    useEffect(() => {
        const emailRegex = /^\S+@\S+\.\S+$/;
        if (!email) setEmailError("");
        else if (!emailRegex.test(email)) setEmailError("Please enter a valid email");
        else setEmailError("");
    }, [email]);

    // ── Password validation ───────────────────────────────────
    useEffect(() => {
        if (!password) { setPasswordError(""); return; }
        if (password.length < 8)
            setPasswordError("Password must be at least 8 characters");
        else if (!/[A-Z]/.test(password))
            setPasswordError("Must contain at least one uppercase letter (A-Z)");
        else if (!/[a-z]/.test(password))
            setPasswordError("Must contain at least one lowercase letter (a-z)");
        else if (!/[0-9]/.test(password))
            setPasswordError("Must contain at least one number (0-9)");
        else if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password))
            setPasswordError("Must contain at least one special character (@, #, !...)");
        else setPasswordError("");
    }, [password]);

    // ── Confirm password validation ───────────────────────────
    useEffect(() => {
        if (!confirmPassword) setConfirmError("");
        else if (confirmPassword !== password) setConfirmError("Passwords do not match");
        else setConfirmError("");
    }, [confirmPassword, password]);

    // ── isFormValid ───────────────────────────────────────────
    const isFormValid =
        firstName &&
        lastName &&
        email &&
        !emailError &&
        isEmailVerified &&
        password.length >= 8 &&
        /[A-Z]/.test(password) &&
        /[a-z]/.test(password) &&
        /[0-9]/.test(password) &&
        /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password) &&
        confirmPassword === password &&
        !confirmError;

    // ── Save to AsyncStorage ──────────────────────────────────
    const saveData = async (updatedFields: {
        firstName?: string; lastName?: string;
        email?: string; password?: string;
    }) => {
        try {
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            const data = saved ? JSON.parse(saved) : {};
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, ...updatedFields }));
        } catch (err) {
            console.log(err);
        }
    };
    const handleSignUp = async () => {

        try {
            const userCredential = await createUserWithEmailAndPassword(
                auth,
                email,
                password
            );
            const user = userCredential.user;

            await setDoc(doc(db, "users", user.uid), {
                firstName,
                lastName,
                email,
                createdAt: new Date().toISOString(),
                uid: user.uid,
            });

            // Save uid locally
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            const data = saved ? JSON.parse(saved) : {};
            await AsyncStorage.setItem(
                STORAGE_KEY,
                JSON.stringify({ ...data, uid: user.uid })
            );

            Router.push("/Gender");
        } catch (error: any) {
            console.log("FULL ERROR:", error);
            Alert.alert("Sign Up Failed", error.message);
        }
    };


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
                    {/* Back */}
                    <Pressable onPress={() => router.back()} style={styles.backBtn}>
                        <Ionicons name="arrow-back" size={28} color="black" />
                    </Pressable>

                    <Image
                        source={require("../assets/images/Skinsight.png")}
                        style={styles.logo}
                        resizeMode="contain"
                    />

                    <Text style={styles.title}>Sign Up</Text>

                    {/* First Name */}
                    <Label style={styles.label}>First Name</Label>
                    <TextInput
                        placeholder="Enter your first name"
                        style={styles.input}
                        value={firstName}
                        onChangeText={(text) => { setFirstName(text); saveData({ firstName: text }); }}
                    />

                    {/* Last Name */}
                    <Label style={styles.label}>Last Name</Label>
                    <TextInput
                        placeholder="Enter your last name"
                        style={styles.input}
                        value={lastName}
                        onChangeText={(text) => { setLastName(text); saveData({ lastName: text }); }}
                    />

                    {/* Email */}
                    <Label style={styles.label}>Email</Label>
                    <View style={styles.emailWrapper}>
                        <TextInput
                            placeholder="Enter your email"
                            value={email}
                            onChangeText={(text) => {
                                setEmail(text);
                                saveData({ email: text });
                                if (text !== verifiedEmail) setIsEmailVerified(false);
                            }}
                            keyboardType="email-address"
                            autoCapitalize="none"
                            style={styles.emailInput}
                        />
                        <TouchableOpacity
                            disabled={isEmailVerified || !email || !!emailError}
                            style={[
                                styles.verifyBtn,
                                {
                                    backgroundColor: isEmailVerified
                                        ? "#28A745"
                                        : email && !emailError
                                            ? "#004F7F"
                                            : "#BFC6CC",
                                },
                            ]}
                            onPress={async () => {
                                const saved = await AsyncStorage.getItem(STORAGE_KEY);
                                const existing = saved ? JSON.parse(saved) : {};
                                await AsyncStorage.setItem(
                                    STORAGE_KEY,
                                    JSON.stringify({
                                        ...existing,
                                        email,
                                        firstName,
                                        lastName,
                                    })
                                );
                                router.push("/Verifyemail");
                            }}
                        >
                            <Text style={styles.verifyText}>
                                {isEmailVerified ? "Verified ✓" : "Verify"}
                            </Text>
                        </TouchableOpacity>
                    </View>
                    {!!emailError && <Text style={styles.errorText}>{emailError}</Text>}

                    {/* Password */}
                    <Label style={styles.label}>Create Password</Label>
                    <View style={styles.passwordWrapper}>
                        <TextInput
                            placeholder="Create your password"
                            secureTextEntry={!showPassword}
                            value={password}
                            textContentType="oneTimeCode"
                            onChangeText={(text) => { setPassword(text); }}
                            style={styles.passwordInput}
                        />
                        <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eyeIcon}>
                            <Ionicons name={showPassword ? "eye" : "eye-off"} size={22} color="#6B7280" />
                        </TouchableOpacity>
                    </View>
                    {!!passwordError && <Text style={styles.errorText}>{passwordError}</Text>}

                    {/* Strength checklist */}
                    {password.length > 0 && (
                        <View style={styles.strengthContainer}>
                            {[
                                { label: "8+ characters",                   pass: password.length >= 8 },
                                { label: "Uppercase letter (A-Z)",          pass: /[A-Z]/.test(password) },
                                { label: "Lowercase letter (a-z)",          pass: /[a-z]/.test(password) },
                                { label: "Number (0-9)",                    pass: /[0-9]/.test(password) },
                                { label: "Special character (@, #, !...)",  pass: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password) },
                            ].map((item) => (
                                <View key={item.label} style={styles.strengthRow}>
                                    <Ionicons
                                        name={item.pass ? "checkmark-circle" : "ellipse-outline"}
                                        size={16}
                                        color={item.pass ? "#22C55E" : "#9CA3AF"}
                                    />
                                    <Text style={[styles.strengthText, { color: item.pass ? "#22C55E" : "#9CA3AF" }]}>
                                        {item.label}
                                    </Text>
                                </View>
                            ))}
                        </View>
                    )}

                    {/* Confirm Password */}
                    <Label style={styles.label}>Confirm Password</Label>
                    <View style={styles.passwordWrapper}>
                        <TextInput
                            placeholder="Re-enter your password"
                            secureTextEntry={!showConfirmPassword}
                            value={confirmPassword}
                            textContentType="oneTimeCode"  // ← أضف
                            onChangeText={setConfirmPassword}
                            style={styles.passwordInput}
                        />
                        <TouchableOpacity onPress={() => setShowConfirmPassword(!showConfirmPassword)} style={styles.eyeIcon}>
                            <Ionicons name={showConfirmPassword ? "eye" : "eye-off"} size={22} color="#6B7280" />
                        </TouchableOpacity>
                    </View>
                    {!!confirmError && <Text style={styles.errorText}>{confirmError}</Text>}

                    {/* SignUp Button */}
                    <TouchableOpacity
                        disabled={!isFormValid}
                        onPress={handleSignUp}
                        style={[styles.signUpBtn, { backgroundColor: isFormValid ? "#004F7F" : "#B0B0B0" }]}
                    >
                        <Text style={styles.signUpText}>Sign Up</Text>
                    </TouchableOpacity>

                    <Text style={styles.termsText}>
                        By creating an account or signing you agree to our Terms and Conditions
                    </Text>

                    <View style={styles.loginRow}>
                        <Text>Have an account? </Text>
                        <TouchableOpacity onPress={() => router.push("/Login1")}>
                            <Text style={styles.loginLink}>Login</Text>
                        </TouchableOpacity>
                    </View>

                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: "#D8E9F0" },
    scrollContent: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 380, flexGrow: 2 },
    backBtn: {
        width: 42, height: 42, backgroundColor: "#D8E9F0",
        borderRadius: 15, borderColor: "#000", borderWidth: 0.5,
        justifyContent: "center", alignItems: "center", elevation: 3,
    },
    logo: { width: 250, height: 50, alignSelf: "center", marginTop: 20 },
    title: { fontSize: 32, fontWeight: "bold", textAlign: "center", marginTop: 20, marginBottom: 10 },
    label: { fontSize: 15, marginTop: 16, alignSelf: "flex-start", marginBottom: 5 },
    input: {
        borderWidth: 1, borderColor: "#000", backgroundColor: "#fff",
        borderRadius: 8, padding: 12,
    },
    emailWrapper: { position: "relative", marginTop: 5 },
    emailInput: {
        borderWidth: 1, borderColor: "#000", backgroundColor: "#fff",
        borderRadius: 8, padding: 12, paddingRight: 110,
    },
    verifyBtn: {
        position: "absolute", right: 10, top: "50%", transform: [{ translateY: -16 }],
        paddingHorizontal: 16, paddingVertical: 6, borderRadius: 20,
        elevation: 4, shadowColor: "#000", shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25, shadowRadius: 3,
    },
    verifyText: { color: "#fff", fontWeight: "600", fontSize: 13 },
    passwordWrapper: { position: "relative", marginTop: 5 },
    passwordInput: {
        borderWidth: 1, borderColor: "#000", backgroundColor: "#fff",
        borderRadius: 8, padding: 12, paddingRight: 48,
    },
    eyeIcon: { position: "absolute", right: 12, height: "100%", justifyContent: "center" },
    errorText: { color: "red", marginTop: 6, fontSize: 13 },
    strengthContainer: { marginTop: 10, gap: 6 },
    strengthRow: { flexDirection: "row", alignItems: "center", gap: 6 },
    strengthText: { fontSize: 13 },
    signUpBtn: { marginTop: 28, padding: 14, borderRadius: 8 },
    signUpText: { color: "#fff", textAlign: "center", fontWeight: "bold", fontSize: 16 },
    termsText: { marginTop: 15, paddingHorizontal: 10, textAlign: "center", color: "#6B7280", fontSize: 13 },
    loginRow: { flexDirection: "row", justifyContent: "center", marginTop: 10 },
    loginLink: { color: "#004F7F", textDecorationLine: "underline" },
});