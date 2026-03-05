import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useLocalSearchParams, useRouter } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
import {
    Alert,
    Image, Pressable, StyleSheet, Text,
    TextInput, TouchableOpacity, View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { auth } from "../Firebase/firebaseConfig";
import { saveProfileToFirestore } from "../Firebase/firestoreProfileService";

const FLASK_URL   = "http://192.168.100.2:5000";
const STORAGE_KEY = "signupDraft";

export default function VerifyEmailChange() {
    const router = useRouter();
    const { newEmail, userName: paramName } = useLocalSearchParams<{ newEmail: string; userName: string }>();

    const [otp, setOtp]               = useState(["", "", "", "", "", ""]);
    const [isFormValid, setIsFormValid] = useState(false);
    const inputs                        = useRef<(TextInput | null)[]>([]);

    const [timer, setTimer]         = useState(60);
    const [canResend, setCanResend] = useState(false);
    const [generatedOtp, setGeneratedOtp] = useState("");
    const [isSending, setIsSending]       = useState(false);
    const [isVerifying, setIsVerifying]   = useState(false);

    // ── Check email via Flask (Firebase Admin) then send OTP ──
    const sendOtpEmail = async () => {
        try {
            setIsSending(true);

            // Step 1: Ask Flask/Firebase Admin if this email is already taken
            console.log("🔍 Checking if email is already in use:", newEmail);
            const checkRes = await fetch(`${FLASK_URL}/api/check-email`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: newEmail }),
            });
            const checkData = await checkRes.json();
            console.log("🔍 Check result:", checkData);

            if (checkData.exists) {
                Alert.alert(
                    "Email Already in Use ❌",
                    `The email "${newEmail}" is already linked to another account. Please go back and choose a different email.`,
                    [{ text: "Go Back", onPress: () => router.back() }]
                );
                return;
            }

            // Step 2: Email is free — generate OTP and send it
            console.log("✅ Email is available, sending OTP...");
            const newOtp = Math.floor(100000 + Math.random() * 900000).toString();
            setGeneratedOtp(newOtp);

            const response = await fetch(`${FLASK_URL}/api/send-email-change-otp`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email: newEmail,
                    name: paramName || "User",
                    otp_code: newOtp,
                }),
            });

            const data = await response.json();
            console.log("📬 Flask Response:", data);

            if (response.ok) {
                console.log("✅ Email change OTP sent!");
                Alert.alert("Code Sent ✅", `A verification code has been sent to ${newEmail}`);
            } else {
                Alert.alert("Error", data.message || "Failed to send verification code.");
            }
        } catch (error: any) {
            console.log("❌ Error:", error.code, error.message);
            Alert.alert(
                "Connection Error",
                `Cannot connect to server.\n\nMake sure:\n1. Flask is running\n2. You're on the same network\n3. IP is 192.168.100.2:5000`
            );
        } finally {
            setIsSending(false);
        }
    };

    // ── Send on mount ─────────────────────────────────────────
    useEffect(() => {
        if (newEmail) sendOtpEmail();
    }, []);

    // ── OTP filled check ──────────────────────────────────────
    useEffect(() => {
        setIsFormValid(otp.every((digit) => digit !== ""));
    }, [otp]);

    // ── Timer ─────────────────────────────────────────────────
    useEffect(() => {
        if (timer === 0) { setCanResend(true); return; }
        const interval = setInterval(() => setTimer((prev) => prev - 1), 1000);
        return () => clearInterval(interval);
    }, [timer]);

    // ── Resend ────────────────────────────────────────────────
    const handleResend = async () => {
        setTimer(60);
        setCanResend(false);
        setOtp(["", "", "", "", "", ""]);
        inputs.current[0]?.focus();
        await sendOtpEmail();
    };

    const handleOtpChange = (text: string, index: number) => {
        const next = [...otp];
        next[index] = text;
        setOtp(next);
        if (text && index < 5) inputs.current[index + 1]?.focus();
    };

    const handleBackspace = (key: string, index: number) => {
        if (key === "Backspace" && !otp[index] && index > 0)
            inputs.current[index - 1]?.focus();
    };

    // ── Verify OTP → update email in Firebase Auth + Firestore ──
    const handleVerify = async () => {
        const entered = otp.join("");

        if (entered !== generatedOtp) {
            Alert.alert(
                "Invalid Code ❌",
                "The code you entered is incorrect. Please try again.",
                [
                    { text: "Try Again", onPress: () => { setOtp(["", "", "", "", "", ""]); inputs.current[0]?.focus(); } },
                    { text: "Resend Code", onPress: handleResend },
                ]
            );
            return;
        }

        setIsVerifying(true);
        try {
            const user = auth.currentUser;
            if (!user) throw new Error("No logged-in user");

            // Update email directly via Flask + Firebase Admin SDK
            // This updates it immediately — no pending verification link
            console.log("🔄 Updating email via Flask Admin to:", newEmail);
            const res = await fetch(`${FLASK_URL}/api/update-email`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ uid: user.uid, new_email: newEmail }),
            });
            const result = await res.json();
            console.log("📬 Flask update-email response:", result);

            if (!result.success) {
                const msg = result.message?.includes("already in use")
                    ? `The email "${newEmail}" is already linked to another account.`
                    : result.message || "Failed to update email. Please try again.";
                Alert.alert("❌ Update Failed", msg);
                return;
            }

            // Persist to AsyncStorage
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            const data  = saved ? JSON.parse(saved) : {};
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({
                ...data,
                email: newEmail,
                isEmailVerified: true,
            }));

            // Persist to Firestore
            await saveProfileToFirestore({ email: newEmail, isEmailVerified: true });

            console.log("✅ Email updated to:", newEmail);
            Alert.alert(
                "✅ Email Updated",
                "Your email has been changed successfully. Please use your new email to sign in.",
                [{ text: "OK", onPress: () => router.back() }]
            );
        } catch (err: any) {
            console.error("Email update error:", err.message);
            Alert.alert("❌ Update Failed", "Connection error. Make sure Flask is running and try again.");
        } finally {
            setIsVerifying(false);
        }
    };

    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: "#D8E9F0" }}>
            <View style={{ margin: 20 }}>
                <Pressable onPress={() => router.back()} style={styles.backBtn}>
                    <Ionicons name="arrow-back" size={28} color="black" />
                </Pressable>

                <Text style={{ fontSize: 18, marginTop: 20, textAlign: "center", fontWeight: "bold" }}>
                    Verify Your Email
                </Text>

                <Image
                    source={require("../assets/images/Mail.png")}
                    style={styles.logo}
                    resizeMode="contain"
                />

                <Text style={{ marginTop: 15, paddingHorizontal: 5, textAlign: "center", fontSize: 22 }}>
                    We&#39;ve sent a verification code to
                </Text>
                <Text style={{ marginTop: 2, textAlign: "center", fontSize: 18, fontWeight: "bold" }}>
                    {newEmail}
                </Text>

                {isSending && (
                    <Text style={{ textAlign: "center", color: "#004F7F", marginTop: 10, fontSize: 13 }}>
                        Sending code...
                    </Text>
                )}

                <Text style={styles.label}>Enter Your Code</Text>
                <View style={styles.otpContainer}>
                    {[0, 1, 2, 3, 4, 5].map((_, index) => (
                        <TextInput
                            key={index}
                            ref={(ref) => { inputs.current[index] = ref; }}
                            style={[styles.otpBox, otp[index] ? styles.otpBoxFilled : {}]}
                            keyboardType="number-pad"
                            maxLength={1}
                            value={otp[index]}
                            onChangeText={(text) => handleOtpChange(text, index)}
                            onKeyPress={({ nativeEvent }) => handleBackspace(nativeEvent.key, index)}
                        />
                    ))}
                </View>

                <TouchableOpacity
                    disabled={!isFormValid || isSending || isVerifying}
                    onPress={handleVerify}
                    style={[styles.verifyBtn, {
                        backgroundColor: isFormValid && !isSending && !isVerifying ? "#004F7F" : "#B0B0B0"
                    }]}
                >
                    <Text style={styles.verifyText}>
                        {isVerifying ? "Verifying..." : "Verify"}
                    </Text>
                </TouchableOpacity>

                <View style={{ marginTop: 25, alignItems: "center" }}>
                    {canResend ? (
                        <TouchableOpacity onPress={handleResend} disabled={isSending}>
                            <Text style={[styles.resendText, isSending && { color: "#B0B0B0" }]}>
                                {isSending ? "Sending..." : "Send code again"}
                            </Text>
                        </TouchableOpacity>
                    ) : (
                        <Text style={styles.timerText}>
                            Send code again in{" "}
                            <Text style={{ fontWeight: "bold" }}>
                                00:{timer < 10 ? `0${timer}` : timer}
                            </Text>
                        </Text>
                    )}
                </View>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    backBtn: {
        width: 42, height: 42, backgroundColor: "#D8E9F0",
        borderRadius: 15, borderColor: "#000", borderWidth: 0.5,
        justifyContent: "center", alignItems: "center", elevation: 3,
    },
    logo: { width: 100, height: 100, alignSelf: "center", marginTop: 20, marginLeft: 20 },
    otpContainer: {
        flexDirection: "row", justifyContent: "space-between",
        marginTop: 15, paddingHorizontal: 10, marginBottom: 80,
    },
    otpBox: {
        width: 45, height: 55, borderRadius: 10, backgroundColor: "#fff",
        textAlign: "center", fontSize: 22, borderWidth: 1, borderColor: "#ccc",
    },
    otpBoxFilled: {
        borderColor: "#004F7F", borderWidth: 2, backgroundColor: "#E8F4F8",
    },
    timerText:  { fontSize: 14, color: "#444" },
    resendText: { fontSize: 14, fontWeight: "bold", color: "#004F7F" },
    verifyBtn:  { marginTop: 25, padding: 14, borderRadius: 8, marginBottom: 5 },
    verifyText: { color: "#fff", textAlign: "center", fontWeight: "bold", fontSize: 16 },
    label:      { fontSize: 15, marginTop: 30, alignSelf: "flex-start", marginLeft: 15 },
});