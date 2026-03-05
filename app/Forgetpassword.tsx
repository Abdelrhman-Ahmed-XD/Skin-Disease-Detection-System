import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import React, { useState, useEffect } from 'react'
import { View, Pressable, StyleSheet, Text, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import AsyncStorage from "@react-native-async-storage/async-storage";
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../Firebase/firebaseConfig';

const FLASK_URL = "http://192.168.100.2:5000";

function generateOTP(): string {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

export default function Forgetpassword() {
    const [email, setEmail] = useState('')
    const [emailError, setEmailError] = useState('')
    const [isLoading, setIsLoading] = useState(false)

    useEffect(() => {
        const regex = /^[a-zA-Z0-9][a-zA-Z0-9._]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        if (!email) setEmailError('')
        else if (!regex.test(email)) setEmailError('Please enter a valid email address')
        else setEmailError('')
    }, [email])

    const isFormValid = email && !emailError

    const handleSendCode = async () => {
        if (!isFormValid) return;
        setIsLoading(true);

        try {
            // Step 1: Check if email exists in Firebase
            // We try signing in with a dummy password - if error is 'wrong-password',
            // the account EXISTS. If error is 'user-not-found', the account does NOT exist.
            console.log('🔍 Checking if email exists in Firebase:', email);
            let emailExists = false;
            try {
                await signInWithEmailAndPassword(auth, email, '___DUMMY_PASSWORD_CHECK___');
            } catch (firebaseError: any) {
                const code = firebaseError.code;
                console.log('🔍 Firebase check result:', code);
                if (
                    code === 'auth/wrong-password' ||
                    code === 'auth/invalid-credential' ||
                    code === 'auth/too-many-requests'
                ) {
                    // These errors mean the account EXISTS (password was just wrong)
                    emailExists = true;
                } else if (code === 'auth/user-not-found') {
                    // Account does NOT exist
                    emailExists = false;
                } else {
                    // Unknown error - allow through to avoid blocking legitimate users
                    emailExists = true;
                }
            }

            if (!emailExists) {
                Alert.alert(
                    'Email Not Found',
                    'No account found with this email address. Please check your email or sign up.'
                );
                setIsLoading(false);
                return;
            }

            console.log('✅ Email found in Firebase, sending OTP...');

            // Step 2: Generate OTP
            const otp = generateOTP();
            console.log('🔐 Generated OTP:', otp);

            // Step 3: Send OTP via Flask
            const response = await fetch(`${FLASK_URL}/api/send-password-reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: email,
                    name: email.split('@')[0],
                    otp_code: otp,
                }),
            });

            const result = await response.json();
            console.log('📬 Flask response:', result);

            if (!response.ok || !result.success) {
                Alert.alert('Error', result.message || 'Failed to send code. Please try again.');
                setIsLoading(false);
                return;
            }

            // Step 4: Save email and OTP to AsyncStorage
            await AsyncStorage.setItem('forgetPasswordEmail', email);
            await AsyncStorage.setItem('forgetPasswordOTP', otp);
            await AsyncStorage.setItem('forgetPasswordOTPExpiry', (Date.now() + 10 * 60 * 1000).toString());

            console.log('✅ OTP saved, navigating to Verifypassword...');
            router.push('/Verifypassword');

        } catch (error: any) {
            console.log('❌ Error:', error);
            Alert.alert('Error', 'Failed to send code. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: "#D8E9F0" }}>
            <View style={{ margin: 20 }}>
                <Pressable onPress={() => router.back()} style={styles.backBtn}>
                    <Ionicons name="arrow-back" size={28} color="black" />
                </Pressable>

                <Text style={styles.title}>Forgot password?</Text>

                <Text style={{ margin: 20, marginTop: 10, fontSize: 16, textAlign: "center" }}>
                    Don't worry! It happens. Please enter the
                    <Text style={{ fontWeight: "bold" }}> email address </Text>
                    associated with your account.
                </Text>

                <Text style={styles.label}>Email Address</Text>
                <TextInput
                    placeholder="Example@gmail.com"
                    style={styles.emailInput}
                    keyboardType="email-address"
                    autoCapitalize="none"
                    value={email}
                    onChangeText={setEmail}
                    editable={!isLoading}
                />

                {!!emailError && <Text style={styles.errorText}>{emailError}</Text>}

                <TouchableOpacity
                    disabled={!isFormValid || isLoading}
                    onPress={handleSendCode}
                    style={[
                        styles.VerifyBtn,
                        { backgroundColor: isFormValid && !isLoading ? "#004F7F" : "#B0B0B0" },
                    ]}
                >
                    {isLoading ? (
                        <ActivityIndicator color="#fff" size="small" />
                    ) : (
                        <Text style={styles.VerifyText}>Send Code</Text>
                    )}
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    )
}

const styles = StyleSheet.create({
    backBtn: {
        width: 42,
        height: 42,
        backgroundColor: "#D8E9F0",
        borderRadius: 15,
        borderColor: "#000",
        borderWidth: 0.5,
        justifyContent: "center",
        alignItems: "center",
        elevation: 3,
    },
    title: {
        fontSize: 32,
        fontWeight: "bold",
        textAlign: "center",
        marginTop: 20,
        marginBottom: 20,
    },
    label: {
        fontSize: 15,
        marginTop: 10,
        alignSelf: "flex-start",
        marginBottom: 5,
        fontWeight: '600',
        color: '#374151',
    },
    emailInput: {
        borderWidth: 1,
        borderColor: "#000",
        backgroundColor: "#fff",
        borderRadius: 8,
        padding: 12,
    },
    errorText: {
        color: "red",
        marginTop: 5,
    },
    VerifyBtn: {
        paddingHorizontal: 16,
        paddingVertical: 14,
        borderRadius: 8,
        marginTop: 20,
        alignItems: "center",
    },
    VerifyText: {
        color: "#fff",
        fontWeight: "600",
        fontSize: 15,
    },
})