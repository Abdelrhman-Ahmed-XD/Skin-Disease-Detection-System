import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity,
    TextInput, Alert, ActivityIndicator
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';

const FLASK_URL = "http://192.168.100.2:5000";
const OTP_LENGTH = 6;

function generateOTP(): string {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

export default function Verifypassword() {
    const [otp, setOtp] = useState<string[]>(Array(OTP_LENGTH).fill(''));
    const [timer, setTimer] = useState(60);
    const [isVerifying, setIsVerifying] = useState(false);
    const [isResending, setIsResending] = useState(false);
    const inputRefs = useRef<Array<TextInput | null>>(Array(OTP_LENGTH).fill(null));

    // Countdown timer
    useEffect(() => {
        if (timer <= 0) return;
        const interval = setInterval(() => setTimer(t => t - 1), 1000);
        return () => clearInterval(interval);
    }, [timer]);

    const handleOtpChange = (value: string, index: number) => {
        if (!/^\d*$/.test(value)) return;
        const newOtp = [...otp];
        newOtp[index] = value.slice(-1);
        setOtp(newOtp);
        if (value && index < OTP_LENGTH - 1) {
            inputRefs.current[index + 1]?.focus();
        }
    };

    const handleKeyPress = (e: any, index: number) => {
        if (e.nativeEvent.key === 'Backspace' && !otp[index] && index > 0) {
            inputRefs.current[index - 1]?.focus();
        }
    };

    const handleVerify = async () => {
        const enteredOtp = otp.join('');
        if (enteredOtp.length < OTP_LENGTH) {
            Alert.alert('Error', 'Please enter the complete 6-digit code.');
            return;
        }

        setIsVerifying(true);
        try {
            const savedOtp = await AsyncStorage.getItem('forgetPasswordOTP');
            const expiryStr = await AsyncStorage.getItem('forgetPasswordOTPExpiry');
            const expiry = expiryStr ? parseInt(expiryStr) : 0;

            if (!savedOtp) {
                Alert.alert('Error', 'No OTP found. Please request a new code.');
                setIsVerifying(false);
                return;
            }

            if (Date.now() > expiry) {
                Alert.alert('Expired', 'Your code has expired. Please request a new one.');
                setIsVerifying(false);
                return;
            }

            if (enteredOtp !== savedOtp) {
                Alert.alert('Invalid Code', 'The code you entered is incorrect. Please try again.');
                setIsVerifying(false);
                return;
            }

            // OTP is correct - clean up and navigate to reset password screen
            await AsyncStorage.removeItem('forgetPasswordOTP');
            await AsyncStorage.removeItem('forgetPasswordOTPExpiry');
            console.log('✅ OTP verified, navigating to Resetpassword...');
            router.push('/Resetpassword');

        } catch (error: any) {
            Alert.alert('Error', 'Verification failed. Please try again.');
        } finally {
            setIsVerifying(false);
        }
    };

    const handleResend = async () => {
        setIsResending(true);
        try {
            const email = await AsyncStorage.getItem('forgetPasswordEmail');
            if (!email) {
                Alert.alert('Error', 'Email not found. Please go back and try again.');
                return;
            }

            const newOtp = generateOTP();
            console.log('🔐 Resending OTP:', newOtp);

            const response = await fetch(`${FLASK_URL}/api/send-password-reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email,
                    name: email.split('@')[0],
                    otp_code: newOtp,
                }),
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                Alert.alert('Error', 'Failed to resend code. Please try again.');
                return;
            }

            await AsyncStorage.setItem('forgetPasswordOTP', newOtp);
            await AsyncStorage.setItem('forgetPasswordOTPExpiry', (Date.now() + 10 * 60 * 1000).toString());

            setOtp(Array(OTP_LENGTH).fill(''));
            setTimer(60);
            inputRefs.current[0]?.focus();
            Alert.alert('Sent!', 'A new verification code has been sent to your email.');

        } catch (error) {
            Alert.alert('Error', 'Failed to resend code. Please try again.');
        } finally {
            setIsResending(false);
        }
    };

    const isComplete = otp.every(d => d !== '');

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                <Ionicons name="arrow-back" size={28} color="black" />
            </TouchableOpacity>

            <View style={styles.content}>
                <View style={styles.iconContainer}>
                    <Ionicons name="lock-closed" size={48} color="#004F7F" />
                </View>

                <Text style={styles.title}>Check your email</Text>
                <Text style={styles.subtitle}>
                    We sent a 6-digit password reset code to your email address.
                </Text>

                {/* OTP Input Boxes */}
                <View style={styles.otpContainer}>
                    {otp.map((digit, index) => (
                        <TextInput
                            key={index}
                            ref={ref => { inputRefs.current[index] = ref; }}
                            style={[styles.otpBox, digit ? styles.otpBoxFilled : null]}
                            value={digit}
                            onChangeText={val => handleOtpChange(val, index)}
                            onKeyPress={e => handleKeyPress(e, index)}
                            keyboardType="number-pad"
                            maxLength={1}
                            selectTextOnFocus
                        />
                    ))}
                </View>

                {/* Verify Button */}
                <TouchableOpacity
                    disabled={!isComplete || isVerifying}
                    onPress={handleVerify}
                    style={[styles.verifyBtn, { backgroundColor: isComplete && !isVerifying ? '#004F7F' : '#B0B0B0' }]}
                >
                    {isVerifying ? (
                        <ActivityIndicator color="#fff" size="small" />
                    ) : (
                        <Text style={styles.verifyText}>Verify Code</Text>
                    )}
                </TouchableOpacity>

                {/* Resend */}
                <View style={styles.resendContainer}>
                    {timer > 0 ? (
                        <Text style={styles.timerText}>
                            Resend code in{' '}
                            <Text style={{ fontWeight: 'bold', color: '#004F7F' }}>
                                00:{timer < 10 ? `0${timer}` : timer}
                            </Text>
                        </Text>
                    ) : (
                        <TouchableOpacity onPress={handleResend} disabled={isResending}>
                            {isResending ? (
                                <ActivityIndicator color="#004F7F" size="small" />
                            ) : (
                                <Text style={styles.resendText}>Resend Code</Text>
                            )}
                        </TouchableOpacity>
                    )}
                </View>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#D8E9F0',
        paddingHorizontal: 20,
    },
    backBtn: {
        width: 42,
        height: 42,
        backgroundColor: '#D8E9F0',
        borderRadius: 15,
        borderColor: '#000',
        borderWidth: 0.5,
        justifyContent: 'center',
        alignItems: 'center',
        elevation: 3,
        marginTop: 10,
    },
    content: {
        flex: 1,
        alignItems: 'center',
        paddingTop: 40,
    },
    iconContainer: {
        width: 90,
        height: 90,
        borderRadius: 45,
        backgroundColor: '#E8F4F8',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
        borderWidth: 2,
        borderColor: '#C5E3ED',
    },
    title: {
        fontSize: 26,
        fontWeight: 'bold',
        color: '#1F2937',
        marginBottom: 10,
        textAlign: 'center',
    },
    subtitle: {
        fontSize: 15,
        color: '#6B7280',
        textAlign: 'center',
        paddingHorizontal: 20,
        lineHeight: 22,
        marginBottom: 36,
    },
    otpContainer: {
        flexDirection: 'row',
        gap: 10,
        marginBottom: 32,
    },
    otpBox: {
        width: 48,
        height: 56,
        borderRadius: 10,
        borderWidth: 1.5,
        borderColor: '#D1D5DB',
        backgroundColor: '#fff',
        textAlign: 'center',
        fontSize: 22,
        fontWeight: 'bold',
        color: '#1F2937',
    },
    otpBoxFilled: {
        borderColor: '#004F7F',
        backgroundColor: '#E8F4F8',
    },
    verifyBtn: {
        width: '100%',
        paddingVertical: 15,
        borderRadius: 10,
        alignItems: 'center',
        marginBottom: 20,
    },
    verifyText: {
        color: '#fff',
        fontWeight: '700',
        fontSize: 16,
    },
    resendContainer: {
        alignItems: 'center',
        marginTop: 8,
    },
    timerText: {
        fontSize: 14,
        color: '#6B7280',
    },
    resendText: {
        fontSize: 15,
        color: '#004F7F',
        fontWeight: '700',
        textDecorationLine: 'underline',
    },
});