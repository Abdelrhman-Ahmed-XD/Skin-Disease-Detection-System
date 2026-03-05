import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    TextInput,
    KeyboardAvoidingView,
    ScrollView,
    Platform,
    Alert,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';

const FLASK_URL = "http://192.168.100.2:5000";
const FORGET_EMAIL_KEY = 'forgetPasswordEmail';

export default function Resetpassword() {
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [passwordError, setPasswordError] = useState('');
    const [confirmError, setConfirmError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (!password) { setPasswordError(''); return; }
        if (password.length < 8) setPasswordError('Password must be at least 8 characters');
        else if (!/[A-Z]/.test(password)) setPasswordError('Must contain at least one uppercase letter (A-Z)');
        else if (!/[a-z]/.test(password)) setPasswordError('Must contain at least one lowercase letter (a-z)');
        else if (!/[0-9]/.test(password)) setPasswordError('Must contain at least one number (0-9)');
        else if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) setPasswordError('Must contain at least one special character (@, #, !, ...)');
        else setPasswordError('');
    }, [password]);

    useEffect(() => {
        if (!confirmPassword) setConfirmError('');
        else if (confirmPassword !== password) setConfirmError('Passwords do not match');
        else setConfirmError('');
    }, [confirmPassword, password]);

    const isFormValid =
        password.length >= 8 &&
        /[A-Z]/.test(password) &&
        /[a-z]/.test(password) &&
        /[0-9]/.test(password) &&
        /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password) &&
        confirmPassword === password &&
        !confirmError;

    const handleReset = async () => {
        if (!isFormValid) return;
        setIsLoading(true);
        try {
            const email = await AsyncStorage.getItem(FORGET_EMAIL_KEY);
            if (!email) {
                Alert.alert('Error', 'Email not found. Please try again from the beginning.');
                setIsLoading(false);
                return;
            }

            console.log(`🔐 Updating password for: ${email}`);

            // Call Flask backend which uses Firebase Admin to update password
            const response = await fetch(`${FLASK_URL}/api/update-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, new_password: password }),
            });

            const result = await response.json();
            console.log('📬 Flask response:', result);

            if (!response.ok || !result.success) {
                Alert.alert('Error', result.message || 'Failed to reset password. Please try again.');
                setIsLoading(false);
                return;
            }

            // Clean up AsyncStorage
            await AsyncStorage.removeItem(FORGET_EMAIL_KEY);
            await AsyncStorage.removeItem('pendingPasswordUpdate');
            await AsyncStorage.removeItem('pendingPasswordEmail');

            console.log('✅ Password updated successfully!');
            Alert.alert(
                'Password Reset! 🎉',
                'Your password has been updated successfully. Please log in with your new password.',
                [{ text: 'OK', onPress: () => router.replace('/Login1') }]
            );
        } catch (err: any) {
            console.log('❌ Error:', err);
            Alert.alert('Error', 'Failed to reset password. Make sure Flask is running and try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            <KeyboardAvoidingView
                style={{ flex: 1 }}
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                >
                    {/* Back Button */}
                    <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                        <Ionicons name="arrow-back" size={28} color="black" />
                    </TouchableOpacity>

                    {/* Icon */}
                    <View style={styles.iconContainer}>
                        <Ionicons name="lock-closed" size={48} color="#004F7F" />
                    </View>

                    {/* Title */}
                    <Text style={styles.title}>Reset Password</Text>
                    <Text style={styles.subtitle}>Please type something you&#39;ll remember</Text>

                    {/* Password Field */}
                    <Text style={styles.label}>New Password</Text>
                    <View style={styles.inputWrapper}>
                        <TextInput
                            placeholder="Create your password"
                            placeholderTextColor="#9CA3AF"
                            secureTextEntry={!showPassword}
                            value={password}
                            onChangeText={setPassword}
                            style={styles.input}
                            editable={!isLoading}
                        />
                        <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={styles.eyeIcon}>
                            <Ionicons name={showPassword ? 'eye' : 'eye-off'} size={22} color="#6B7280" />
                        </TouchableOpacity>
                    </View>
                    {!!passwordError && <Text style={styles.errorText}>{passwordError}</Text>}

                    {/* Password strength indicators */}
                    {password.length > 0 && (
                        <View style={styles.strengthContainer}>
                            {[
                                { label: '8+ characters',                   pass: password.length >= 8 },
                                { label: 'Uppercase letter (A-Z)',          pass: /[A-Z]/.test(password) },
                                { label: 'Lowercase letter (a-z)',          pass: /[a-z]/.test(password) },
                                { label: 'Number (0-9)',                    pass: /[0-9]/.test(password) },
                                { label: 'Special character (@, #, !...)',  pass: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password) },
                            ].map((item) => (
                                <View key={item.label} style={styles.strengthRow}>
                                    <Ionicons
                                        name={item.pass ? 'checkmark-circle' : 'ellipse-outline'}
                                        size={16}
                                        color={item.pass ? '#22C55E' : '#9CA3AF'}
                                    />
                                    <Text style={[styles.strengthText, { color: item.pass ? '#22C55E' : '#9CA3AF' }]}>
                                        {item.label}
                                    </Text>
                                </View>
                            ))}
                        </View>
                    )}

                    {/* Confirm Password Field */}
                    <Text style={[styles.label, { marginTop: 24 }]}>Confirm Password</Text>
                    <View style={styles.inputWrapper}>
                        <TextInput
                            placeholder="Re-enter your password"
                            placeholderTextColor="#9CA3AF"
                            secureTextEntry={!showConfirmPassword}
                            value={confirmPassword}
                            onChangeText={setConfirmPassword}
                            style={styles.input}
                            editable={!isLoading}
                        />
                        <TouchableOpacity onPress={() => setShowConfirmPassword(!showConfirmPassword)} style={styles.eyeIcon}>
                            <Ionicons name={showConfirmPassword ? 'eye' : 'eye-off'} size={22} color="#6B7280" />
                        </TouchableOpacity>
                    </View>
                    {!!confirmError && <Text style={styles.errorText}>{confirmError}</Text>}

                    {/* Reset Button */}
                    <TouchableOpacity
                        disabled={!isFormValid || isLoading}
                        onPress={handleReset}
                        style={[styles.resetBtn, { backgroundColor: isFormValid && !isLoading ? '#004F7F' : '#B0B0B0' }]}
                        activeOpacity={0.85}
                    >
                        {isLoading ? (
                            <ActivityIndicator color="#fff" size="small" />
                        ) : (
                            <Text style={styles.resetText}>Reset Password</Text>
                        )}
                    </TouchableOpacity>

                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#D8E9F0',
    },
    scrollContent: {
        paddingHorizontal: 20,
        paddingTop: 16,
        paddingBottom: 40,
        flexGrow: 1,
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
    },
    iconContainer: {
        width: 90,
        height: 90,
        borderRadius: 45,
        backgroundColor: '#E8F4F8',
        justifyContent: 'center',
        alignItems: 'center',
        alignSelf: 'center',
        marginTop: 24,
        marginBottom: 8,
        borderWidth: 2,
        borderColor: '#C5E3ED',
    },
    title: {
        fontSize: 25,
        marginTop: 16,
        textAlign: 'center',
        fontWeight: 'bold',
        color: '#1F2937',
    },
    subtitle: {
        marginTop: 10,
        textAlign: 'center',
        fontSize: 16,
        color: '#6B7280',
        paddingHorizontal: 10,
    },
    label: {
        fontSize: 15,
        fontWeight: '600',
        color: '#374151',
        marginTop: 32,
        marginBottom: 8,
    },
    inputWrapper: {
        position: 'relative',
        justifyContent: 'center',
    },
    input: {
        borderWidth: 1,
        borderColor: '#D1D5DB',
        backgroundColor: '#fff',
        borderRadius: 10,
        padding: 13,
        paddingRight: 48,
        fontSize: 15,
        color: '#1F2937',
    },
    eyeIcon: {
        position: 'absolute',
        right: 14,
        height: '100%',
        justifyContent: 'center',
    },
    errorText: {
        color: '#EF4444',
        fontSize: 13,
        marginTop: 6,
    },
    resetBtn: {
        padding: 15,
        borderRadius: 10,
        marginTop: 40,
        justifyContent: 'center',
        alignItems: 'center',
    },
    resetText: {
        color: '#fff',
        fontWeight: '700',
        fontSize: 16,
    },
    strengthContainer: {
        marginTop: 10,
        gap: 6,
    },
    strengthRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
    },
    strengthText: {
        fontSize: 13,
    },
});