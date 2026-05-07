import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import { EmailAuthProvider, reauthenticateWithCredential, updatePassword } from "firebase/auth";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator, Alert, ScrollView, StyleSheet, Text,
  TextInput, TouchableOpacity, View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { auth } from "../../Firebase/firebaseConfig";
import { FONT_FAMILY_MAP, useCustomize } from "../Customize/Customizecontext";
import { useTranslation } from "../Customize/translations";
import { useTheme } from "../ThemeContext";

export default function ChangePassword() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();
  const { t, isArabic } = useTranslation(settings.language);

  const customText = {
    fontSize: settings.fontSize,
    color: isDark ? "#FFFFFF" : settings.textColor,
    fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
  };

  const pageBg = isDark ? colors.background : settings.backgroundColor;

  const [currentPassword, setCurrentPassword]   = useState("");
  const [newPassword, setNewPassword]           = useState("");
  const [confirmPassword, setConfirmPassword]   = useState("");
  const [showCurrent, setShowCurrent]           = useState(false);
  const [showNew, setShowNew]                   = useState(false);
  const [showConfirm, setShowConfirm]           = useState(false);
  const [newPasswordError, setNewPasswordError] = useState("");
  const [confirmError, setConfirmError]         = useState("");
  const [isLoading, setIsLoading]               = useState(false);

  useEffect(() => {
    if (!newPassword) { setNewPasswordError(""); return; }
    if (newPassword.length < 8)           setNewPasswordError(t('passwordMin'));
    else if (!/[A-Z]/.test(newPassword))  setNewPasswordError(t('passwordUppercase'));
    else if (!/[a-z]/.test(newPassword))  setNewPasswordError(t('passwordLowercase'));
    else if (!/[0-9]/.test(newPassword))  setNewPasswordError(t('passwordNumber'));
    else if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(newPassword))
      setNewPasswordError(t('passwordSpecial'));
    else setNewPasswordError("");
  }, [newPassword, settings.language]);

  useEffect(() => {
    if (!confirmPassword) setConfirmError("");
    else if (confirmPassword !== newPassword) setConfirmError(t('passwordsNotMatch'));
    else setConfirmError("");
  }, [confirmPassword, newPassword, settings.language]);

  const isNewPasswordValid =
      newPassword.length >= 8 &&
      /[A-Z]/.test(newPassword) &&
      /[a-z]/.test(newPassword) &&
      /[0-9]/.test(newPassword) &&
      /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(newPassword);

  const isFormValid =
      !!currentPassword && isNewPasswordValid &&
      confirmPassword === newPassword && !confirmError;

  const strengthChecks = [
    { label: t('eightChars'),      pass: newPassword.length >= 8 },
    { label: t('uppercaseLetter'), pass: /[A-Z]/.test(newPassword) },
    { label: t('lowercaseLetter'), pass: /[a-z]/.test(newPassword) },
    { label: t('numberDigit'),     pass: /[0-9]/.test(newPassword) },
    { label: t('specialChar'),     pass: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(newPassword) },
  ];

  const handleChange = async () => {
    if (!isFormValid || isLoading) return;

    const user = auth.currentUser;
    if (!user) {
      Alert.alert('Error', 'No logged-in user found. Please log in again.');
      return;
    }

    setIsLoading(true);
    try {
      // Get the most up-to-date email — user.email can be stale after Admin SDK email change
      const saved = await AsyncStorage.getItem('signupDraft');
      const draft = saved ? JSON.parse(saved) : {};
      const email = draft.email || user.email;

      if (!email) {
        Alert.alert('Error', 'Could not determine your email. Please log in again.');
        return;
      }

      // Step 1 — Re-authenticate with current password using latest email
      console.log('🔐 Re-authenticating user with email:', email);
      const credential = EmailAuthProvider.credential(email, currentPassword);
      await reauthenticateWithCredential(user, credential);
      console.log('✅ Re-authentication successful');

      // Step 2 — Update password in Firebase Auth
      console.log('🔑 Updating password...');
      await updatePassword(user, newPassword);
      console.log('✅ Password updated successfully');

     router.push(
       "/Settingsoptions/ConfirmCP",
     );
    } catch (err: any) {
      console.log('❌ Change password error:', err.code, err.message);

      let message = 'Failed to change password. Please try again.';
      if (err.code === 'auth/wrong-password' || err.code === 'auth/invalid-credential') {
        message = 'Current password is incorrect. Please try again.';
      } else if (err.code === 'auth/too-many-requests') {
        message = 'Too many failed attempts. Please wait a moment and try again.';
      } else if (err.code === 'auth/requires-recent-login') {
        message = 'Session expired. Please log out and log back in, then try again.';
      } else if (err.code === 'auth/weak-password') {
        message = 'New password is too weak. Please choose a stronger password.';
      } else if (err.code === 'auth/network-request-failed') {
        message = 'Network error. Please check your connection and try again.';
      }

      Alert.alert('Error', message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
        <View style={[styles.header, { backgroundColor: colors.card }]}>
          <TouchableOpacity
              style={[styles.backBtn, { borderColor: colors.border }]}
              onPress={() => router.back()}
              disabled={isLoading}
          >
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </TouchableOpacity>
          <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.headerTitle, customText]}>{t('changePassword')}</Text>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>

          {/* Current Password */}
          <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>
            {t('currentPassword')}
          </Text>
          <View style={styles.passwordWrapper}>
            <TextInput
                placeholder={t('enterCurrentPassword')}
                placeholderTextColor={colors.subText}
                secureTextEntry={!showCurrent}
                value={currentPassword}
                onChangeText={setCurrentPassword}
                textAlign={isArabic ? 'right' : 'left'}
                editable={!isLoading}
                style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.passwordInput, customText, {
                  backgroundColor: colors.card,
                  borderColor: colors.border,
                  paddingRight: isArabic ? 13 : 45,
                  paddingLeft:  isArabic ? 45 : 13,
                }]}
            />
            <TouchableOpacity
                onPress={() => setShowCurrent(!showCurrent)}
                style={[styles.eyeIcon, isArabic ? { left: 13, right: undefined } : { right: 13 }]}
            >
              <Ionicons name={showCurrent ? "eye" : "eye-off"} size={20} color={colors.subText} />
            </TouchableOpacity>
          </View>

          {/* New Password */}
          <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>
            {t('newPassword')}
          </Text>
          <View style={styles.passwordWrapper}>
            <TextInput
                placeholder={t('enterNewPassword')}
                placeholderTextColor={colors.subText}
                secureTextEntry={!showNew}
                value={newPassword}
                onChangeText={setNewPassword}
                textAlign={isArabic ? 'right' : 'left'}
                editable={!isLoading}
                style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.passwordInput, customText, {
                  backgroundColor: colors.card,
                  borderColor: colors.border,
                  paddingRight: isArabic ? 13 : 45,
                  paddingLeft:  isArabic ? 45 : 13,
                }]}
            />
            <TouchableOpacity
                onPress={() => setShowNew(!showNew)}
                style={[styles.eyeIcon, isArabic ? { left: 13, right: undefined } : { right: 13 }]}
            >
              <Ionicons name={showNew ? "eye" : "eye-off"} size={20} color={colors.subText} />
            </TouchableOpacity>
          </View>
          {!!newPasswordError && (
              <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.errorText, customText, { color: 'red', textAlign: isArabic ? 'right' : 'left' }]}>
                {newPasswordError}
              </Text>
          )}

          {/* Strength indicators */}
          {newPassword.length > 0 && (
              <View style={styles.strengthContainer}>
                {strengthChecks.map((item) => (
                    <View key={item.label} style={[styles.strengthRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                      <Ionicons
                          name={item.pass ? "checkmark-circle" : "ellipse-outline"}
                          size={16}
                          color={item.pass ? "#22C55E" : colors.subText}
                      />
                      <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.strengthText, customText, { color: item.pass ? "#22C55E" : settings.textColor }]}>
                        {item.label}
                      </Text>
                    </View>
                ))}
              </View>
          )}

          {/* Confirm Password */}
          <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>
            {t('confirmNewPassword')}
          </Text>
          <View style={styles.passwordWrapper}>
            <TextInput
                placeholder={t('confirmYourPassword')}
                placeholderTextColor={colors.subText}
                secureTextEntry={!showConfirm}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                textAlign={isArabic ? 'right' : 'left'}
                editable={!isLoading}
                style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] },
                  styles.passwordInput, customText,
                  {
                    backgroundColor: colors.card,
                    borderColor: confirmPassword && newPassword !== confirmPassword ? 'red' : colors.border,
                    paddingRight: isArabic ? 13 : 45,
                    paddingLeft:  isArabic ? 45 : 13,
                  },
                ]}
            />
            <TouchableOpacity
                onPress={() => setShowConfirm(!showConfirm)}
                style={[styles.eyeIcon, isArabic ? { left: 13, right: undefined } : { right: 13 }]}
            >
              <Ionicons name={showConfirm ? "eye" : "eye-off"} size={20} color={colors.subText} />
            </TouchableOpacity>
          </View>
          {!!confirmError && (
              <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.errorText, customText, { color: 'red', textAlign: isArabic ? 'right' : 'left' }]}>
                {confirmError}
              </Text>
          )}

          {/* Change Button */}
          <TouchableOpacity
              disabled={!isFormValid || isLoading}
              onPress={handleChange}
              style={[
                styles.saveBtn,
                { backgroundColor: isFormValid && !isLoading ? colors.primary : isDark ? "#334155" : "#aeaeae" },
              ]}
          >
            {isLoading ? (
                <ActivityIndicator color="#fff" size="small" />
            ) : (
                <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.saveBtnText]}>
                  {t('change')}
                </Text>
            )}
          </TouchableOpacity>

        </ScrollView>
      </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:         { flex: 1 },
  header:            { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backBtn:           { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: "center", justifyContent: "center" },
  headerTitle:       { fontSize: 22},
  content:           { paddingHorizontal: 20, paddingTop: 10, paddingBottom: 40 },
  label:             { fontSize: 15, fontWeight: "600", marginTop: 22, marginBottom: 8 },
  passwordWrapper:   { position: "relative" },
  passwordInput:     { borderWidth: 1, borderRadius: 10, padding: 13, fontSize: 15 },
  eyeIcon:           { position: "absolute", top: "50%", transform: [{ translateY: -11 }] },
  errorText:         { marginTop: 6, fontSize: 13 },
  strengthContainer: { marginTop: 10, gap: 6 },
  strengthRow:       { alignItems: "center", gap: 6 },
  strengthText:      { fontSize: 13 },
  saveBtn:           { marginTop: 40, padding: 15, borderRadius: 12, alignItems: "center" },
  saveBtnText:       { color: "#fff", fontSize: 16 },
});