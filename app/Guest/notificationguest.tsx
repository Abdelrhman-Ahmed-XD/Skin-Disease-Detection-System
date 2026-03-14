import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React from 'react';
import {
  StatusBar, StyleSheet, Text,
  TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';

export default function NotificationsGuestPage() {
  const { colors, isDark } = useTheme();
  const { settings } = useCustomize();

  // ── أبيض في الدارك مود، أسود في الليت مود ─────────────────
  const textColor = isDark ? '#FFFFFF' : (settings.textColor || '#1F2937');

  const customText = {
    fontSize:   settings.fontSize,
    color:      textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  };

  const pageBg      = isDark ? colors.background : settings.backgroundColor;

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity
          style={[styles.backButton, { borderColor: colors.border }]}
          onPress={() => router.back()}
        >
          <Ionicons
            name={"chevron-back"}
            size={24}
            color={textColor}
          />
        </TouchableOpacity>
        <View style={styles.headerTitleRow}>
          <Text style={[styles.headerTitle, customText]}>
            {("notifications")}
          </Text>
        </View>
        <View style={{ width: 40 }} />
      </View>

      {/* Content */}
      <View style={styles.guestContainer}>
        <View
          style={[
            styles.iconWrap,
            { backgroundColor: isDark ? "#fff" : "#004F7F" },
          ]}
        >
          <Ionicons
            name="lock-closed-outline"
            size={48}
            style={{ color: isDark ? "#004F7F" : "#fff" }}
          />
        </View>

        <Text style={[styles.guestTitle, customText]}>
          Login Required
        </Text>

        <Text
          style={[
            styles.guestSubtitle,
            {
              color: isDark ? "#AAAAAA" : colors.subText,
              fontFamily: customText.fontFamily,
              fontSize: customText.fontSize,
            },
          ]}
        >
          You need to sign up or log in to access notifications and use the full app.
        </Text>

        <TouchableOpacity
          style={[
            styles.signUpBtn,
            { backgroundColor: "#004F7F" },
          ]}
          onPress={() => router.push("/SignUp")}
        >
          <Ionicons
            name="person-add-outline"
            size={18}
            style={{ color: "#fff"}}
          />
          <Text
            style={[
              styles.signUpBtnText,
              { fontFamily: customText.fontFamily },
            ]}
          >
            Sign Up
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.loginBtn,
            { borderColor: isDark ? "#fff" : "#004F7F" },
          ]}
          onPress={() => router.push("/Login1")}
        >
          <Ionicons
            name="log-in-outline"
            size={28}
            style={{ color: isDark ? "#fff" : "#004F7F" }}
          />
          <Text
            style={[
              styles.loginBtnText,
              {
                color: isDark ? "#fff" : "#004F7F",
                fontFamily: customText.fontFamily,
              },
            ]}
          >
            Log In
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 15,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
    margin: 15,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 12,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  headerTitleRow: { flexDirection: "row", alignItems: "center" },
  headerTitle: { fontSize: 22, fontWeight: "bold" },
  guestContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 36,
  },
  iconWrap: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 24,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
  },
  guestTitle: {
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 12,
    textAlign: "center",
  },
  guestSubtitle: {
    fontSize: 14,
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 32,
  },
  signUpBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 40,
    marginBottom: 12,
    width: "100%",
    justifyContent: "center",
  },
  signUpBtnText: { color: "#FFFFFF", fontSize: 16, fontWeight: "700" },
  loginBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 14,
    paddingVertical: 13,
    paddingHorizontal: 40,
    borderWidth: 2,
    width: "100%",
    justifyContent: "center",
  },
  loginBtnText: { fontSize: 16, fontWeight: "700", color: "#FFFFFF" }
});