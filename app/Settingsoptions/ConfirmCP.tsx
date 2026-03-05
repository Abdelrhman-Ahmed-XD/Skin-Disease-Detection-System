import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import React from 'react'
import { Image, Pressable, StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useCustomize } from '../Customize/Customizecontext'
import { useTranslation } from '../Customize/translations'
import { useTheme } from '../ThemeContext'

export default function ConfirmCP() {
  const { colors, isDark } = useTheme()
  const { settings } = useCustomize()
  const { t, isArabic } = useTranslation(settings.language)

  const customText = {
    fontSize:   settings.fontSize,
    color:      settings.textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  }

  // ✅ background من settings في light mode
  const pageBg = isDark ? colors.background : settings.backgroundColor

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: pageBg }}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />
      <View style={{ margin: 20 }}>
        <Pressable onPress={() => router.back()} style={[styles.backBtn, {
          backgroundColor: colors.card,
          borderColor: colors.border,
        }]}>
          <Ionicons name={isArabic ? "chevron-back" : "chevron-back"} size={28} color={colors.text} />
        </Pressable>

        <View style={{ margin: 20, justifyContent: "center", alignItems: "center" }}>
          <Image
            source={require("../../assets/images/checkmark.png")}
            style={{
              width: 250,
              height: 250,
              marginBottom: 5,
              marginTop: 30,
              tintColor: isDark ? "#22C55E" : undefined,
            }}
          />
          <Text style={[customText, { fontSize: settings.fontSize > 20 ? settings.fontSize : 24, fontWeight: "bold", textAlign: "center" }]}>
            {t('passwordChangedSuccess')}
          </Text>
          <TouchableOpacity onPress={() => router.push("/Settingsoptions/Editprofile")} style={{ width: "100%" }}>
            <Text style={[styles.Continue, customText, { backgroundColor: colors.primary, color: '#fff' }]}>
              {t('continue')}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  backBtn: {
    width: 42,
    height: 42,
    borderRadius: 15,
    borderWidth: 0.5,
    justifyContent: "center",
    alignItems: "center",
    elevation: 3,
  },
  Continue: {
    marginTop: "50%",
    padding: 14,
    borderRadius: 8,
    marginBottom: 5,
    fontSize: 16,
    fontWeight: "bold",
    textAlign: "center",
  },
})