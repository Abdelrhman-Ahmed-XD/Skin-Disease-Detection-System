import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Animated, Image, StatusBar,
  StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTheme } from '../ThemeContext';

const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

export default function GuestHistoryPage() {
  const router = useRouter();
  const { isDark, colors } = useTheme();
  const { settings } = useCustomize();

  // ── أبيض في الدارك مود، أسود في الليت مود ─────────────────
  const textColor = isDark ? '#FFFFFF' : (settings.textColor || '#1F2937');

  const customText = {
    fontSize:   settings.fontSize,
    color:      textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  };

  const pageBg      = isDark ? colors.background : settings.backgroundColor;
  const [activeTab, setActiveTab] = useState('History');

  useFocusEffect(React.useCallback(() => { setActiveTab('History'); }, []));

  const [showModal, setShowModal] = useState(false);
  const modalFade  = useRef(new Animated.Value(0)).current;
  const modalScale = useRef(new Animated.Value(0.85)).current;

  const openModal = () => {
    setShowModal(true);
    modalFade.setValue(0); modalScale.setValue(0.85);
    Animated.parallel([
      Animated.timing(modalFade,  { toValue: 1, duration: 220, useNativeDriver: true }),
      Animated.spring(modalScale, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
    ]).start();
  };

  const closeModal = () => {
    Animated.timing(modalFade, { toValue: 0, duration: 180, useNativeDriver: true }).start(() => setShowModal(false));
  };

  const bottomTabs = [
    { name: 'Home',     iconImg: Icons.home     },
    { name: 'Reports',  iconImg: Icons.reports  },
    { name: 'History',  iconImg: Icons.history  },
    { name: 'Settings', iconImg: Icons.settings },
  ];

  const handleTabPress = (tabName: string) => {
    setActiveTab(tabName);
    if (tabName === 'Camera') { openModal(); return; }
    switch (tabName) {
      case 'Home':     router.push('/Guest/Guest');          break;
      case 'Reports':  router.push('/Guest/reportguest');    break;
      case 'Settings': router.push('/Guest/settingsguest');  break;
    }
  };

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: pageBg }]}
      edges={["top"]}
    >
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity
          style={[styles.backButton, { borderColor: colors.border }]}
          onPress={() => router.back()}
        >
          <Ionicons name="chevron-back" size={24} color={textColor} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, customText]}>{("history")}</Text>
        <View style={{ width: 40 }} />
      </View>

      <View style={styles.lockContainer}>
        <View
          style={[
            styles.iconWrap,
            { backgroundColor: isDark ? "#fff" : "#004F7F" },
          ]}
        >
          <Ionicons
            name="time-outline"
            size={48}
            style={{ color: isDark ? "#004F7F" : "#fff" }}
          />
        </View>
        <Text style={[styles.lockTitle, customText]}>
          History Unavailable
        </Text>
        <Text
          style={[
            styles.lockSubtitle,
            {
              color: isDark ? "#AAAAAA" : colors.subText,
              fontFamily: customText.fontFamily,
            },
          ]}
        >
          You need to log in or create an account to view your past scan history.
        </Text>
        <TouchableOpacity
          style={[styles.signUpBtn, { backgroundColor: "#004F7F" }]}
          onPress={() => router.push("/SignUp")}
          activeOpacity={0.85}
        >
          <Ionicons name="person-add-outline" size={20} color="#fff" />
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
          style={[styles.loginBtn, { borderColor: "#004F7F" }]}
          onPress={() => router.push("/Login1")}
          activeOpacity={0.85}
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

      <View style={styles.bottomNavContainer}>
        <View
          style={[
            styles.bottomNav,
            { backgroundColor: colors.navBg, borderTopColor: colors.border },
          ]}
        >
          {["Home", "Reports"].map((tabName) => {
            const tab = bottomTabs.find((t) => t.name === tabName)!;
            const isActive = activeTab === tab.name;
            return (
              <TouchableOpacity
                key={tab.name}
                style={styles.navItem}
                onPress={() => handleTabPress(tab.name)}
              >
                <View
                  style={[
                    styles.navIcon,
                    isActive && {
                      backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                      borderWidth: 2,
                      borderColor: "#0c5e5e",
                    },
                  ]}
                >
                  <Image
                    source={tab.iconImg}
                    style={styles.navIconImg}
                    resizeMode="contain"
                  />
                </View>
                <Text
                  style={[
                    styles.navText,
                    {
                      color: isActive
                        ? colors.navActive
                        : isDark
                          ? "#FFFFFF"
                          : "#6B7280",
                    },
                    isActive && { fontWeight: "700" },
                  ]}
                >
                  {tabName === "Home" ? ("home") : ("reportsTab")}
                </Text>
              </TouchableOpacity>
            );
          })}
          <View style={styles.navCenterSpacer} />
          {["History", "Settings"].map((tabName) => {
            const tab = bottomTabs.find((t) => t.name === tabName)!;
            const isActive = activeTab === tab.name;
            return (
              <TouchableOpacity
                key={tab.name}
                style={styles.navItem}
                onPress={() => handleTabPress(tab.name)}
              >
                <View
                  style={[
                    styles.navIcon,
                    isActive && {
                      backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8",
                      borderWidth: 2,
                      borderColor: isDark ? "#00A3A3" : "#C5E3ED",
                    },
                  ]}
                >
                  <Image
                    source={tab.iconImg}
                    style={styles.navIconImg}
                    resizeMode="contain"
                  />
                </View>
                <Text
                  style={[
                    styles.navText,
                    {
                      color: isActive
                        ? colors.navActive
                        : isDark
                          ? "#FFFFFF"
                          : "#6B7280",
                    },
                    isActive && { fontWeight: "700" },
                  ]}
                >
                  {tabName === "History" ? ("historyTab") : ("settingsTab")}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
        <TouchableOpacity
          style={[
            styles.cameraButton,
            {
              backgroundColor: colors.navBg,
              borderColor: isDark ? "#374151" : "#C5E3ED",
            },
          ]}
          onPress={openModal}
          activeOpacity={0.85}
        >
          <Ionicons
            name="camera-outline"
            size={30}
            color={isDark ? "#FFFFFF" : "#6B7280"}
          />
        </TouchableOpacity>
      </View>

      {showModal && (
        <View style={styles.modalOverlay} pointerEvents="box-none">
          <TouchableOpacity
            style={StyleSheet.absoluteFillObject}
            activeOpacity={1}
            onPress={closeModal}
          />
          <Animated.View
            style={[
              styles.modalBox,
              {
                backgroundColor: "#004F7F",
                opacity: modalFade,
                transform: [{ scale: modalScale }],
              },
            ]}
          >
            <View style={styles.modalHeader}>
              <View style={styles.modalIconCircle}>
                <Ionicons name="camera-outline" size={15} color="#004F7F" />
              </View>
              <Text style={styles.modalTitle}>
                Login Required
              </Text>
              <TouchableOpacity onPress={closeModal} style={styles.modalClose}>
                <Ionicons name="close" size={14} color="#fff" />
              </TouchableOpacity>
            </View>
            <Text style={styles.modalDesc}>
              You need to create an account or log in to use the camera and analyze your skin.
            </Text>
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.modalSignUp}
                onPress={() => {
                  closeModal();
                  router.push("/SignUp");
                }}
                activeOpacity={0.85}
              >
                <Ionicons name="person-add-outline" size={14} color="#fff" />
                <Text style={styles.modalSignUpText}>
                  "Create Account"
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalLogin}
                onPress={() => {
                  closeModal();
                  router.push("/Login1");
                }}
                activeOpacity={0.85}
              >
                <Ionicons name="log-in-outline" size={14} color="#004F7F" />
                <Text style={styles.modalLoginText}>
                  Log In
                </Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:          { flex: 1 },
  header:             { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:         { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:        { fontSize: 22, fontWeight: 'bold' },
  lockContainer:      { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 36, paddingBottom: 80 },
  iconWrap:           { width: 100, height: 100, borderRadius: 50, alignItems: 'center', justifyContent: 'center', marginBottom: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 6, elevation: 3 },
  lockTitle:          { fontSize: 22, fontWeight: '700', marginBottom: 12, textAlign: 'center' },
  lockSubtitle:       { fontSize: 14, textAlign: 'center', lineHeight: 22, marginBottom: 32 },
  signUpBtn:          { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 14, paddingVertical: 14, paddingHorizontal: 40, marginBottom: 12, width: '100%', justifyContent: 'center' },
  signUpBtnText:      { color: '#fff', fontSize: 16, fontWeight: '700' },
  loginBtn:           { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 14, paddingVertical: 13, paddingHorizontal: 40, borderWidth: 2, width: '100%', justifyContent: 'center' },
  loginBtnText:       { fontSize: 16, fontWeight: '700' },
  bottomNavContainer: { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
  bottomNav:          { flexDirection: 'row', paddingVertical: 10, borderTopWidth: 1, width: '100%', paddingBottom: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20 },
  navCenterSpacer:    { flex: 1 },
  navItem:            { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navIcon:            { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
  navIconImg:         { width: 44, height: 44 },
  navText:            { fontSize: 11, fontWeight: '500' },
  cameraButton:       { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 6 },
  modalOverlay:       { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,10,20,0.6)', justifyContent: 'center', alignItems: 'center', zIndex: 9999 },
  modalBox:           { width: 280, borderRadius: 18, padding: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.35, shadowRadius: 14, elevation: 14 },
  modalHeader:        { flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 8 },
  modalIconCircle:    { width: 30, height: 30, borderRadius: 15, backgroundColor: '#C5E3ED', alignItems: 'center', justifyContent: 'center' },
  modalTitle:         { flex: 1, color: '#fff', fontWeight: '700', fontSize: 13 },
  modalClose:         { width: 26, height: 26, borderRadius: 13, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  modalDesc:          { color: '#B8D4DE', fontSize: 12, lineHeight: 18, textAlign: 'center', marginBottom: 14 },
  modalActions:       { flexDirection: 'row', gap: 8 },
  modalSignUp:        { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#00A3A3', paddingVertical: 10, borderRadius: 12 },
  modalSignUpText:    { color: '#fff', fontSize: 12, fontWeight: '700' },
  modalLogin:         { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#C5E3ED', paddingVertical: 10, borderRadius: 12 },
  modalLoginText:     { fontSize: 12, fontWeight: '700' },
});