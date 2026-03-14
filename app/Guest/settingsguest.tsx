import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Animated, Image, ScrollView, StatusBar,
  StyleSheet, Switch, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useCustomize } from '../Customize/Customizecontext';
import { useTheme } from '../ThemeContext';

const Icons = {
  home:         require('../../assets/Icons/home.png'),
  reports:      require('../../assets/Icons/Reports.png'),
  history:      require('../../assets/Icons/history.png'),
  settings:     require('../../assets/Icons/setting.png'),
  person:       require('../../assets/Icons/Account person.png'),
  about:        require('../../assets/Icons/about.png'),
  darkMode:     require('../../assets/Icons/dark mode.png'),
  help:         require('../../assets/Icons/help.png'),
  notification: require('../../assets/Icons/notification.png'),
};

const ICON_SIZES = {
  darkMode: { width: 32, height: 32 },
  about:    { width: 26, height: 26 },
  help:     { width: 26, height: 26 },
};

export default function GuestSettingsPage() {
  const router = useRouter();
  const { isDark, toggleTheme, colors } = useTheme();
  const { settings } = useCustomize();

  const textColor = isDark ? '#FFFFFF' : (settings.textColor || '#1F2937');

  const customText = {
    fontSize:   settings.fontSize,
    color:      textColor,
    fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
  };

  const pageBg = isDark ? colors.background : settings.backgroundColor;
  const [activeTab, setActiveTab] = useState('Settings');

  const [showLoginModal, setShowLoginModal] = useState(false);
  const modalFade  = useRef(new Animated.Value(0)).current;
  const modalScale = useRef(new Animated.Value(0.85)).current;

  const openLoginModal = () => {
    setShowLoginModal(true);
    modalFade.setValue(0); modalScale.setValue(0.85);
    Animated.parallel([
      Animated.timing(modalFade,  { toValue: 1, duration: 220, useNativeDriver: true }),
      Animated.spring(modalScale, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
    ]).start();
  };

  const closeLoginModal = () => {
    Animated.timing(modalFade, { toValue: 0, duration: 180, useNativeDriver: true })
      .start(() => setShowLoginModal(false));
  };

  useFocusEffect(React.useCallback(() => { setActiveTab('Settings'); }, []));

  const bottomTabs = [
    { name: 'Home',     iconImg: Icons.home     },
    { name: 'Reports',  iconImg: Icons.reports  },
    { name: 'History',  iconImg: Icons.history  },
    { name: 'Settings', iconImg: Icons.settings },
  ];

  const handleTabPress = (tabName: string) => {
    setActiveTab(tabName);
    if (tabName === 'Camera') { openLoginModal(); return; }
    switch (tabName) {
      case 'Home':     router.push('/Guest/Guest');         break;
      case 'Reports':  router.push('/Guest/reportguest');   break;
      case 'History':  router.push('/Guest/histroyguest');  break;
    }
  };

  type RowProps = {
    style?:        object;
    iconBgColor?:  string;   // ← لون الباكجراوند بتاع الأيكون
    iconImg:       any;
    iconSize:      { width: number; height: number };
    label:         string;
    onPress?:      () => void;
    rightElement?: React.ReactNode;
    isLast?:       boolean;
    disabled?:     boolean;
  };

  const SettingsRow: React.FC<RowProps> = ({
    style, iconBgColor, iconImg, iconSize, label, onPress, rightElement, isLast = false, disabled = false,
  }) => (
    <TouchableOpacity
      style={[
        styles.settingsRow,
        style,
        { flexDirection: 'row', opacity: disabled ? 0.4 : 1 },
        !isLast && { borderBottomWidth: 1, borderBottomColor: colors.border },
      ]}
      onPress={disabled ? undefined : onPress}
      activeOpacity={disabled ? 1 : 0.7}
    >
      <View style={[
        styles.settingsIconWrap,
        {
          marginRight: 14,
          marginLeft: 0,
          backgroundColor: iconBgColor ?? (isDark ? '#1A3A4A' : '#fff'),  // ← هنا
        },
      ]}>
        <Image source={iconImg} style={{ width: iconSize.width, height: iconSize.height }} resizeMode="contain" />
      </View>
      <Text style={[styles.settingsLabel, customText]}>
        {label}
      </Text>
      {rightElement ?? (onPress && !disabled
        ? <Ionicons name="chevron-forward" size={18} color={colors.border} />
        : null)}
    </TouchableOpacity>
  );

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: pageBg }]}
      edges={["top"]}
    >
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity
          style={[styles.backButton, { borderColor: colors.border }]}
          onPress={() => router.back()}
        >
          <Ionicons name="chevron-back" size={24} color={textColor} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, customText]}>{"settings"}</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Guest Profile */}
        <TouchableOpacity
          style={[
            styles.profileCard,
            {
              backgroundColor: colors.card,
              flexDirection: "row",
            },
          ]}
          onPress={openLoginModal}
          activeOpacity={0.8}
        >
          <View
            style={[
              styles.profileAvatar,
              {
                backgroundColor: isDark ? "#2A3F50" : "#fff",
                marginRight: 14,
                marginLeft: 0,
              },
            ]}
          >
            <Image
              source={Icons.person}
              style={styles.profileAvatarIconImg}
              resizeMode="contain"
            />
          </View>
          <View style={[styles.profileInfo, { alignItems: "flex-start" }]}>
            <Text
              style={[styles.profileName, customText, { fontWeight: "700" }]}
            >
              Guest
            </Text>
            <Text
              style={[
                styles.profileEmail,
                {
                  color: isDark ? "#AAAAAA" : colors.subText,
                  fontFamily: customText.fontFamily,
                  fontSize: customText.fontSize,
                },
              ]}
            >
              Tap to sign in
            </Text>
          </View>
          <View style={[styles.loginBadge, { backgroundColor: "#004F7F" }]}>
            <Text style={styles.loginBadgeText}>Sign In</Text>
          </View>
        </TouchableOpacity>

        {/* Preferences — Dark Mode */}
        <Text
          style={[
            styles.sectionTitle,
            {
              color: isDark ? "#AAAAAA" : colors.subText,
              textAlign: "left",
              fontSize: customText.fontSize ? customText.fontSize * 0.75 : 13,
            },
          ]}
        >
          {"preferences"}
        </Text>
        <View style={[styles.card, { backgroundColor: colors.card }]}>
          <SettingsRow
            iconImg={Icons.darkMode}
            iconSize={ICON_SIZES.darkMode}
            iconBgColor={isDark ? "#1E2A35" : "#fff"} // ← غيّر اللون هنا
            label={"darkMode"}
            isLast
            rightElement={
              <Switch
                value={isDark}
                onValueChange={toggleTheme}
                trackColor={{ false: colors.border, true: "#C5E3ED" }}
                thumbColor={isDark ? colors.primary : colors.subText}
              />
            }
          />
        </View>

        {/* App — About & Help */}
        <Text
          style={[
            styles.sectionTitle,
            {
              color: isDark ? "#AAAAAA" : colors.subText,
              textAlign: "left",
              fontSize: customText.fontSize ? customText.fontSize * 0.75 : 13,
            },
          ]}
        >
          {"app"}
        </Text>
        <View style={[styles.card, { backgroundColor: colors.card }]}>
          <SettingsRow
            style={{
              backgroundColor: isDark ? "#1E2A35" : "#fff",
            }}
            iconImg={Icons.about}
            iconSize={ICON_SIZES.about}
            iconBgColor={isDark ? "#1E2A35" : "#fff"} // ← غيّر اللون هنا
            label={"about"}
            onPress={() => router.push("/Guest/aboutguest")}
          />
          <SettingsRow
            style={{ backgroundColor: isDark ? "#1E2A35" : "#fff" }}
            iconImg={Icons.help}
            iconSize={ICON_SIZES.help}
            iconBgColor={isDark ? "#1E2A35" : "#fff"} // ← غيّر اللون هنا
            label={"helpSupport"}
            onPress={() => router.push("/Guest/helpguest")}
            isLast
            
          />
        </View>

        {/* CTA */}
        <View
          style={[
            styles.ctaCard,
            {
              backgroundColor: isDark ? "#1E2A35" : "#E8F4F8",
              borderColor: "#C5E3ED",
            },
          ]}
        >
          <Ionicons
            name="lock-closed-outline"
            size={22}
            style={{ marginBottom: 8, color: isDark ? "#fff" : "#004F7F" }}
          />
          <Text
            style={[styles.ctaText, { color: isDark ? "#FFFFFF" : "#374151" }]}
          >
            Sign in to unlock all features including language, customization,
            notifications, and your personal data.
          </Text>
          <View style={styles.ctaButtons}>
            <TouchableOpacity
              style={[styles.ctaSignUp, { backgroundColor: "#004F7F" }]}
              onPress={() => router.push("/SignUp")}
              activeOpacity={0.85}
            >
              <Ionicons name="person-add-outline" size={15} color="#fff" />
              <Text style={styles.ctaSignUpText}>Sign Up</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.ctaLogin, { borderColor: "#004F7F" }]}
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
                  styles.ctaLoginText,
                  { color: isDark ? "#fff" : "#004F7F" },
                ]}
              >
                Log In
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      {/* Bottom Nav */}
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
                  {tabName === "Home" ? "home" : "reportsTab"}
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
                  {tabName === "History" ? "historyTab" : "settingsTab"}
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
          onPress={openLoginModal}
          activeOpacity={0.85}
        >
          <Ionicons
            name="camera-outline"
            size={30}
            color={isDark ? "#FFFFFF" : "#6B7280"}
          />
        </TouchableOpacity>
      </View>

      {/* Login Modal */}
      {showLoginModal && (
        <View style={styles.modalOverlay} pointerEvents="box-none">
          <TouchableOpacity
            style={StyleSheet.absoluteFillObject}
            activeOpacity={1}
            onPress={closeLoginModal}
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
                <Ionicons
                  name="lock-closed-outline"
                  size={18}
                  color="#004F7F"
                />
              </View>
              <Text style={styles.modalTitle}>Login Required</Text>
              <TouchableOpacity
                onPress={closeLoginModal}
                style={styles.modalClose}
              >
                <Ionicons name="close" size={14} color="#fff" />
              </TouchableOpacity>
            </View>
            <Text style={styles.modalDesc}>
              This feature is only available to registered users. Create an
              account or log in to continue.
            </Text>
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.modalSignUp}
                onPress={() => {
                  closeLoginModal();
                  router.push("/SignUp");
                }}
                activeOpacity={0.85}
              >
                <Ionicons name="person-add-outline" size={14} color="#fff" />
                <Text style={styles.modalSignUpText}>Create Account</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalLogin}
                onPress={() => {
                  closeLoginModal();
                  router.push("/Login1");
                }}
                activeOpacity={0.85}
              >
                <Ionicons name="log-in-outline" size={14} color="#004F7F" />
                <Text style={styles.modalLoginText}>Log In</Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:           { flex: 1 },
  header:              { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:          { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:         { fontSize: 20, fontWeight: 'bold' },
  scrollView:          { flex: 1 },
  scrollContent:       { padding: 16, paddingBottom: 110 },
  profileCard:         { borderRadius: 16, padding: 16, flexDirection: 'row', alignItems: 'center', marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
  profileAvatar:       { width: 52, height: 52, borderRadius: 26, justifyContent: 'center', alignItems: 'center', overflow: 'hidden' },
  profileAvatarIconImg:{ width: 52, height: 52 },
  profileInfo:         { flex: 1 },
  profileName:         { fontSize: 16, fontWeight: '700' },
  profileEmail:        { fontSize: 13, marginTop: 2 },
  loginBadge:          { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  loginBadgeText:      { color: '#fff', fontSize: 12, fontWeight: '700' },
  sectionTitle:        { fontSize: 13, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 8, marginLeft: 4 },
  card:                { borderRadius: 16, marginBottom: 20, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
  settingsRow:         { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 14 },
  settingsIconWrap:    { width: 36, height: 36, borderRadius: 10, justifyContent: 'center', alignItems: 'center' },
  settingsLabel:       { flex: 1, fontSize: 15, fontWeight: '500' },
  ctaCard:             { borderRadius: 16, padding: 18, alignItems: 'center', borderWidth: 1, marginBottom: 20 },
  ctaText:             { fontSize: 13, textAlign: 'center', lineHeight: 20, marginBottom: 16 },
  ctaButtons:          { flexDirection: 'row', gap: 10, width: '100%' },
  ctaSignUp:           { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 12, paddingVertical: 12 },
  ctaSignUpText:       { color: '#fff', fontWeight: '700', fontSize: 14 },
  ctaLogin:            { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 12, paddingVertical: 12, borderWidth: 2 },
  ctaLoginText:        { fontWeight: '700', fontSize: 14 },
  bottomNavContainer:  { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
  bottomNav:           { flexDirection: 'row', paddingVertical: 10, borderTopWidth: 1, width: '100%', paddingBottom: 16, borderTopLeftRadius: 20, borderTopRightRadius: 20 },
  navCenterSpacer:     { flex: 1 },
  navItem:             { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navIcon:             { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
  navIconImg:          { width: 44, height: 44 },
  navText:             { fontSize: 11, fontWeight: '500' },
  cameraButton:        { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 6, elevation: 6 },
  modalOverlay:        { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,10,20,0.6)', justifyContent: 'center', alignItems: 'center', zIndex: 9999 },
  modalBox:            { width: 280, borderRadius: 18, padding: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.35, shadowRadius: 14, elevation: 14 },
  modalHeader:         { flexDirection: 'row', alignItems: 'center', marginBottom: 10, gap: 8 },
  modalIconCircle:     { width: 30, height: 30, borderRadius: 15, backgroundColor: '#C5E3ED', alignItems: 'center', justifyContent: 'center' },
  modalTitle:          { flex: 1, color: '#fff', fontWeight: '700', fontSize: 13 },
  modalClose:          { width: 26, height: 26, borderRadius: 13, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  modalDesc:           { color: '#B8D4DE', fontSize: 12, lineHeight: 18, textAlign: 'center', marginBottom: 14 },
  modalActions:        { flexDirection: 'row', gap: 8 },
  modalSignUp:         { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#00A3A3', paddingVertical: 10, borderRadius: 12 },
  modalSignUpText:     { color: '#fff', fontSize: 12, fontWeight: '700' },
  modalLogin:          { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#C5E3ED', paddingVertical: 10, borderRadius: 12 },
  modalLoginText:      { fontSize: 12, fontWeight: '700' },
});