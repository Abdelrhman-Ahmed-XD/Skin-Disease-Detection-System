import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  Image,
  ScrollView,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from '../ThemeContext';
import GuestBottomNav from './Guestbottomnav';

const { width, height } = Dimensions.get('window');

const NAV_PILL_BOTTOM = 16;
const NAV_PILL_HEIGHT = 70;
const NAV_PILL_TOP    = height - NAV_PILL_BOTTOM - NAV_PILL_HEIGHT;
const FAB_CENTER_Y    = NAV_PILL_TOP - 26 + 30;
const FAB_CENTER_X    = width / 2;

const GUEST_FREE_SCAN_KEY = 'guestFreeScanUsed';

const Icons = {
  person:   require('../../assets/Icons/Account person.png'),
  about:    require('../../assets/Icons/about.png'),
  darkMode: require('../../assets/Icons/dark mode.png'),
  help:     require('../../assets/Icons/help.png'),
};

export default function GuestSettingsPage() {
  const router = useRouter();
  const { isDark, toggleTheme, colors } = useTheme();

  const pageBg = isDark ? colors.background : '#D8E9F0';
  const color  = isDark ? '#fff' : '#000';

  const [freeScanUsed,    setFreeScanUsed]    = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [cameraMode,      setCameraMode]      = useState<'free' | 'locked'>('free');

  const cameraFade  = useRef(new Animated.Value(0)).current;
  const cameraScale = useRef(new Animated.Value(0.85)).current;
  const pulseAnim   = useRef(new Animated.Value(1)).current;
  const pulseLoop   = useRef<Animated.CompositeAnimation | null>(null);

  useFocusEffect(
    React.useCallback(() => {
      AsyncStorage.getItem(GUEST_FREE_SCAN_KEY).then(val => {
        setFreeScanUsed(val === 'true');
      });
    }, [])
  );

  const openCameraModal = () => {
    AsyncStorage.getItem(GUEST_FREE_SCAN_KEY).then(val => {
      const used = val === 'true';
      setFreeScanUsed(used);
      setCameraMode(used ? 'locked' : 'free');
      setShowCameraModal(true);
      cameraFade.setValue(0);
      cameraScale.setValue(0.85);
      Animated.parallel([
        Animated.timing(cameraFade,  { toValue: 1, duration: 220, useNativeDriver: true }),
        Animated.spring(cameraScale, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
      ]).start();
      pulseAnim.setValue(1);
      pulseLoop.current?.stop();
      pulseLoop.current = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.18, duration: 700, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1,    duration: 700, useNativeDriver: true }),
        ])
      );
      pulseLoop.current.start();
    });
  };

  const closeCameraModal = () => {
    pulseLoop.current?.stop();
    setShowCameraModal(false);
  };

  const renderCameraModal = () => {
    if (!showCameraModal) return null;
    const SPOT     = 58;
    const spotLeft = FAB_CENTER_X - SPOT / 2;
    const spotTop  = FAB_CENTER_Y - SPOT / 2;
    const isLocked = cameraMode === 'locked';

    return (
      <View style={[StyleSheet.absoluteFill, cm.root]} pointerEvents="box-none">
        <TouchableOpacity style={[StyleSheet.absoluteFill, cm.overlay]} activeOpacity={1} onPress={closeCameraModal} />
        <Animated.View pointerEvents="none" style={[cm.spotlight, { width: SPOT, height: SPOT, borderRadius: SPOT / 2, left: spotLeft, top: spotTop, transform: [{ scale: pulseAnim }] }]} />
        <Animated.View style={[cm.cardWrapper, { opacity: cameraFade, transform: [{ scale: cameraScale }] }]}>
          <View style={cm.card}>
            <View style={cm.header}>
              <View style={[cm.iconCircle, isLocked && { backgroundColor: '#FFE8E8' }]}>
                <Ionicons name={isLocked ? 'lock-closed-outline' : 'gift-outline'} size={15} color={isLocked ? '#EF4444' : '#004F7F'} />
              </View>
              <Text style={cm.titleText}>{isLocked ? 'Free Scan Used' : '1 Free Scan'}</Text>
              <TouchableOpacity onPress={closeCameraModal} style={cm.closeBtn} activeOpacity={0.8}>
                <Ionicons name="close" size={14} color="#fff" />
              </TouchableOpacity>
            </View>
            <View style={cm.badge}>
              <Ionicons name="scan-circle-outline" size={26} color={isLocked ? '#9CA3AF' : '#00A3A3'} />
              <View style={{ flex: 1 }}>
                <Text style={cm.badgeTitle}>{isLocked ? '1 / 1 Free Scans Used' : '0 / 1 Free Scans Used'}</Text>
                <Text style={cm.badgeSub}>Create an account for unlimited scans</Text>
              </View>
            </View>
            <View style={cm.progressTrack}>
              <View style={[cm.progressFill, { width: isLocked ? '100%' : '0%' }]} />
            </View>
            <Text style={cm.desc}>
              {isLocked
                ? "You've used your free scan. Sign up to unlock unlimited AI skin analyses."
                : 'You have 1 free scan as a guest. Use it now or create an account for unlimited scans.'}
            </Text>
            {!isLocked && (
              <TouchableOpacity style={[cm.primaryBtn, { marginBottom: 8 }]} activeOpacity={0.85}
                onPress={() => { closeCameraModal(); router.push('/Guest/cameraguest' as any); }}>
                <Ionicons name="camera-outline" size={14} color="#fff" />
                <Text style={cm.primaryBtnText}>Use Free Scan</Text>
              </TouchableOpacity>
            )}
            <View style={cm.actions}>
              <TouchableOpacity style={[cm.actionBtn, { backgroundColor: '#004F7F' }]} activeOpacity={0.85}
                onPress={() => { closeCameraModal(); router.push('/SignUp'); }}>
                <Ionicons name="person-add-outline" size={13} color="#fff" />
                <Text style={[cm.actionBtnText, { color: '#fff' }]}>Sign Up</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[cm.actionBtn, { backgroundColor: '#C5E3ED' }]} activeOpacity={0.85}
                onPress={() => { closeCameraModal(); router.push('/Login1'); }}>
                <Ionicons name="log-in-outline" size={13} color="#004F7F" />
                <Text style={[cm.actionBtnText, { color: '#004F7F' }]}>Log In</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Animated.View>
      </View>
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={['top']}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.replace('/Guest/Guest')}>
          <Ionicons name="chevron-back" size={24} color={isDark ? '#FFFFFF' : '#1F2937'} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: isDark ? '#fff' : '#000' }]}>Settings</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Guest profile card */}
        <TouchableOpacity
          style={[styles.profileCard, { backgroundColor: colors.card, flexDirection: 'row' }]}
          onPress={() => router.push('/SignUp')}
          activeOpacity={0.8}
        >
          <View style={[styles.profileAvatar, { backgroundColor: isDark ? '#2A3F50' : '#fff', marginRight: 14 }]}>
            <Image source={Icons.person} style={styles.profileAvatarIcon} resizeMode="contain" />
          </View>
          <View style={[styles.profileInfo, { alignItems: 'flex-start' }]}>
            <Text style={[styles.profileName, { color: isDark ? '#fff' : '#000' }]}>Guest</Text>
            <Text style={[styles.profileEmail, { color: isDark ? '#AAAAAA' : colors.subText }]}>Tap to sign in</Text>
          </View>
          <View style={[styles.loginBadge, { backgroundColor: '#004F7F' }]}>
            <Text style={styles.loginBadgeText}>Sign In</Text>
          </View>
        </TouchableOpacity>

        {/* Preferences */}
        <Text style={[styles.sectionTitle, { color: isDark ? '#AAAAAA' : colors.subText }]}>PREFERENCES</Text>
        <View style={[styles.card, { backgroundColor: colors.card }]}>
          <View style={[styles.settingsRow, { borderBottomWidth: 1, borderBottomColor: colors.border }]}>
            <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1E2A35' : '#fff' }]}>
              <Image source={Icons.darkMode} style={{ width: 32, height: 32 }} resizeMode="contain" />
            </View>
            <Text style={[styles.settingsLabel, { color }]}>Dark Mode</Text>
            <Switch
              value={isDark}
              onValueChange={toggleTheme}
              trackColor={{ false: colors.border, true: '#C5E3ED' }}
              thumbColor={isDark ? colors.primary : colors.subText}
            />
          </View>
        </View>

        {/* App */}
        <Text style={[styles.sectionTitle, { color: isDark ? '#AAAAAA' : colors.subText }]}>APP</Text>
        <View style={[styles.card, { backgroundColor: colors.card }]}>
          <TouchableOpacity
            style={[styles.settingsRow, { borderBottomWidth: 1, borderBottomColor: colors.border }]}
            onPress={() => router.push('/Guest/aboutguest')}
          >
            <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1E2A35' : '#fff' }]}>
              <Image source={Icons.about} style={{ width: 26, height: 26 }} resizeMode="contain" />
            </View>
            <Text style={[styles.settingsLabel, { color }]}>About</Text>
            <Ionicons name="chevron-forward" size={18} color={colors.border} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.settingsRow}
            onPress={() => router.push('/Guest/helpguest')}
          >
            <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1E2A35' : '#fff' }]}>
              <Image source={Icons.help} style={{ width: 26, height: 26 }} resizeMode="contain" />
            </View>
            <Text style={[styles.settingsLabel, { color }]}>Help & Support</Text>
            <Ionicons name="chevron-forward" size={18} color={colors.border} />
          </TouchableOpacity>
        </View>

        {/* CTA */}
        <View style={[styles.ctaCard, { backgroundColor: isDark ? '#1E2A35' : '#E8F4F8', borderColor: '#C5E3ED' }]}>
          <Ionicons name="lock-closed-outline" size={22} color={isDark ? '#fff' : '#004F7F'} style={{ marginBottom: 8 }} />
          <Text style={[styles.ctaText, { color: isDark ? '#FFFFFF' : '#374151' }]}>
            Sign in to unlock all features including language, customization, notifications, and your personal data.
          </Text>
          <View style={styles.ctaButtons}>
            <TouchableOpacity
              style={[styles.ctaSignUp, { backgroundColor: '#004F7F' }]}
              onPress={() => router.push('/SignUp')}
              activeOpacity={0.85}
            >
              <Ionicons name="person-add-outline" size={15} color="#fff" />
              <Text style={styles.ctaSignUpText}>Sign Up</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.ctaLogin, { borderColor: '#004F7F' }]}
              onPress={() => router.push('/Login1')}
              activeOpacity={0.85}
            >
              <Ionicons name="log-in-outline" size={28} color={isDark ? '#fff' : '#004F7F'} />
              <Text style={[styles.ctaLoginText, { color: isDark ? '#fff' : '#004F7F' }]}>Log In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      <GuestBottomNav activeTab="Settings" onCameraPress={openCameraModal} />
      {renderCameraModal()}
    </SafeAreaView>
  );
}

const cm = StyleSheet.create({
  root:          { zIndex: 9999, elevation: 9999 },
  overlay:       { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,10,20,0.60)' },
  spotlight:     { position: 'absolute', backgroundColor: 'rgba(0,163,163,0.15)', borderWidth: 2.5, borderColor: '#00A3A3', zIndex: 2, shadowColor: '#00A3A3', shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.7, shadowRadius: 12, elevation: 8 },
  cardWrapper:   { position: 'absolute', zIndex: 3, width: 270, left: (width - 270) / 2, top: height * 0.25 },
  card:          { backgroundColor: '#004F7F', borderRadius: 18, padding: 14 },
  header:        { flexDirection: 'row', alignItems: 'center', marginBottom: 8, gap: 6 },
  iconCircle:    { width: 28, height: 28, borderRadius: 14, backgroundColor: '#C5E3ED', alignItems: 'center', justifyContent: 'center' },
  titleText:     { flex: 1, color: '#fff', fontWeight: '700', fontSize: 13 },
  closeBtn:      { width: 26, height: 26, borderRadius: 13, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  badge:         { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 10, padding: 8, marginBottom: 8 },
  badgeTitle:    { color: '#fff', fontSize: 12, fontWeight: '700' },
  badgeSub:      { color: '#93C5CE', fontSize: 10, marginTop: 2 },
  progressTrack: { height: 5, backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 3, marginBottom: 10, overflow: 'hidden' },
  progressFill:  { height: '100%', backgroundColor: '#EF4444', borderRadius: 3 },
  desc:          { color: '#B8D4DE', fontSize: 11.5, lineHeight: 17, textAlign: 'center', marginBottom: 10 },
  primaryBtn:    { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#00A3A3', paddingVertical: 10, borderRadius: 14 },
  primaryBtnText:{ color: '#fff', fontSize: 12, fontWeight: '700' },
  actions:       { flexDirection: 'row', gap: 8 },
  actionBtn:     { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, paddingVertical: 10, borderRadius: 14 },
  actionBtnText: { fontSize: 12, fontWeight: '700' },
});

const styles = StyleSheet.create({
  container:      { flex: 1 },
  header:         { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:     { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:    { fontSize: 20, fontWeight: 'bold' },
  scrollView:     { flex: 1 },
  scrollContent:  { padding: 16, paddingBottom: 110 },
  profileCard:    { borderRadius: 16, padding: 16, alignItems: 'center', marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
  profileAvatar:  { width: 52, height: 52, borderRadius: 26, justifyContent: 'center', alignItems: 'center', overflow: 'hidden' },
  profileAvatarIcon: { width: 52, height: 52 },
  profileInfo:    { flex: 1 },
  profileName:    { fontSize: 16, fontWeight: '700' },
  profileEmail:   { fontSize: 13, marginTop: 2 },
  loginBadge:     { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  loginBadgeText: { color: '#fff', fontSize: 12, fontWeight: '700' },
  sectionTitle:   { fontSize: 13, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 8, marginLeft: 4 },
  card:           { borderRadius: 16, marginBottom: 20, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
  settingsRow:    { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 14 },
  iconWrap:       { width: 36, height: 36, borderRadius: 10, justifyContent: 'center', alignItems: 'center', marginRight: 14 },
  settingsLabel:  { flex: 1, fontSize: 15, fontWeight: '500' },
  ctaCard:        { borderRadius: 16, padding: 18, alignItems: 'center', borderWidth: 1, marginBottom: 20 },
  ctaText:        { fontSize: 13, textAlign: 'center', lineHeight: 20, marginBottom: 16 },
  ctaButtons:     { flexDirection: 'row', gap: 10, width: '100%' },
  ctaSignUp:      { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 12, paddingVertical: 12 },
  ctaSignUpText:  { color: '#fff', fontWeight: '700', fontSize: 14 },
  ctaLogin:       { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 12, paddingVertical: 12, borderWidth: 2 },
  ctaLoginText:   { fontWeight: '700', fontSize: 14 },
});