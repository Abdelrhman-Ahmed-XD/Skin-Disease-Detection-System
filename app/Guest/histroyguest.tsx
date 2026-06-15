import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  StatusBar,
  StyleSheet,
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

export default function GuestHistoryPage() {
  const router = useRouter();
  const { isDark, colors } = useTheme();

  const pageBg = isDark ? colors.background : '#D8E9F0';

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
        <Text style={[styles.headerTitle, { color: isDark ? '#fff' : '#000' }]}>History</Text>
        <View style={{ width: 40 }} />
      </View>

      <View style={styles.lockContainer}>
        <View style={[styles.iconWrap, { backgroundColor: isDark ? '#fff' : '#004F7F' }]}>
          <Ionicons name="time-outline" size={48} color={isDark ? '#004F7F' : '#fff'} />
        </View>
        <Text style={[styles.lockTitle, { color: isDark ? '#fff' : '#000' }]}>History Unavailable</Text>
        <Text style={[styles.lockSubtitle, { color: isDark ? '#AAAAAA' : colors.subText }]}>
          You need to log in or create an account to view your past scan history.
        </Text>
        <TouchableOpacity
          style={[styles.signUpBtn, { backgroundColor: '#004F7F' }]}
          onPress={() => router.push('/SignUp')}
          activeOpacity={0.85}
        >
          <Ionicons name="person-add-outline" size={18} color="#fff" />
          <Text style={styles.signUpBtnText}>Sign Up</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.loginBtn, { borderColor: '#004F7F' }]}
          onPress={() => router.push('/Login1')}
          activeOpacity={0.85}
        >
          <Ionicons name="log-in-outline" size={28} color={isDark ? '#fff' : '#004F7F'} />
          <Text style={[styles.loginBtnText, { color: isDark ? '#fff' : '#004F7F' }]}>Log In</Text>
        </TouchableOpacity>
      </View>

      <GuestBottomNav activeTab="History" onCameraPress={openCameraModal} />
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
  container:    { flex: 1 },
  header:       { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:   { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitle:  { fontSize: 22, fontWeight: 'bold' },
  lockContainer:{ flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 36, paddingBottom: 100 },
  iconWrap:     { width: 100, height: 100, borderRadius: 50, alignItems: 'center', justifyContent: 'center', marginBottom: 24 },
  lockTitle:    { fontSize: 22, fontWeight: '700', marginBottom: 12, textAlign: 'center' },
  lockSubtitle: { fontSize: 14, textAlign: 'center', lineHeight: 22, marginBottom: 32 },
  signUpBtn:    { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 14, paddingVertical: 14, paddingHorizontal: 40, marginBottom: 12, width: '100%', justifyContent: 'center' },
  signUpBtnText:{ color: '#fff', fontSize: 16, fontWeight: '700' },
  loginBtn:     { flexDirection: 'row', alignItems: 'center', gap: 8, borderRadius: 14, paddingVertical: 13, paddingHorizontal: 40, borderWidth: 2, width: '100%', justifyContent: 'center' },
  loginBtnText: { fontSize: 16, fontWeight: '700' },
});