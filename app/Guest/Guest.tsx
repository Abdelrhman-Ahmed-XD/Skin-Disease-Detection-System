import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  Image,
  Modal,
  PanResponder,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useTheme } from '../ThemeContext';

const { width, height } = Dimensions.get('window');

export const GUEST_FREE_SCAN_KEY = 'guestFreeScanUsed';
const Icons = {
  home:         require('../../assets/Icons/home.png'),
  reports:      require('../../assets/Icons/Reports.png'),
  history:      require('../../assets/Icons/history.png'),
  settings:     require('../../assets/Icons/setting.png'),
  notification: require('../../assets/Icons/notification.png'),
  person:       require('../../assets/Icons/Account person.png'),
};

const BODY_IMG_W = width * 0.85;
const BODY_IMG_H = height * 0.55;

function inRect(nx: number, ny: number, x1: number, y1: number, x2: number, y2: number) {
  return nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2;
}
function inEllipse(nx: number, ny: number, cx: number, cy: number, rx: number, ry: number) {
  return ((nx - cx) / rx) ** 2 + ((ny - cy) / ry) ** 2 <= 1;
}
function checkBodyHit(nx: number, ny: number): boolean {
  if (inEllipse(nx, ny, 0.50, 0.09, 0.13, 0.10)) return true;
  if (inRect(nx, ny, 0.43, 0.17, 0.57, 0.22))    return true;
  if (inRect(nx, ny, 0.30, 0.22, 0.70, 0.56))    return true;
  if (inRect(nx, ny, 0.32, 0.54, 0.68, 0.62))    return true;
  if (inRect(nx, ny, 0.08, 0.22, 0.30, 0.52))    return true;
  if (inRect(nx, ny, 0.04, 0.50, 0.22, 0.62))    return true;
  if (inRect(nx, ny, 0.70, 0.22, 0.92, 0.52))    return true;
  if (inRect(nx, ny, 0.78, 0.50, 0.96, 0.62))    return true;
  if (inRect(nx, ny, 0.32, 0.62, 0.50, 0.82))    return true;
  if (inRect(nx, ny, 0.50, 0.62, 0.68, 0.82))    return true;
  if (inRect(nx, ny, 0.33, 0.82, 0.49, 1.00))    return true;
  if (inRect(nx, ny, 0.51, 0.82, 0.67, 1.00))    return true;
  return false;
}

const ONBOARDING_STEPS = [
  { id: 'home',     title: 'Home Screen', description: 'Home screen is for showing you your body and points of diseases in your back or front.', tabIcon: 'home-outline'          as const, navSlot: 0 },
  { id: 'reports',  title: 'Reports',     description: 'View detailed AI-generated reports about your skin health and mole analysis history.',    tabIcon: 'document-text-outline' as const, navSlot: 1 },
  { id: 'camera',   title: 'Camera',      description: 'Take a photo of a mole or skin area to get an instant AI-powered skin analysis.',        tabIcon: 'camera-outline'        as const, navSlot: 2 },
  { id: 'history',  title: 'History',     description: 'Track all your past scans and monitor changes in your skin over time.',                   tabIcon: 'time-outline'          as const, navSlot: 3 },
  { id: 'settings', title: 'Settings',    description: 'Manage your profile, notifications, and app preferences here.',                           tabIcon: 'settings-outline'      as const, navSlot: 4 },
];

const TAB_ROUTES: Record<string, string> = {
  Home:     '/Guest/Guest',
  Reports:  '/Guest/reportguest',
  Camera:   '',
  History:  '/Guest/histroyguest',
  Settings: '/Guest/settingsguest',
};

// ── Nav pill geometry ──────────────────────────────────────────
// pill: left:16, right:16, height≈70, bottom:16 from screen
// 4 tabs + 1 center spacer → each tab = navWidth/5
const NAV_PILL_BOTTOM = 16;          // pill bottom edge from screen bottom
const NAV_PILL_HEIGHT = 70;          // approx pill height
const NAV_PILL_TOP    = height - NAV_PILL_BOTTOM - NAV_PILL_HEIGHT;

// Icon is 44px tall with 4px marginBottom inside the pill.
// paddingVertical:10 at top → icon top = pillTop + 10
// icon center Y = pillTop + 10 + 22 = pillTop + 32
const ICON_CENTER_Y = NAV_PILL_TOP + 10 + 22;  // page-absolute Y of nav icon center

function getNavIconCenter(slot: number): { x: number; y: number } {
  const navWidth = width - 32; // left:16 + right:16
  const tabW     = navWidth / 5;
  let x: number;
  if      (slot === 0) x = 16 + tabW * 0.5;  // Home
  else if (slot === 1) x = 16 + tabW * 1.5;  // Reports
  else if (slot === 2) x = width / 2;          // Camera FAB (absolute center)
  else if (slot === 3) x = 16 + tabW * 3.5;  // History
  else                 x = 16 + tabW * 4.5;  // Settings

  // Camera FAB center Y is higher (it sits above the pill)
  const y = slot === 2
    ? NAV_PILL_TOP - 26 + 30   // FAB top = pillTop - 26; FAB center = +30
    : ICON_CENTER_Y;

  return { x, y };
}

type Mole     = { id: string; x: number; y: number; timestamp: number; photoUri?: string; bodyView: 'front' | 'back' };
type BodyView = 'front' | 'back';

export default function Guest() {
  const router = useRouter();
  const { isDark, colors } = useTheme();

  const [userName,  setUserName]  = useState('Guest');
  const [bodyView,  setBodyView]  = useState<BodyView>('front');
  const [moles,     setMoles]     = useState<Mole[]>([]);
  const [activeTab, setActiveTab] = useState<string>('Home');

  const [freeScanUsed, setFreeScanUsed] = useState<boolean>(false);

  const [showLoginModal,  setShowLoginModal]  = useState(false);

  // Toast
  const [showFreeScanToast, setShowFreeScanToast] = useState(false);
  const toastAnim  = useRef(new Animated.Value(0)).current;
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Onboarding
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingStep, setOnboardingStep] = useState(0);
  const fadeAnim     = useRef(new Animated.Value(0)).current;
  const scaleTooltip = useRef(new Animated.Value(0.85)).current;
  const pulseAnim    = useRef(new Animated.Value(1)).current;
  const pulseLoop    = useRef<Animated.CompositeAnimation | null>(null);

  // Tips modal (shown after onboarding)
  const [showTips, setShowTips] = useState(false);

  // Camera modal
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [cameraModalMode, setCameraModalMode] = useState<'free' | 'locked'>('free');
  const cameraFade  = useRef(new Animated.Value(0)).current;
  const cameraScale = useRef(new Animated.Value(0.85)).current;

  // Zoom & Pan
  const scale      = useRef(new Animated.Value(1)).current;
  const translateX = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(0)).current;
  const scaleVal   = useRef(1);
  const txVal      = useRef(0);
  const tyVal      = useRef(0);
  const bodyViewRef    = useRef<BodyView>('front');
  const bodyWrapperRef = useRef<any>(null);

  useEffect(() => { bodyViewRef.current = bodyView; }, [bodyView]);

  useEffect(() => {
    const s = scale.addListener(({ value }) => { scaleVal.current = value; });
    const x = translateX.addListener(({ value }) => { txVal.current = value; });
    const y = translateY.addListener(({ value }) => { tyVal.current = value; });
    return () => { scale.removeListener(s); translateX.removeListener(x); translateY.removeListener(y); };
  }, []);

  const lastDistance = useRef<number | null>(null);
  const isPinching   = useRef(false);
  const tapStartTime = useRef<number>(0);
  const tapStartPos  = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const panStartTx   = useRef(0);
  const panStartTy   = useRef(0);

  const clampTranslation = (tx: number, ty: number, sc: number) => {
    const maxX = (BODY_IMG_W * (sc - 1)) / 2;
    const maxY = (BODY_IMG_H * (sc - 1)) / 2;
    return { x: Math.max(-maxX, Math.min(maxX, tx)), y: Math.max(-maxY, Math.min(maxY, ty)) };
  };

  const panResponder = useRef(PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder:  () => true,
    onPanResponderGrant: (evt) => {
      const t = evt.nativeEvent.touches;
      isPinching.current = t.length >= 2;
      if (t.length === 1) {
        tapStartTime.current = Date.now();
        tapStartPos.current  = { x: t[0].pageX, y: t[0].pageY };
        panStartTx.current   = txVal.current;
        panStartTy.current   = tyVal.current;
      } else if (t.length === 2) {
        const dx = t[0].pageX - t[1].pageX;
        const dy = t[0].pageY - t[1].pageY;
        lastDistance.current = Math.sqrt(dx * dx + dy * dy);
      }
    },
    onPanResponderMove: (evt) => {
      const t = evt.nativeEvent.touches;
      if (t.length === 2) {
        isPinching.current = true;
        const dx = t[0].pageX - t[1].pageX;
        const dy = t[0].pageY - t[1].pageY;
        const newDist = Math.sqrt(dx * dx + dy * dy);
        if (lastDistance.current !== null) {
          const ratio    = newDist / lastDistance.current;
          const newScale = Math.max(1, Math.min(4, scaleVal.current * ratio));
          scale.setValue(newScale);
          scaleVal.current = newScale;
        }
        lastDistance.current = newDist;
        return;
      }
      if (t.length === 1 && scaleVal.current > 1 && !isPinching.current) {
        const dx = t[0].pageX - tapStartPos.current.x;
        const dy = t[0].pageY - tapStartPos.current.y;
        const c  = clampTranslation(panStartTx.current + dx, panStartTy.current + dy, scaleVal.current);
        translateX.setValue(c.x); txVal.current = c.x;
        translateY.setValue(c.y); tyVal.current = c.y;
      }
    },
    onPanResponderRelease: (evt) => {
      const touch = evt.nativeEvent.changedTouches[0];
      lastDistance.current = null;
      if (isPinching.current) { isPinching.current = false; return; }
      const elapsed = Date.now() - tapStartTime.current;
      const movedX  = Math.abs(touch.pageX - tapStartPos.current.x);
      const movedY  = Math.abs(touch.pageY - tapStartPos.current.y);
      if (elapsed < 300 && movedX < 10 && movedY < 10) {
        bodyWrapperRef.current?.measure((_fx: number, _fy: number, fw: number, fh: number, px: number, py: number) => {
          const imgL = px + (fw - BODY_IMG_W) / 2;
          const imgT = py + (fh - BODY_IMG_H) / 2;
          const cx   = imgL + BODY_IMG_W / 2;
          const cy   = imgT + BODY_IMG_H / 2;
          const relX = (touch.pageX - txVal.current - cx) / scaleVal.current + BODY_IMG_W / 2;
          const relY = (touch.pageY - tyVal.current - cy) / scaleVal.current + BODY_IMG_H / 2;
          if (checkBodyHit(relX / BODY_IMG_W, relY / BODY_IMG_H)) {
            setShowLoginModal(true);
          }
        });
      }
    },
  })).current;

  useFocusEffect(
  React.useCallback(() => {
    setActiveTab('Home');

    AsyncStorage.getItem(GUEST_FREE_SCAN_KEY).then((val) => {
      setFreeScanUsed(val === 'true');
    });

    // ✅ الحل: استخدم AsyncStorage بدل متغير session
    AsyncStorage.getItem('guestOnboardingSeen').then((seen) => {
      if (seen === 'true') return; // سبق وشافه
      // أول مرة بس
      AsyncStorage.setItem('guestOnboardingSeen', 'true');
      setOnboardingStep(0);
      setShowOnboarding(false);
      const t = setTimeout(() => {
        setShowOnboarding(true);
        animateIn();
      }, 350);
      return () => clearTimeout(t);
    });
  }, [])
);

  const animateIn = () => {
    fadeAnim.setValue(0);
    scaleTooltip.setValue(0.85);
    Animated.parallel([
      Animated.timing(fadeAnim,     { toValue: 1, duration: 260, useNativeDriver: true }),
      Animated.spring(scaleTooltip, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
    ]).start();
  };

  useEffect(() => {
    pulseLoop.current?.stop();
    if (!showOnboarding) return;
    pulseAnim.setValue(1);
    pulseLoop.current = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.2, duration: 700, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1,   duration: 700, useNativeDriver: true }),
      ])
    );
    pulseLoop.current.start();
    return () => pulseLoop.current?.stop();
  }, [showOnboarding, onboardingStep]);

  const navigateToTab = (tabName: string) => {
    const route = TAB_ROUTES[tabName];
    if (route && tabName !== 'Home' && tabName !== 'Camera') router.push(route as any);
  };

  const handleNext = () => {
    if (onboardingStep < ONBOARDING_STEPS.length - 1) {
      setOnboardingStep(prev => prev + 1);
      setTimeout(animateIn, 30);
    } else {
      // Last step → show Tips
      setShowOnboarding(false);
      pulseLoop.current?.stop();
      setShowTips(true);
    }
  };

  const handleSkipAll = () => {
    setShowOnboarding(false);
    pulseLoop.current?.stop();
    setShowTips(true);
  };

  // Toast
  const triggerFreeScanToast = () => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setShowFreeScanToast(true);
    toastAnim.setValue(0);
    Animated.spring(toastAnim, { toValue: 1, tension: 120, friction: 9, useNativeDriver: true }).start();
    toastTimer.current = setTimeout(() => {
      Animated.timing(toastAnim, { toValue: 0, duration: 250, useNativeDriver: true }).start(() => {
        setShowFreeScanToast(false);
      });
    }, 20000);
  };

  const dismissFreeScanToast = () => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    Animated.timing(toastAnim, { toValue: 0, duration: 200, useNativeDriver: true }).start(() => {
      setShowFreeScanToast(false);
    });
  };

  useEffect(() => {
    return () => { if (toastTimer.current) clearTimeout(toastTimer.current); };
  }, []);

  const openCameraModal = () => {
    AsyncStorage.getItem(GUEST_FREE_SCAN_KEY).then(val => {
      const used = val === 'true';
      setFreeScanUsed(used);
      const mode = used ? 'locked' : 'free';
      setCameraModalMode(mode);
      setShowCameraModal(true);
      cameraFade.setValue(0);
      cameraScale.setValue(0.85);
      Animated.parallel([
        Animated.timing(cameraFade,  { toValue: 1, duration: 260, useNativeDriver: true }),
        Animated.spring(cameraScale, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
      ]).start();
      if (mode === 'locked') triggerFreeScanToast();
    });
  };

  const closeCameraModal = () => setShowCameraModal(false);

  const handleUseFreeScan = () => {
    closeCameraModal();
    router.push('/Guest/cameraguest' as any);
  };

  const handleTabPress = (tabName: string) => {
    setActiveTab(tabName);
    if (tabName === 'Camera') { openCameraModal(); return; }
    if (tabName === 'Home')   { return; }
    navigateToTab(tabName);
  };

  const currentMoles = moles.filter(m => m.bodyView === bodyView);

  const toggleBodyView = (view: BodyView) => {
    setBodyView(view);
    Animated.parallel([
      Animated.spring(scale,      { toValue: 1, useNativeDriver: true }),
      Animated.spring(translateX, { toValue: 0, useNativeDriver: true }),
      Animated.spring(translateY, { toValue: 0, useNativeDriver: true }),
    ]).start();
    scaleVal.current = 1; txVal.current = 0; tyVal.current = 0;
  };

  const bottomTabs = [
    { name: 'Home',     iconImg: Icons.home     },
    { name: 'Reports',  iconImg: Icons.reports  },
    { name: 'History',  iconImg: Icons.history  },
    { name: 'Settings', iconImg: Icons.settings },
  ];

  // ── Onboarding overlay ─────────────────────────────────────
  // ── Onboarding overlay ─────────────────────────────────────
  const renderOnboarding = () => {
    if (!showOnboarding) return null;
    const step   = ONBOARDING_STEPS[onboardingStep];
    const isLast = onboardingStep === ONBOARDING_STEPS.length - 1;

    // Get the true icon center in page coordinates
    const { x: iconCX, y: iconCY } = getNavIconCenter(step.navSlot);

    // التعديل 1: تحديد هل الخطوة دي للكاميرا ولا تاب عادي
    const isCamera = step.navSlot === 2;

    // التعديل 2: تكبير حجم الدايرة لـ 90
    const SPOT = 90; 
    
    // التعديل 3: ننزل الدايرة لتحت شوية في التابات العادية علشان تغطي الكلمة 
    // الكاميرا هتفضل زي ما هي في السنتر بتاعها
    const adjustedCY = isCamera ? iconCY : iconCY + 12;

    const spotLeft   = iconCX - SPOT / 2;
    const spotTop    = adjustedCY - SPOT / 2;

    // Tooltip width and horizontal clamp
    const TW      = 215;
    let   tLeft   = iconCX - TW / 2;
    tLeft         = Math.max(12, Math.min(width - TW - 12, tLeft));

    // التعديل 4: رفع المربع الأزرق لفوق شوية علشان مساحة الدايرة الجديدة
    const tooltipTop = spotTop - 175; 

    // Arrow points down toward the spotlight; offset within tooltip
    const arrowLeft = Math.max(14, Math.min(TW - 34, iconCX - tLeft - 14));

    return (
      <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
        <View style={[StyleSheet.absoluteFill, ob.overlay]} pointerEvents="none" />

        {/* Spotlight centered exactly on the icon/text */}
        <Animated.View
          pointerEvents="none"
          style={[
            ob.spotlight,
            {
              width:        SPOT,
              height:       SPOT,
              borderRadius: SPOT / 2,
              left:         spotLeft,
              top:          spotTop,
              transform: [{ scale: pulseAnim }],
            },
          ]}
        />

        {/* Tooltip above spotlight, using absolute top positioning */}
        <Animated.View
          style={[
            ob.tooltipWrapper,
            {
              left:      tLeft,
              width:     TW,
              top:       tooltipTop,
              opacity:   fadeAnim,
              transform: [{ scale: scaleTooltip }],
            },
          ]}
        >
          <View style={ob.tooltip}>
            <View style={ob.header}>
              <View style={ob.iconCircle}>
                <Ionicons name={step.tabIcon} size={15} color="#004F7F" />
              </View>
              <Text style={ob.titleText}>{step.title}</Text>
              <TouchableOpacity onPress={handleNext} style={ob.nextBtn} activeOpacity={0.8}>
                <Text style={ob.nextBtnText}>{isLast ? 'Done' : 'Next'}</Text>
                {!isLast && <Ionicons name="arrow-forward" size={12} color="#fff" />}
              </TouchableOpacity>
            </View>
            <Text style={ob.desc}>{step.description}</Text>
            <View style={ob.footer}>
              <View style={ob.dots}>
                {ONBOARDING_STEPS.map((_, i) => (
                  <View key={i} style={[ob.dot, i === onboardingStep && ob.dotActive, i < onboardingStep && ob.dotDone]} />
                ))}
              </View>
              {/* X close button */}
              <TouchableOpacity onPress={handleSkipAll} activeOpacity={0.7} style={ob.skipBtn}>
                <Ionicons name="close" size={14} color="rgba(255,255,255,0.6)" />
              </TouchableOpacity>
            </View>
          </View>
          {/* Arrow pointing down toward spotlight */}
          <View style={[ob.arrow, { left: arrowLeft }]} />
        </Animated.View>
      </View>
    );
  };

  // ── Tips modal ─────────────────────────────────────────────
  const renderTips = () => (
    <Modal
      visible={showTips}
      transparent
      animationType="fade"
      onRequestClose={() => setShowTips(false)}
    >
      <View style={tipsStyles.overlay}>
        <View style={[tipsStyles.card, { backgroundColor: isDark ? '#1A2A35' : '#FFFFFF' }]}>
          <Ionicons name="hand-left-outline" size={40} color="#004F7F" style={{ marginBottom: 12 }} />
          <Text style={[tipsStyles.title, { color: isDark ? '#fff' : '#1F2937' }]}>Tips</Text>
          <View style={tipsStyles.row}>
            <Ionicons name="search-outline" size={20} color="#00A3A3" />
            <Text style={[tipsStyles.text, { color: isDark ? '#ccc' : '#374151' }]}>
              You can pinch to zoom in/out on the body map.
            </Text>
          </View>
          <View style={tipsStyles.row}>
            <Ionicons name="finger-print-outline" size={20} color="#00A3A3" />
            <Text style={[tipsStyles.text, { color: isDark ? '#ccc' : '#374151' }]}>
              Tap on your body to mark a skin area and scan it.
            </Text>
          </View>
          <View style={tipsStyles.row}>
            <Ionicons name="camera-outline" size={20} color="#00A3A3" />
            <Text style={[tipsStyles.text, { color: isDark ? '#ccc' : '#374151' }]}>
              You have 1 free guest scan — use the camera button!
            </Text>
          </View>
          <TouchableOpacity
            style={tipsStyles.btn}
            onPress={() => setShowTips(false)}
            activeOpacity={0.85}
          >
            <Text style={tipsStyles.btnText}>Got it!</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  // ── Free-scan-used toast ───────────────────────────────────
  const renderFreeScanToast = () => {
    if (!showFreeScanToast) return null;
    return (
      <Animated.View
        pointerEvents="box-none"
        style={[
          toastStyles.wrapper,
          {
            opacity:   toastAnim,
            transform: [{ translateY: toastAnim.interpolate({ inputRange: [0, 1], outputRange: [-30, 0] }) }],
          },
        ]}
      >
        <View style={[toastStyles.card, { backgroundColor: isDark ? '#2A1414' : '#FEF2F2', borderColor: '#EF4444' }]}>
          <View style={toastStyles.iconCircle}>
            <Ionicons name="lock-closed-outline" size={16} color="#EF4444" />
          </View>
          <View style={toastStyles.textCol}>
            <Text style={[toastStyles.title, { color: isDark ? '#FFFFFF' : '#7F1D1D' }]}>Free Scan Used</Text>
            <Text style={[toastStyles.subtitle, { color: isDark ? '#FCA5A5' : '#B91C1C' }]}>Sign up to unlock unlimited scans</Text>
          </View>
          <TouchableOpacity onPress={dismissFreeScanToast} activeOpacity={0.7} style={toastStyles.closeBtn}>
            <Ionicons name="close" size={16} color={isDark ? '#FCA5A5' : '#B91C1C'} />
          </TouchableOpacity>
        </View>
      </Animated.View>
    );
  };

  // ── Camera modal (two modes) ──────────────────────────────
  const renderCameraModal = () => {
    if (!showCameraModal) return null;

    // Camera FAB spotlight position
    const fabCX = width / 2;
    const fabCY = NAV_PILL_TOP - 26 + 30; // FAB center Y

    if (cameraModalMode === 'locked') {
      return (
        <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
          <TouchableOpacity style={[StyleSheet.absoluteFill, ob.overlay]} activeOpacity={1} onPress={closeCameraModal} />
          <Animated.View
            pointerEvents="none"
            style={[ob.spotlight, {
              width: 58, height: 58, borderRadius: 29,
              left: fabCX - 29,
              top:  fabCY - 29,
              transform: [{ scale: pulseAnim }],
            }]}
          />
          <Animated.View style={[ob.cameraModalWrapper, { opacity: cameraFade, transform: [{ scale: cameraScale }] }]}>
            <View style={ob.tooltip}>
              <View style={ob.header}>
                <View style={[ob.iconCircle, { backgroundColor: '#FFE8E8' }]}>
                  <Ionicons name="lock-closed-outline" size={15} color="#EF4444" />
                </View>
                <Text style={ob.titleText}>Free Scan Used</Text>
                <TouchableOpacity onPress={closeCameraModal} style={ob.closeBtn} activeOpacity={0.8}>
                  <Ionicons name="close" size={14} color="#fff" />
                </TouchableOpacity>
              </View>
              <View style={ob.lockedBadge}>
                <Ionicons name="scan-circle-outline" size={28} color="#9CA3AF" />
                <View style={ob.lockedBadgeTextCol}>
                  <Text style={ob.lockedBadgeTitle}>1 / 1 Free Scans Used</Text>
                  <Text style={ob.lockedBadgeSub}>Create an account for unlimited scans</Text>
                </View>
              </View>
              <View style={ob.progressTrack}>
                <View style={ob.progressFill} />
              </View>
              <Text style={ob.desc}>
                You&#39;ve used your free scan. Sign up to unlock unlimited AI skin analyses, full reports, and scan history.
              </Text>
              <View style={ob.cameraActions}>
                <TouchableOpacity style={ob.signUpBtn} activeOpacity={0.85}
                  onPress={() => { closeCameraModal(); router.push('/SignUp'); }}>
                  <Ionicons name="person-add-outline" size={14} color="#fff" />
                  <Text style={ob.signUpBtnText}>Create Account</Text>
                </TouchableOpacity>
                <TouchableOpacity style={ob.loginBtn} activeOpacity={0.85}
                  onPress={() => { closeCameraModal(); router.push('/Login1'); }}>
                  <Ionicons name="log-in-outline" size={14} color="#004F7F" />
                  <Text style={ob.loginBtnText}>Log In</Text>
                </TouchableOpacity>
              </View>
            </View>
          </Animated.View>
        </View>
      );
    }

    return (
      <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
        <TouchableOpacity style={[StyleSheet.absoluteFill, ob.overlay]} activeOpacity={1} onPress={closeCameraModal} />
        <Animated.View
          pointerEvents="none"
          style={[ob.spotlight, {
            width: 58, height: 58, borderRadius: 29,
            left: fabCX - 29,
            top:  fabCY - 29,
            transform: [{ scale: pulseAnim }],
          }]}
        />
        <Animated.View style={[ob.cameraModalWrapper, { opacity: cameraFade, transform: [{ scale: cameraScale }] }]}>
          <View style={ob.tooltip}>
            <View style={ob.header}>
              <View style={ob.iconCircle}>
                <Ionicons name="gift-outline" size={15} color="#004F7F" />
              </View>
              <Text style={ob.titleText}>1 Free Scan</Text>
              <TouchableOpacity onPress={closeCameraModal} style={ob.closeBtn} activeOpacity={0.8}>
                <Ionicons name="close" size={14} color="#fff" />
              </TouchableOpacity>
            </View>
            <View style={ob.lockedBadge}>
              <Ionicons name="scan-circle-outline" size={28} color="#00A3A3" />
              <View style={ob.lockedBadgeTextCol}>
                <Text style={ob.lockedBadgeTitle}>0 / 1 Free Scans Used</Text>
                <Text style={ob.lockedBadgeSub}>Sign up for unlimited scans</Text>
              </View>
            </View>
            <View style={ob.progressTrack}>
              <View style={[ob.progressFill, { width: '0%' }]} />
            </View>
            <Text style={ob.desc}>
              You have 1 free scan as a guest. Use it now, or create an account to get unlimited AI skin analyses.
            </Text>
            <TouchableOpacity style={[ob.signUpBtn, { marginBottom: 8 }]} activeOpacity={0.85}
              onPress={handleUseFreeScan}>
              <Ionicons name="camera-outline" size={14} color="#fff" />
              <Text style={ob.signUpBtnText}>Use Free Scan</Text>
            </TouchableOpacity>
            <View style={ob.cameraActions}>
              <TouchableOpacity style={[ob.loginBtn, { backgroundColor: '#004F7F' }]} activeOpacity={0.85}
                onPress={() => { closeCameraModal(); router.push('/SignUp'); }}>
                <Ionicons name="person-add-outline" size={14} color="#fff" />
                <Text style={[ob.loginBtnText, { color: '#fff' }]}>Sign Up Free</Text>
              </TouchableOpacity>
              <TouchableOpacity style={ob.loginBtn} activeOpacity={0.85}
                onPress={() => { closeCameraModal(); router.push('/Login1'); }}>
                <Ionicons name="log-in-outline" size={14} color="#004F7F" />
                <Text style={ob.loginBtnText}>Log In</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Animated.View>
      </View>
    );
  };

  // ── Login modal ───────────────────────────────────────────
  const renderLoginModal = () => (
    <Modal
      visible={showLoginModal}
      transparent
      animationType="fade"
      onRequestClose={() => setShowLoginModal(false)}
    >
      <View style={loginModal.overlay}>
        <View style={[loginModal.card, { backgroundColor: isDark ? '#1A2A35' : '#FFFFFF' }]}>
          <View style={loginModal.iconWrap}>
            <Ionicons name="lock-closed" size={32} color="#004F7F" />
          </View>
          <Text style={[loginModal.title, { color: isDark ? '#FFFFFF' : '#111827' }]}>Feature Locked</Text>
          <Text style={[loginModal.message, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>
            Mole mapping is available only for registered users.
          </Text>
          <TouchableOpacity style={loginModal.primaryBtn} activeOpacity={0.85}
            onPress={() => { setShowLoginModal(false); router.push('/SignUp'); }}>
            <Ionicons name="person-add-outline" size={16} color="#fff" />
            <Text style={loginModal.primaryBtnText}>Create Free Account</Text>
          </TouchableOpacity>
          <TouchableOpacity style={loginModal.secondaryBtn} activeOpacity={0.85}
            onPress={() => { setShowLoginModal(false); router.push('/Login1'); }}>
            <Ionicons name="log-in-outline" size={16} color="#004F7F" />
            <Text style={loginModal.secondaryBtnText}>Log In</Text>
          </TouchableOpacity>
          <TouchableOpacity style={loginModal.ghostBtn} activeOpacity={0.7}
            onPress={() => setShowLoginModal(false)}>
            <Text style={[loginModal.ghostBtnText, { color: isDark ? '#6B7280' : '#9CA3AF' }]}>Maybe Later</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={["top"]}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={colors.background} />

      {/* Header */}
      <View style={[styles.headerCard, { backgroundColor: colors.card }]}>
        <View style={styles.headerContent}>
          <TouchableOpacity style={[styles.profileIconContainer, { backgroundColor: isDark ? "#1A3A4A" : "#E8F4F8" }]}>
            <Image source={Icons.person} style={styles.headerIconImg} resizeMode="contain" />
          </TouchableOpacity>
          <View style={styles.welcomeContainer}>
            <Text style={[styles.welcomeLabel, { color: "#00A3A3" }]}>Welcome,</Text>
            <Text style={{ fontWeight: "bold", marginLeft: 4, marginTop: 3, fontSize: 17, color: isDark ? "#fff" : "#000" }}>
              {userName}
            </Text>
          </View>
          <TouchableOpacity
            style={[styles.notificationButton, { backgroundColor: isDark ? "#1E2A35" : "#fff" }]}
            onPress={() => router.push("/Guest/notificationguest")}
            activeOpacity={0.8}
          >
            <Image source={Icons.notification} style={styles.notifIconImg} resizeMode="contain" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Title */}
      <View style={styles.titleContainer}>
        <Text style={[styles.title, { color: isDark ? "#fff" : "#000" }]}>
          Let&#39;s Check your{" "}
          <Text style={[styles.titleBold, { color: isDark ? "#00A3A3" : "#004F7F" }]}>Skin</Text>
        </Text>
      </View>

      {/* Body map */}
      <View style={[styles.bodyMainContainer, { backgroundColor: colors.background }]}>
        <View
          style={[styles.bodyClipWrapper, { backgroundColor: colors.background }]}
          {...panResponder.panHandlers}
          ref={(r) => { bodyWrapperRef.current = r; }}
        >
          <Animated.View
            style={[
              styles.bodyImageWrapper,
              { backgroundColor: colors.background, transform: [{ scale }, { translateX }, { translateY }] },
            ]}
          >
            <Image
              source={bodyView === "front"
                ? require("../../assets/images/body-front.png")
                : require("../../assets/images/body-back.png")}
              style={[styles.bodyImage, { backgroundColor: colors.background }]}
              resizeMode="contain"
            />
            {currentMoles.map((mole) => {
              const S = 28;
              return (
                <View key={mole.id} style={[styles.moleContainer, { left: mole.x - S / 2, top: mole.y - S / 2 }]} pointerEvents="box-none">
                  <TouchableOpacity
                    activeOpacity={0.8}
                    style={{ flexDirection: "row", alignItems: "center", gap: 4 }}
                    onPress={() => setShowLoginModal(true)}
                  >
                    <View style={styles.moleInner}>
                      <Text style={styles.moleIcon}>+</Text>
                    </View>
                    {mole.photoUri && (
                      <Image source={{ uri: mole.photoUri }} style={styles.moleThumbnail} />
                    )}
                  </TouchableOpacity>
                </View>
              );
            })}
          </Animated.View>
        </View>
      </View>

      {/* Toggle Front/Back */}
      <View style={styles.bottomControls}>
        <View style={[styles.toggleWrapper, { backgroundColor: isDark ? "#1A3A4A" : "#B8D4DE" }]}>
          <TouchableOpacity onPress={() => toggleBodyView("front")} style={[styles.toggleButton, bodyView === "front" && styles.toggleButtonActive]}>
            <Text style={[styles.toggleText, { color: isDark ? "#FFFFFF" : "#6B7280" }, bodyView === "front" && styles.toggleTextActive]}>Front</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => toggleBodyView("back")} style={[styles.toggleButton, bodyView === "back" && styles.toggleButtonActive]}>
            <Text style={[styles.toggleText, { color: isDark ? "#FFFFFF" : "#6B7280" }, bodyView === "back" && styles.toggleTextActive]}>Back</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Bottom Nav */}
      <View style={styles.bottomNavContainer}>
        <View style={[styles.bottomNav, {
          backgroundColor: colors.navBg,
          borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
        }]}>
          {["Home", "Reports"].map((tabName) => {
            const tab = bottomTabs.find((t) => t.name === tabName)!;
            const isActive = activeTab === tab.name;
            return (
              <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                <View style={[styles.navIcon, { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                  isActive && { backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8", borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#C5E3ED" }]}>
                  <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                </View>
                <Text style={[styles.navText, { color: isActive ? colors.navActive : isDark ? "#FFFFFF" : "#6B7280" }, isActive && { fontWeight: "700" }]}>
                  {tab.name}
                </Text>
              </TouchableOpacity>
            );
          })}
          <View style={styles.navCenterSpacer} />
          {["History", "Settings"].map((tabName) => {
            const tab = bottomTabs.find((t) => t.name === tabName)!;
            const isActive = activeTab === tab.name;
            return (
              <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                <View style={[styles.navIcon, { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                  isActive && { backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8", borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#C5E3ED" }]}>
                  <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                </View>
                <Text style={[styles.navText, { color: isActive ? colors.navActive : isDark ? "#FFFFFF" : "#6B7280" }, isActive && { fontWeight: "700" }]}>
                  {tab.name}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Camera FAB */}
        <TouchableOpacity
          style={[
            styles.cameraButton,
            {
              backgroundColor: colors.navBg,
              // التعديل هنا: تم تغيير اللون ليكون أحمر صريح وثابت (Opacity 100%) في الوضعين
              borderColor: freeScanUsed
                ? '#EF4444' 
                : (isDark ? '#374151' : '#C5E3ED'),
            },
            activeTab === "Camera" && { borderColor: "#004F7F", backgroundColor: isDark ? "#1A3A4A" : "#E8F4F8" },
          ]}
          onPress={() => handleTabPress("Camera")}
          activeOpacity={0.85}
        >
          <Ionicons
            name={freeScanUsed ? "lock-closed-outline" : "camera-outline"}
            size={30}
            color={
              freeScanUsed ? '#EF4444' // اللون هنا كمان أحمر صريح للقفل نفسه
                : activeTab === "Camera" ? "#004F7F"
                : isDark ? "#FFFFFF" : "#6B7280"
            }
          />
        </TouchableOpacity>
      </View>

      {renderFreeScanToast()}
      {renderTips()}
      {renderOnboarding()}
      {renderCameraModal()}
      {renderLoginModal()}
    </SafeAreaView>
  );
}

const tipsStyles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 28 },
  card: { width: '100%', borderRadius: 24, paddingVertical: 28, paddingHorizontal: 24, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.18, shadowRadius: 20, elevation: 16 },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 16 },
  row: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12, width: '100%' },
  text: { flex: 1, fontSize: 14, lineHeight: 20 },
  btn: { marginTop: 8, backgroundColor: '#004F7F', borderRadius: 12, paddingVertical: 12, paddingHorizontal: 32 },
  btnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});

const toastStyles = StyleSheet.create({
  wrapper: { position: 'absolute', top: 100, left: 16, right: 16, zIndex: 10000, elevation: 10000 },
  card: { flexDirection: 'row', alignItems: 'center', gap: 10, borderRadius: 16, borderWidth: 1.5, paddingVertical: 12, paddingHorizontal: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 10, elevation: 10 },
  iconCircle: { width: 32, height: 32, borderRadius: 16, backgroundColor: 'rgba(239,68,68,0.15)', alignItems: 'center', justifyContent: 'center' },
  textCol: { flex: 1 },
  title: { fontSize: 13, fontWeight: '700' },
  subtitle: { fontSize: 11.5, marginTop: 2 },
  closeBtn: { padding: 4 },
});

const ob = StyleSheet.create({
  root:    { zIndex: 9999, elevation: 9999 },
  overlay: { backgroundColor: 'rgba(0,10,20,0.60)', zIndex: 1 },
  spotlight: {
    position: 'absolute',
    backgroundColor: 'rgba(0,163,163,0.15)',
    borderWidth: 2.5, borderColor: '#00A3A3', zIndex: 2,
    shadowColor: '#00A3A3', shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.7, shadowRadius: 12, elevation: 8,
  },
  tooltipWrapper:     { position: 'absolute', zIndex: 3 },
  cameraModalWrapper: { position: 'absolute', zIndex: 3, width: 270, left: (width - 270) / 2, top: height * 0.25 },
  tooltip: {
    backgroundColor: '#004F7F', borderRadius: 18, padding: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.35, shadowRadius: 14, elevation: 14,
  },
  header:      { flexDirection: 'row', alignItems: 'center', marginBottom: 8, gap: 6 },
  iconCircle:  { width: 28, height: 28, borderRadius: 14, backgroundColor: '#C5E3ED', alignItems: 'center', justifyContent: 'center' },
  titleText:   { flex: 1, color: '#fff', fontWeight: '700', fontSize: 13 },
  nextBtn:     { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: '#00A3A3', paddingVertical: 5, paddingHorizontal: 10, borderRadius: 14 },
  nextBtnText: { color: '#fff', fontSize: 11, fontWeight: '700' },
  closeBtn:    { width: 26, height: 26, borderRadius: 13, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center' },
  desc:        { color: '#B8D4DE', fontSize: 11.5, lineHeight: 17, textAlign: 'center', marginBottom: 10 },
  footer:      { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  dots:        { flexDirection: 'row', gap: 4, alignItems: 'center' },
  dot:         { width: 5, height: 5, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.25)' },
  dotActive:   { width: 14, backgroundColor: '#00A3A3' },
  dotDone:     { backgroundColor: 'rgba(255,255,255,0.55)' },
  skipBtn:     { width: 26, height: 26, borderRadius: 13, backgroundColor: 'rgba(255,255,255,0.15)', alignItems: 'center', justifyContent: 'center' },
  arrow:       { position: 'absolute', bottom: -10, width: 0, height: 0, borderLeftWidth: 12, borderRightWidth: 12, borderTopWidth: 11, borderLeftColor: 'transparent', borderRightColor: 'transparent', borderTopColor: '#004F7F' },
  cameraActions:      { flexDirection: 'row', gap: 8, marginTop: 4 },
  signUpBtn:          { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#00A3A3', paddingVertical: 10, borderRadius: 14 },
  signUpBtnText:      { color: '#fff', fontSize: 12, fontWeight: '700' },
  loginBtn:           { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#C5E3ED', paddingVertical: 10, borderRadius: 14 },
  loginBtnText:       { color: '#004F7F', fontSize: 12, fontWeight: '700' },
  lockedBadge:        { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 10, padding: 8, marginBottom: 8 },
  lockedBadgeTextCol: { flex: 1 },
  lockedBadgeTitle:   { color: '#fff', fontSize: 12, fontWeight: '700' },
  lockedBadgeSub:     { color: '#93C5CE', fontSize: 10, marginTop: 2 },
  progressTrack:      { height: 5, backgroundColor: 'rgba(255,255,255,0.15)', borderRadius: 3, marginBottom: 10, overflow: 'hidden' },
  progressFill:       { height: '100%', width: '100%', backgroundColor: '#EF4444', borderRadius: 3 },
});

const styles = StyleSheet.create({
  container:           { flex: 1 },
  titleContainer:      { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 4 },
  title:               { fontSize: 20, textAlign: 'center' },
  titleBold:           { fontWeight: '700' },
  bodyMainContainer:   { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16, marginBottom: 160 },
  bodyClipWrapper:     { width: BODY_IMG_W, height: BODY_IMG_H, overflow: 'hidden', alignItems: 'center', justifyContent: 'center', borderRadius: 12 },
  bodyImageWrapper:    { width: BODY_IMG_W, height: BODY_IMG_H, alignItems: 'center', justifyContent: 'center' },
  bodyImage:           { width: '100%', height: '100%' },
  moleContainer:       { position: 'absolute', flexDirection: 'row', alignItems: 'center', gap: 4 },
  moleInner:           { width: 28, height: 28, borderRadius: 14, backgroundColor: '#004F7F', alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#FFFFFF' },
  moleIcon:            { color: '#FFFFFF', fontSize: 18, fontWeight: '700', lineHeight: 22 },
  moleThumbnail:       { width: 38, height: 38, borderRadius: 8, borderWidth: 2, borderColor: '#FFFFFF', backgroundColor: '#ccc' },
  bottomControls:      { position: 'absolute', bottom: 140, left: 0, right: 0, alignItems: 'center' },
  toggleWrapper:       { flexDirection: 'row', borderRadius: 25, padding: 4, width: width * 0.45 },
  toggleButton:        { flex: 1, paddingVertical: 8, alignItems: 'center', justifyContent: 'center', borderRadius: 20 },
  toggleButtonActive:  { backgroundColor: '#004F7F' },
  toggleText:          { fontSize: 14, fontWeight: '600' },
  toggleTextActive:    { color: '#FFFFFF', fontWeight: '700' },
  bottomNavContainer:  { position: 'absolute', bottom: 16, left: 16, right: 16, alignItems: 'center' },
  bottomNav:           { flexDirection: 'row', paddingVertical: 10, paddingBottom: 14, borderRadius: 28, borderWidth: 1, width: '100%', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.12, shadowRadius: 12, elevation: 8 },
  navCenterSpacer:     { flex: 1 },
  navItem:             { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navIcon:             { width: 42, height: 42, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
  navIconImg:          { width: 42, height: 42 },
  headerIconImg:       { width: 55, height: 55 },
  notifIconImg:        { width: 56, height: 56 },
  navText:             { fontSize: 11, fontWeight: '500' },
  cameraButton:        { position: 'absolute', top: -26, alignSelf: 'center', width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 3, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.15, shadowRadius: 8, elevation: 8 },
  headerCard:          { marginHorizontal: 16, marginTop: 12, marginBottom: 8, borderRadius: 20, paddingVertical: 14, paddingHorizontal: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 },
  headerContent:       { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  profileIconContainer:{ width: 52, height: 52, borderRadius: 26, justifyContent: 'center', alignItems: 'center', borderWidth: 2, borderColor: '#C5E3ED' },
  welcomeContainer:    { flex: 1, marginLeft: 12, flexDirection: 'row', alignItems: 'center' },
  welcomeLabel:        { fontSize: 18, fontStyle: 'italic' },
  notificationButton:  { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
});

const loginModal = StyleSheet.create({
  overlay:         { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 28 },
  card:            { width: '100%', borderRadius: 24, paddingVertical: 28, paddingHorizontal: 24, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.18, shadowRadius: 20, elevation: 16 },
  iconWrap:        { width: 64, height: 64, borderRadius: 32, backgroundColor: '#E8F4F8', alignItems: 'center', justifyContent: 'center', marginBottom: 16 },
  title:           { fontSize: 20, fontWeight: '700', marginBottom: 8, textAlign: 'center' },
  message:         { fontSize: 14, lineHeight: 20, textAlign: 'center', marginBottom: 24 },
  primaryBtn:      { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, width: '100%', backgroundColor: '#004F7F', paddingVertical: 13, borderRadius: 14, marginBottom: 10 },
  primaryBtnText:  { color: '#fff', fontSize: 14, fontWeight: '700' },
  secondaryBtn:    { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, width: '100%', backgroundColor: '#C5E3ED', paddingVertical: 13, borderRadius: 14, marginBottom: 10 },
  secondaryBtnText:{ color: '#004F7F', fontSize: 14, fontWeight: '700' },
  ghostBtn:        { paddingVertical: 8, paddingHorizontal: 16, marginTop: 2 },
  ghostBtnText:    { fontSize: 13, fontWeight: '500' },
});