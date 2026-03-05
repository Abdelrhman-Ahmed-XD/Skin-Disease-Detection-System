import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  Image,
  PanResponder,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const MOLES_STORAGE_KEY = 'savedMoles';
const { width, height } = Dimensions.get('window');

// ── Body hit-test ──────────────────────────────────────────────
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

// ── Onboarding steps (exact same as Nextscreens) ───────────────
const ONBOARDING_STEPS = [
    { id: 'home',     title: 'Home Screen', description: 'Home screen is for showing you your body and points of diseases in your back or front.', tabIcon: 'home-outline'          as const, navSlot: 0 },
    { id: 'reports',  title: 'Reports',     description: 'View detailed AI-generated reports about your skin health and mole analysis history.',    tabIcon: 'document-text-outline' as const, navSlot: 1 },
    { id: 'camera',   title: 'Camera',      description: 'Take a photo of a mole or skin area to get an instant AI-powered skin analysis.',        tabIcon: 'camera-outline'        as const, navSlot: 2 },
    { id: 'history',  title: 'History',     description: 'Track all your past scans and monitor changes in your skin over time.',                   tabIcon: 'time-outline'          as const, navSlot: 3 },
    { id: 'settings', title: 'Settings',    description: 'Manage your profile, notifications, and app preferences here.',                           tabIcon: 'settings-outline'      as const, navSlot: 4 },
];

const TAB_TO_STEP: Record<string, number> = {
    Home: 0, Reports: 1, Camera: 2, History: 3, Settings: 4,
};

const NAV_BAR_HEIGHT = 55;

function getNavX(slot: number): number {
    const s = width / 5;
    if (slot === 2) return width / 2;
    return s * slot + s / 2;
}

type Mole     = { id: string; x: number; y: number; timestamp: number; photoUri?: string; bodyView: 'front' | 'back' };
type BodyView = 'front' | 'back';

export default function Guest() {
    const router = useRouter();
    const [userName,  setUserName]  = useState('');
    const [bodyView,  setBodyView]  = useState<BodyView>('front');
    const [moles,     setMoles]     = useState<Mole[]>([]);
    const [activeTab, setActiveTab] = useState<string>('Home');

    // ── Onboarding state ───────────────────────────────────────
    const [showOnboarding, setShowOnboarding] = useState(false);
    const [onboardingStep, setOnboardingStep] = useState(0);
    const fadeAnim     = useRef(new Animated.Value(0)).current;
    const scaleTooltip = useRef(new Animated.Value(0.85)).current;
    const pulseAnim    = useRef(new Animated.Value(1)).current;
    const pulseLoop    = useRef<Animated.CompositeAnimation | null>(null);

    // ── Camera signup modal state ──────────────────────────────
    const [showCameraModal, setShowCameraModal] = useState(false);
    const cameraFade  = useRef(new Animated.Value(0)).current;
    const cameraScale = useRef(new Animated.Value(0.85)).current;

    // ── Zoom & Pan ─────────────────────────────────────────────
    const scale        = useRef(new Animated.Value(1)).current;
    const translateX   = useRef(new Animated.Value(0)).current;
    const translateY   = useRef(new Animated.Value(0)).current;
    const scaleVal     = useRef(1);
    const txVal        = useRef(0);
    const tyVal        = useRef(0);
    const bodyViewRef  = useRef<BodyView>('front');
    const bodyWrapperRef   = useRef<any>(null);
    const openCameraModalRef = useRef<() => void>(() => {});

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
        const maxX = (width  * (sc - 1)) / 2;
        const maxY = (height * (sc - 1)) / 2;
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
                    const imgW = width * 0.85;
                    const imgH = height * 0.55;
                    const cx   = px + (fw - imgW) / 2 + imgW / 2;
                    const cy   = py + (fh - imgH) / 2 + imgH / 2;
                    const relX = (touch.pageX - txVal.current - cx) / scaleVal.current + imgW / 2;
                    const relY = (touch.pageY - tyVal.current - cy) / scaleVal.current + imgH / 2;
                    if (checkBodyHit(relX / imgW, relY / imgH)) {
                        openCameraModalRef.current();
                    }
                });
            }
        },
    })).current;

    // ── Guest counter ──────────────────────────────────────────
    useEffect(() => {
        const assignGuestName = async () => {
            try {
                const countStr = await AsyncStorage.getItem('guest_count');
                const count    = countStr ? parseInt(countStr) + 1 : 1;
                await AsyncStorage.setItem('guest_count', String(count));
                setUserName(`Guest ${count}`);
            } catch {}
        };
        assignGuestName();
    }, []);

    useFocusEffect(
        React.useCallback(() => { setActiveTab('Home'); }, [])
    );

    // ── Onboarding animation ───────────────────────────────────
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

    const openOnboarding = (tabName: string) => {
        const step = TAB_TO_STEP[tabName] ?? 0;
        setOnboardingStep(step);
        setShowOnboarding(true);
        setTimeout(animateIn, 30);
    };

    const handleNext = () => {
        if (onboardingStep < ONBOARDING_STEPS.length - 1) {
            setOnboardingStep(prev => prev + 1);
            setTimeout(animateIn, 30);
        } else {
            setShowOnboarding(false);
        }
    };

    const handleSkipAll = () => setShowOnboarding(false);

    // ── Camera signup modal ────────────────────────────────────
    const openCameraModal = () => {
        setShowCameraModal(true);
        cameraFade.setValue(0);
        cameraScale.setValue(0.85);
        Animated.parallel([
            Animated.timing(cameraFade,  { toValue: 1, duration: 260, useNativeDriver: true }),
            Animated.spring(cameraScale, { toValue: 1, tension: 130, friction: 8, useNativeDriver: true }),
        ]).start();
    };

    // Keep ref in sync so panResponder can call it
    useEffect(() => { openCameraModalRef.current = openCameraModal; });

    const closeCameraModal = () => setShowCameraModal(false);

    // ── Tab press ──────────────────────────────────────────────
    const handleTabPress = (tabName: string) => {
        setActiveTab(tabName);
        if (tabName === 'Camera') {
            openCameraModal();
        } else {
            openOnboarding(tabName);
        }
        // No navigation for guests at all
    };

    // ── Mole helpers ───────────────────────────────────────────
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
        { name: 'Home',     icon: 'home-outline'          },
        { name: 'Reports',  icon: 'document-text-outline' },
        { name: 'History',  icon: 'time-outline'          },
        { name: 'Settings', icon: 'settings-outline'      },
        { name: 'Camera',   icon: 'camera-outline'        },
    ];

    // ── Render onboarding (exact same UI as Nextscreens) ───────
    const renderOnboarding = () => {
        if (!showOnboarding) return null;

        const step   = ONBOARDING_STEPS[onboardingStep];
        const isLast = onboardingStep === ONBOARDING_STEPS.length - 1;
        const navX   = getNavX(step.navSlot);
        const spotY  = height - NAV_BAR_HEIGHT + (step.navSlot === 2 ? -30 : 6);

        const TW        = 210;
        let   tLeft     = navX - TW / 2;
        tLeft = Math.max(12, Math.min(width - TW - 12, tLeft));
        const tBottom   = NAV_BAR_HEIGHT + 50;
        const arrowLeft = Math.max(14, Math.min(TW - 34, navX - tLeft - 14));

        return (
            <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
                <TouchableOpacity
                    style={[StyleSheet.absoluteFill, ob.overlay]}
                    activeOpacity={1}
                    onPress={handleSkipAll}
                />
                <Animated.View
                    pointerEvents="none"
                    style={[ob.spotlight, {
                        left: navX - 34,
                        top:  spotY - 34,
                        transform: [{ scale: pulseAnim }],
                    }]}
                />
                <Animated.View style={[ob.tooltipWrapper, {
                    bottom: tBottom, left: tLeft, width: TW,
                    opacity: fadeAnim, transform: [{ scale: scaleTooltip }],
                }]}>
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
                                    <View key={i} style={[
                                        ob.dot,
                                        i === onboardingStep && ob.dotActive,
                                        i <  onboardingStep && ob.dotDone,
                                    ]} />
                                ))}
                            </View>
                            <TouchableOpacity onPress={handleSkipAll} activeOpacity={0.7}>
                                <Text style={ob.skip}>Skip all</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                    <View style={[ob.arrow, { left: arrowLeft }]} />
                </Animated.View>
            </View>
        );
    };

    // ── Render camera signup modal ─────────────────────────────
    const renderCameraModal = () => {
        if (!showCameraModal) return null;

        return (
            <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
                <TouchableOpacity
                    style={[StyleSheet.absoluteFill, ob.overlay]}
                    activeOpacity={1}
                    onPress={closeCameraModal}
                />
                {/* Spotlight on camera button */}
                <Animated.View
                    pointerEvents="none"
                    style={[ob.spotlight, {
                        left: width / 2 - 34,
                        top:  height - NAV_BAR_HEIGHT - 64,
                        transform: [{ scale: pulseAnim }],
                    }]}
                />
                {/* Modal card — centered on screen */}
                <Animated.View style={[ob.cameraModalWrapper, {
                    opacity: cameraFade,
                    transform: [{ scale: cameraScale }],
                }]}>
                    <View style={ob.tooltip}>
                        {/* Header */}
                        <View style={ob.header}>
                            <View style={ob.iconCircle}>
                                <Ionicons name="camera-outline" size={15} color="#004F7F" />
                            </View>
                            <Text style={ob.titleText}>Camera Access</Text>
                            <TouchableOpacity onPress={closeCameraModal} style={ob.closeBtn} activeOpacity={0.8}>
                                <Ionicons name="close" size={14} color="#fff" />
                            </TouchableOpacity>
                        </View>

                        {/* Message */}
                        <Text style={ob.desc}>
                            To take photos and analyze your skin, you need to create an account or log in first.
                        </Text>

                        {/* Action buttons */}
                        <View style={ob.cameraActions}>
                            <TouchableOpacity
                                style={ob.signUpBtn}
                                activeOpacity={0.85}
                                onPress={() => { closeCameraModal(); router.push('/SignUp'); }}
                            >
                                <Ionicons name="person-add-outline" size={14} color="#fff" />
                                <Text style={ob.signUpBtnText}>Create Account</Text>
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={ob.loginBtn}
                                activeOpacity={0.85}
                                onPress={() => { closeCameraModal(); router.push('/Login1'); }}
                            >
                                <Ionicons name="log-in-outline" size={14} color="#004F7F" />
                                <Text style={ob.loginBtnText}>Log In</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </Animated.View>
            </View>
        );
    };

    // ══════════════════════════════════════════════════════════
    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            <StatusBar barStyle="dark-content" backgroundColor="#D8E9F0" />

            {/* Header */}
            <View style={styles.headerCard}>
                <View style={styles.headerContent}>
                    <TouchableOpacity style={styles.profileIconContainer}>
                        <Ionicons name="person-outline" size={32} color="#004F7F" />
                    </TouchableOpacity>
                    <View style={styles.welcomeContainer}>
                        <Text style={styles.welcomeLabel}>Welcome,</Text>
                        <Text style={{ fontWeight: 'bold', marginLeft: 4, marginTop: 3, fontSize: 17 }}>{userName}</Text>
                    </View>
                    <TouchableOpacity style={styles.notificationButton}>
                        <Ionicons name="notifications-outline" size={28} color="#00A3A3" />
                    </TouchableOpacity>
                </View>
            </View>

            {/* Title */}
            <View style={styles.titleContainer}>
                <Text style={styles.title}>Let's Check your <Text style={styles.titleBold}>Skin</Text></Text>
            </View>

            {/* Body + gestures */}
            <View style={styles.bodyMainContainer}>
                <View style={styles.bodyTouchable} {...panResponder.panHandlers} ref={r => { bodyWrapperRef.current = r; }}>
                    <Animated.View style={[styles.bodyImageWrapper, { transform: [{ scale }, { translateX }, { translateY }] }]}>
                        <Image
                            source={bodyView === 'front'
                                ? require('../../assets/images/body-front.png')
                                : require('../../assets/images/body-back.png')}
                            style={styles.bodyImage}
                            resizeMode="contain"
                        />
                        {currentMoles.map(mole => {
                            const S = 28;
                            return (
                                <View key={mole.id} style={[styles.moleContainer, { left: mole.x - S / 2, top: mole.y - S / 2 }]} pointerEvents="box-none">
                                    <TouchableOpacity
                                        activeOpacity={0.8}
                                        style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}
                                        onPress={openCameraModal}
                                    >
                                        <View style={styles.moleInner}><Text style={styles.moleIcon}>+</Text></View>
                                        {mole.photoUri && <Image source={{ uri: mole.photoUri }} style={styles.moleThumbnail} />}
                                    </TouchableOpacity>
                                </View>
                            );
                        })}
                    </Animated.View>
                </View>
            </View>

            {/* Toggle Front / Back */}
            <View style={styles.bottomControls}>
                <View style={styles.toggleWrapper}>
                    <TouchableOpacity onPress={() => toggleBodyView('front')} style={[styles.toggleButton, bodyView === 'front' && styles.toggleButtonActive]}>
                        <Text style={[styles.toggleText, bodyView === 'front' && styles.toggleTextActive]}>Front</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={() => toggleBodyView('back')} style={[styles.toggleButton, bodyView === 'back' && styles.toggleButtonActive]}>
                        <Text style={[styles.toggleText, bodyView === 'back' && styles.toggleTextActive]}>Back</Text>
                    </TouchableOpacity>
                </View>
            </View>

            {/* Bottom Nav */}
            <View style={styles.bottomNavContainer}>
                <View style={styles.bottomNav}>
                    {['Home', 'Reports'].map(tabName => {
                        const tab = bottomTabs.find(t => t.name === tabName)!;
                        return (
                            <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                                <View style={[styles.navIcon, activeTab === tab.name && styles.navIconActive]}>
                                    <Ionicons name={tab.icon as any} size={26} color={activeTab === tab.name ? '#004F7F' : '#6B7280'} />
                                </View>
                                <Text style={[styles.navText, activeTab === tab.name && styles.navTextActive]}>{tab.name}</Text>
                            </TouchableOpacity>
                        );
                    })}
                    <View style={styles.navCenterSpacer} />
                    {['History', 'Settings'].map(tabName => {
                        const tab = bottomTabs.find(t => t.name === tabName)!;
                        return (
                            <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                                <View style={[styles.navIcon, activeTab === tab.name && styles.navIconActive]}>
                                    <Ionicons name={tab.icon as any} size={26} color={activeTab === tab.name ? '#004F7F' : '#6B7280'} />
                                </View>
                                <Text style={[styles.navText, activeTab === tab.name && styles.navTextActive]}>{tab.name}</Text>
                            </TouchableOpacity>
                        );
                    })}
                </View>
                <TouchableOpacity
                    style={[styles.cameraButton, activeTab === 'Camera' && styles.cameraButtonActive]}
                    onPress={() => handleTabPress('Camera')}
                    activeOpacity={0.85}
                >
                    <Ionicons name="camera-outline" size={30} color={activeTab === 'Camera' ? '#004F7F' : '#6B7280'} />
                </TouchableOpacity>
            </View>

            {/* Overlays — always last so on top */}
            {renderOnboarding()}
            {renderCameraModal()}

        </SafeAreaView>
    );
}

// ── Styles ─────────────────────────────────────────────────────
const ob = StyleSheet.create({
    root:    { zIndex: 9999, elevation: 9999 },
    overlay: { backgroundColor: 'rgba(0,10,20,0.60)', zIndex: 1 },
    spotlight: {
        position: 'absolute', width: 68, height: 68, borderRadius: 34,
        backgroundColor: 'rgba(0,163,163,0.15)',
        borderWidth: 2.5, borderColor: '#00A3A3', zIndex: 2,
        shadowColor: '#00A3A3', shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.7, shadowRadius: 12, elevation: 8,
    },
    tooltipWrapper: { position: 'absolute', zIndex: 3 },
    cameraModalWrapper: {
        position: 'absolute', zIndex: 3,
        width: 260,
        left: (width - 260) / 2,
        top:  height * 0.28,
    },
    tooltip: {
        backgroundColor: '#004F7F', borderRadius: 18, padding: 14,
        shadowColor: '#000', shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.35, shadowRadius: 14, elevation: 14,
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
    skip:        { color: 'rgba(255,255,255,0.5)', fontSize: 11, fontWeight: '600' },
    arrow: {
        position: 'absolute', bottom: -10, width: 0, height: 0,
        borderLeftWidth: 12, borderRightWidth: 12, borderTopWidth: 11,
        borderLeftColor: 'transparent', borderRightColor: 'transparent',
        borderTopColor: '#004F7F',
    },
    cameraActions:  { flexDirection: 'row', gap: 8, marginTop: 4 },
    signUpBtn:      { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#00A3A3', paddingVertical: 9, borderRadius: 14 },
    signUpBtnText:  { color: '#fff', fontSize: 11, fontWeight: '700' },
    loginBtn:       { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, backgroundColor: '#C5E3ED', paddingVertical: 9, borderRadius: 14 },
    loginBtnText:   { color: '#004F7F', fontSize: 11, fontWeight: '700' },
});

const styles = StyleSheet.create({
    container:            { flex: 1, backgroundColor: '#D8E9F0' },
    titleContainer:       { padding: 20, marginTop: 16 },
    title:                { fontSize: 20, textAlign: 'center', color: '#1F2937' },
    titleBold:            { fontWeight: '700' },
    bodyMainContainer:    { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16, marginBottom: 200 },
    bodyTouchable:        { width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center' },
    bodyImageWrapper:     { position: 'relative', width: width * 0.85, height: height * 0.55, alignItems: 'center', justifyContent: 'center', backgroundColor: '#D8E9F0', overflow: 'visible' },
    bodyImage:            { width: '100%', height: '100%', backgroundColor: '#D8E9F0' },
    moleContainer:        { position: 'absolute', flexDirection: 'row', alignItems: 'center', gap: 4 },
    moleInner:            { width: 28, height: 28, borderRadius: 14, backgroundColor: '#004F7F', alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#FFFFFF' },
    moleIcon:             { color: '#FFFFFF', fontSize: 18, fontWeight: '700', lineHeight: 22 },
    moleThumbnail:        { width: 38, height: 38, borderRadius: 8, borderWidth: 2, borderColor: '#FFFFFF', backgroundColor: '#ccc' },
    bottomControls:       { position: 'absolute', bottom: 100, left: 0, right: 0, alignItems: 'center', marginBottom: 35 },
    toggleWrapper:        { flexDirection: 'row', backgroundColor: '#B8D4DE', borderRadius: 25, padding: 4, width: width * 0.45 },
    toggleButton:         { flex: 1, paddingVertical: 8, alignItems: 'center', justifyContent: 'center', borderRadius: 20 },
    toggleButtonActive:   { backgroundColor: '#004F7F' },
    toggleText:           { fontSize: 14, fontWeight: '600', color: '#6B7280' },
    toggleTextActive:     { color: '#FFFFFF', fontWeight: '700' },
    bottomNavContainer:   { position: 'absolute', bottom: 0, left: 0, right: 0, alignItems: 'center' },
    bottomNav:            { flexDirection: 'row', backgroundColor: '#FFFFFF', paddingVertical: 10, borderTopWidth: 1, borderTopColor: '#E5E7EB', width: '100%', paddingBottom: 16 },
    navCenterSpacer:      { flex: 1 },
    navItem:              { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:              { width: 44, height: 44, borderRadius: 22, backgroundColor: '#F9FAFB', justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconActive:        { backgroundColor: '#E8F4F8', borderWidth: 2, borderColor: '#C5E3ED' },
    navText:              { fontSize: 11, color: '#6B7280', fontWeight: '500' },
    navTextActive:        { fontSize: 11, color: '#004F7F', fontWeight: '700' },
    cameraButton: {
        position: 'absolute', top: -26, alignSelf: 'center',
        width: 60, height: 60, borderRadius: 30,
        backgroundColor: '#FFFFFF', justifyContent: 'center', alignItems: 'center',
        borderWidth: 3, borderColor: '#C5E3ED',
        shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.12, shadowRadius: 6, elevation: 6,
    },
    cameraButtonActive:   { borderColor: '#004F7F', backgroundColor: '#E8F4F8' },
    headerCard: {
        backgroundColor: '#FFFFFF', marginHorizontal: 16, marginTop: 12, marginBottom: 8,
        borderRadius: 20, paddingVertical: 14, paddingHorizontal: 16,
        shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
    },
    headerContent:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    profileIconContainer: { width: 52, height: 52, borderRadius: 26, backgroundColor: '#E8F4F8', justifyContent: 'center', alignItems: 'center', borderWidth: 2, borderColor: '#C5E3ED' },
    welcomeContainer:     { flex: 1, marginLeft: 12, flexDirection: 'row', alignItems: 'center' },
    welcomeLabel:         { fontSize: 18, color: '#00A3A3', fontStyle: 'italic' },
    userName:             { fontSize: 18, fontWeight: '700', color: '#1F2937', marginTop: 2 },
    notificationButton:   { width: 44, height: 44, borderRadius: 22, backgroundColor: '#F9FAFB', justifyContent: 'center', alignItems: 'center' },
});