import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import { loadMolesFromFirestore, deleteMole as deleteMoleFromFirestore } from '../../Firebase/firestoreService';
import {
    Alert,
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

// ── Custom Icon Images ─────────────────────────────────────────
const Icons = {
  home:         require('../../assets/Icons/home.png'),
  reports:      require('../../assets/Icons/Reports.png'),
  history:      require('../../assets/Icons/history.png'),
  settings:     require('../../assets/Icons/setting.png'),
  notification: require('../../assets/Icons/notification.png'),
  person:       require('../../assets/Icons/Account person.png'),
};

const STORAGE_KEY      = 'signupDraft';
const MOLES_STORAGE_KEY = 'savedMoles';

const { width, height } = Dimensions.get('window');

function inRect(nx: number, ny: number, x1: number, y1: number, x2: number, y2: number) {
    return nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2;
}
function inEllipse(nx: number, ny: number, cx: number, cy: number, rx: number, ry: number) {
    return ((nx - cx) / rx) ** 2 + ((ny - cy) / ry) ** 2 <= 1;
}
function checkBodyHit(nx: number, ny: number, view: 'front' | 'back'): boolean {
    if (inEllipse(nx, ny, 0.50, 0.09, 0.13, 0.10)) return true;
    if (inRect(nx, ny, 0.43, 0.17, 0.57, 0.22)) return true;
    if (inRect(nx, ny, 0.30, 0.22, 0.70, 0.56)) return true;
    if (inRect(nx, ny, 0.32, 0.54, 0.68, 0.62)) return true;
    if (inRect(nx, ny, 0.08, 0.22, 0.30, 0.52)) return true;
    if (inRect(nx, ny, 0.04, 0.50, 0.22, 0.62)) return true;
    if (inRect(nx, ny, 0.70, 0.22, 0.92, 0.52)) return true;
    if (inRect(nx, ny, 0.78, 0.50, 0.96, 0.62)) return true;
    if (inRect(nx, ny, 0.32, 0.62, 0.50, 0.82)) return true;
    if (inRect(nx, ny, 0.50, 0.62, 0.68, 0.82)) return true;
    if (inRect(nx, ny, 0.33, 0.82, 0.49, 1.00)) return true;
    if (inRect(nx, ny, 0.51, 0.82, 0.67, 1.00)) return true;
    return false;
}

const ONBOARDING_STEPS = [
    { id: 'home',     title: 'Home Screen', description: 'Home screen is for showing you your body and points of diseases in your back or front.', tabIcon: 'home-outline'          as const, navSlot: 0 },
    { id: 'reports',  title: 'Reports',     description: 'View detailed AI-generated reports about your skin health and mole analysis history.',    tabIcon: 'document-text-outline' as const, navSlot: 1 },
    { id: 'camera',   title: 'Camera',      description: 'Take a photo of a mole or skin area to get an instant AI-powered skin analysis.',        tabIcon: 'camera-outline'        as const, navSlot: 2 },
    { id: 'history',  title: 'History',     description: 'Track all your past scans and monitor changes in your skin over time.',                   tabIcon: 'time-outline'          as const, navSlot: 3 },
    { id: 'settings', title: 'Settings',    description: 'Manage your profile, notifications, and app preferences here.',                           tabIcon: 'settings-outline'      as const, navSlot: 4 },
];

function getNavX(slot: number): number {
    const s = width / 5;
    if (slot === 0) return s * 0 + s / 2;
    if (slot === 1) return s * 1 + s / 2;
    if (slot === 2) return width / 2;
    if (slot === 3) return s * 3 + s / 2;
    if (slot === 4) return s * 4 + s / 2;
    return width / 2;
}

const NAV_BAR_HEIGHT = 55;

type Mole     = { id: string; x: number; y: number; timestamp: number; photoUri?: string; bodyView: 'front' | 'back'; };
type BodyView = 'front' | 'back';

export default function Nextscreens() {
    const router = useRouter();
    const [userName,   setUserName]   = useState('');
    const [bodyView,   setBodyView]   = useState<BodyView>('front');
    const [moles,      setMoles]      = useState<Mole[]>([]);
    const [activeTab,  setActiveTab]  = useState<string>('Home');

    const [showOnboarding, setShowOnboarding] = useState(false);
    const [onboardingStep, setOnboardingStep] = useState(0);
    const fadeAnim     = useRef(new Animated.Value(0)).current;
    const scaleTooltip = useRef(new Animated.Value(0.85)).current;
    const pulseAnim    = useRef(new Animated.Value(1)).current;
    const pulseLoop    = useRef<Animated.CompositeAnimation | null>(null);

    useEffect(() => { bodyViewRef.current = bodyView; }, [bodyView]);

    const scale        = useRef(new Animated.Value(1)).current;
    const translateX   = useRef(new Animated.Value(0)).current;
    const translateY   = useRef(new Animated.Value(0)).current;
    const scaleVal     = useRef(1);
    const txVal        = useRef(0);
    const tyVal        = useRef(0);
    const bodyViewRef  = useRef<BodyView>('front');
    const bodyWrapperRef = useRef<any>(null);

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
                const doNavigate = (iL: number, iT: number, iW: number, iH: number) => {
                    const cx   = iL + iW / 2;
                    const cy   = iT + iH / 2;
                    const relX = (touch.pageX - txVal.current - cx) / scaleVal.current + iW / 2;
                    const relY = (touch.pageY - tyVal.current - cy) / scaleVal.current + iH / 2;
                    if (checkBodyHit(relX / iW, relY / iH, bodyViewRef.current)) {
                        router.push({ pathname: '/Screensbar/Camera', params: { tapX: relX.toFixed(2), tapY: relY.toFixed(2), bodyView: bodyViewRef.current } });
                    }
                };
                bodyWrapperRef.current?.measure((_fx: number, _fy: number, fw: number, fh: number, px: number, py: number) => {
                    const imgW = width * 0.85;
                    const imgH = height * 0.55;
                    doNavigate(px + (fw - imgW) / 2, py + (fh - imgH) / 2, imgW, imgH);
                });
            }
        },
    })).current;

    useEffect(() => {
        AsyncStorage.getItem(STORAGE_KEY).then(saved => {
            if (saved) {
                const d = JSON.parse(saved);
                setUserName(`${d.firstName || ''} ${d.lastName || ''}`);
            }
        }).catch(() => {});
    }, []);

    // ── FIX: line 205 — cast result to Mole[] ─────────────────
    useEffect(() => {
        const loadMoles = async () => {
            try {
                const data = await loadMolesFromFirestore();
                setMoles(data as Mole[]);
            } catch (err) {
                console.log('Error loading moles:', err);
            }
        };
        loadMoles();
    }, []);

    // ── FIX: line 224 — cast result to Mole[] ─────────────────
    useFocusEffect(
        React.useCallback(() => {
            setActiveTab('Home');
            setOnboardingStep(0);
            setShowOnboarding(false);
            const t = setTimeout(() => {
                setShowOnboarding(true);
                animateIn();
            }, 350);

            loadMolesFromFirestore()
                .then(data => setMoles(data as Mole[]))
                .catch(err => console.log('Focus reload moles error:', err));

            return () => clearTimeout(t);
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

    const handleNext = () => {
        if (onboardingStep < ONBOARDING_STEPS.length - 1) {
            const next = onboardingStep + 1;
            setOnboardingStep(next);
            setTimeout(animateIn, 30);
        } else {
            setShowOnboarding(false);
            router.push('/Screensbar/FirstHomePage');
        }
    };
    const handleSkipAll = () => {
        setShowOnboarding(false);
        router.push('/Screensbar/FirstHomePage');
    };

    const currentMoles = moles.filter(m => m.bodyView === bodyView);

    const deleteMole = async (id: string) => {
        const updated = moles.filter(m => m.id !== id);
        setMoles(updated);
        try {
            await deleteMoleFromFirestore(id);
        } catch (err) {
            console.log('Delete mole error:', err);
            await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(updated));
        }
    };

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

    const handleTabPress = (tabName: string) => {
        setActiveTab(tabName);
        switch (tabName) {
            case 'Camera':   router.push('/Screensbar/Camera');  break;
            case 'History':  router.push('/Screensbar/History'); break;
            case 'Reports':  router.push('/Screensbar/Reports'); break;
            case 'Settings': router.push('/Screensbar/Setting'); break;
        }
    };

    const renderOnboarding = () => {
        if (!showOnboarding) return null;

        const step   = ONBOARDING_STEPS[onboardingStep];
        const isLast = onboardingStep === ONBOARDING_STEPS.length - 1;
        const navX   = getNavX(step.navSlot);
        const spotY = height - NAV_BAR_HEIGHT + (step.navSlot === 2 ? -30 : 6);

        const TW     = 210;
        let   tLeft  = navX - TW / 2;
        tLeft = Math.max(12, Math.min(width - TW - 12, tLeft));
        const tBottom = NAV_BAR_HEIGHT + 50;
        const arrowLeft = Math.max(14, Math.min(TW - 34, navX - tLeft - 14));

        return (
            <View style={[StyleSheet.absoluteFill, ob.root]} pointerEvents="box-none">
                <View style={[StyleSheet.absoluteFill, ob.overlay]} pointerEvents="none" />
                <Animated.View
                    pointerEvents="none"
                    style={[ob.spotlight, {
                        left: navX - 34,
                        top:  spotY - 34,
                        transform: [{ scale: pulseAnim }],
                    }]}
                />
                <Animated.View style={[ob.tooltipWrapper, {
                    bottom: tBottom,
                    left:   tLeft,
                    width:  TW,
                    opacity:   fadeAnim,
                    transform: [{ scale: scaleTooltip }],
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

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            <StatusBar barStyle="dark-content" backgroundColor="#D8E9F0" />

            {/* Header */}
            <View style={styles.headerCard}>
                <View style={styles.headerContent}>
                    <TouchableOpacity style={styles.profileIconContainer}>
                        {/* ── Custom person icon ── */}
                        <Image source={Icons.person} style={styles.headerIconImg} resizeMode="contain" />
                    </TouchableOpacity>
                    <View style={styles.welcomeContainer}>
                        <Text style={styles.welcomeLabel}>Welcome,</Text>
                        <Text style={{ fontWeight: 'bold', marginLeft: 4, marginTop: 3, fontSize: 17 }}>{userName}</Text>
                    </View>
                    <TouchableOpacity style={styles.notificationButton}>
                        {/* ── Custom notification icon ── */}
                        <Image source={Icons.notification} style={styles.notifIconImg} resizeMode="contain" />
                    </TouchableOpacity>
                </View>
            </View>

            {/* Title */}
            <View style={styles.titleContainer}>
                <Text style={styles.title}>Let&#39;s Check your <Text style={styles.titleBold}>Skin</Text></Text>
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
                                        delayLongPress={500}
                                        style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}
                                        onPress={() => router.push({ pathname: '/Screensbar/Camera', params: { tapX: mole.x.toFixed(2), tapY: mole.y.toFixed(2), bodyView: mole.bodyView, moleId: mole.id, existingPhotoUri: mole.photoUri || '' } })}
                                        onLongPress={() => Alert.alert('Delete Point', 'Are you sure?', [
                                            { text: 'Cancel', style: 'cancel' },
                                            { text: 'Delete', style: 'destructive', onPress: () => deleteMole(mole.id) },
                                        ])}
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
                        const isActive = activeTab === tab.name;
                        return (
                            <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                                <View style={[styles.navIcon, isActive && styles.navIconActive]}>
                                    <Image
                                        source={tab.iconImg}
                                        style={styles.navIconImg}
                                        resizeMode="contain"
                                    />
                                </View>
                                <Text style={[styles.navText, isActive && styles.navTextActive]}>{tab.name}</Text>
                            </TouchableOpacity>
                        );
                    })}
                    <View style={styles.navCenterSpacer} />
                    {['History', 'Settings'].map(tabName => {
                        const tab = bottomTabs.find(t => t.name === tabName)!;
                        const isActive = activeTab === tab.name;
                        return (
                            <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                                <View style={[styles.navIcon, isActive && styles.navIconActive]}>
                                    <Image
                                        source={tab.iconImg}
                                        style={styles.navIconImg}
                                        resizeMode="contain"
                                    />
                                </View>
                                <Text style={[styles.navText, isActive && styles.navTextActive]}>{tab.name}</Text>
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

            {renderOnboarding()}

        </SafeAreaView>
    );
}

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
    navIconImg:           { width: 44, height: 44 },
    headerIconImg:        { width: 40, height: 40 },
    notifIconImg:         { width: 36, height: 36 },
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