import { DancingScript_700Bold, useFonts } from '@expo-google-fonts/dancing-script';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useRouter } from 'expo-router';
import { deleteDoc, doc as firestoreDoc, getDoc } from "firebase/firestore";
import React, { useEffect, useRef, useState } from 'react';
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
    BackHandler,
    Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { auth, db } from "../../Firebase/firebaseConfig";
import { FONT_FAMILY_MAP, useCustomize } from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { collection, query, where, onSnapshot } from 'firebase/firestore';
import { useTheme } from '../ThemeContext';

const Icons = {
  home:         require('../../assets/Icons/home.png'),
  reports:      require('../../assets/Icons/Reports.png'),
  history:      require('../../assets/Icons/history.png'),
  settings:     require('../../assets/Icons/setting.png'),
  notification: require('../../assets/Icons/notification.png'),
  person:       require('../../assets/Icons/Account person.png'),
};

const STORAGE_KEY       = 'signupDraft';
const MOLES_STORAGE_KEY = 'savedMoles';
const ONBOARDING_KEY    = 'homeOnboardingSeen';
const { width, height } = Dimensions.get('window');

// ── body image dimensions (same as before) ────────────────────
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

type Mole     = { id: string; x: number; y: number; timestamp: number; photoUri?: string; bodyView: 'front' | 'back'; firestoreId?: string; };
type BodyView = 'front' | 'back';

export default function FirstHomePage() {
    const router = useRouter();
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const [fontsLoaded] = useFonts({ DancingScript_700Bold });

    const customText = {
      fontSize: settings.fontSize,
      color: isDark ? "#FFFFFF" : settings.textColor,
      fontFamily: FONT_FAMILY_MAP[settings.fontFamily],
    };

    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const [userName, setUserName]                         = useState('');
    const [photoUri, setPhotoUri]                         = useState<string | null>(null);
    const [bodyView, setBodyView]                         = useState<BodyView>('front');
    const [moles, setMoles]                               = useState<Mole[]>([]);
    const [activeTab, setActiveTab]                       = useState<string>('Home');
    const [unreadCount, setUnreadCount]                   = useState<number>(0);
    const [notificationsEnabled, setNotificationsEnabled] = useState<boolean>(true);
    const [showOnboarding, setShowOnboarding]             = useState(false);

    // ── Prevent back to startup for logged-in users ─────────
    useFocusEffect(
      React.useCallback(() => {
        const onBackPress = () => {
          const user = auth.currentUser;
          if (user && !user.isAnonymous) {
            BackHandler.exitApp();
            return true;
          }
          return false;
        };
        const subscription = BackHandler.addEventListener('hardwareBackPress', onBackPress);
        return () => subscription.remove();
      }, [])
    );

    useEffect(() => { bodyViewRef.current = bodyView; }, [bodyView]);

    useEffect(() => {
      const checkOnboarding = async () => {
        try {
          const seen = await AsyncStorage.getItem(ONBOARDING_KEY);
          if (!seen) setShowOnboarding(true);
        } catch {}
      };
      checkOnboarding();
    }, []);

    const dismissOnboarding = async () => {
      setShowOnboarding(false);
      try { await AsyncStorage.setItem(ONBOARDING_KEY, 'true'); } catch {}
    };

    const scale      = useRef(new Animated.Value(1)).current;
    const translateX = useRef(new Animated.Value(0)).current;
    const translateY = useRef(new Animated.Value(0)).current;
    const scaleVal   = useRef(1);
    const txVal      = useRef(0);
    const tyVal      = useRef(0);
    const bodyViewRef    = useRef<BodyView>('front');
    const bodyWrapperRef = useRef<any>(null);

    useEffect(() => {
        const s = scale.addListener(({ value })      => { scaleVal.current = value; });
        const x = translateX.addListener(({ value }) => { txVal.current    = value; });
        const y = translateY.addListener(({ value }) => { tyVal.current    = value; });
        return () => { scale.removeListener(s); translateX.removeListener(x); translateY.removeListener(y); };
    }, []);

    const lastDistance = useRef<number | null>(null);
    const isPinching   = useRef(false);
    const tapStartTime = useRef<number>(0);
    const tapStartPos  = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const panStartTx   = useRef(0);
    const panStartTy   = useRef(0);

    // ── FIXED: clamp translation within the scaled container ──
    const clampTranslation = (tx: number, ty: number, sc: number) => {
        const maxX = (BODY_IMG_W  * (sc - 1)) / 2;
        const maxY = (BODY_IMG_H * (sc - 1)) / 2;
        return { x: Math.max(-maxX, Math.min(maxX, tx)), y: Math.max(-maxY, Math.min(maxY, ty)) };
    };

    const panResponder = useRef(PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder:  () => true,
        onPanResponderGrant: (evt) => {
            const touches = evt.nativeEvent.touches;
            isPinching.current = touches.length >= 2;
            if (touches.length === 1) {
                tapStartTime.current = Date.now();
                tapStartPos.current  = { x: touches[0].pageX, y: touches[0].pageY };
                panStartTx.current   = txVal.current;
                panStartTy.current   = tyVal.current;
            } else if (touches.length === 2) {
                const dx = touches[0].pageX - touches[1].pageX;
                const dy = touches[0].pageY - touches[1].pageY;
                lastDistance.current = Math.sqrt(dx * dx + dy * dy);
            }
        },
        onPanResponderMove: (evt) => {
            const touches = evt.nativeEvent.touches;
            if (touches.length === 2) {
                isPinching.current = true;
                const dx      = touches[0].pageX - touches[1].pageX;
                const dy      = touches[0].pageY - touches[1].pageY;
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
            if (touches.length === 1 && scaleVal.current > 1 && !isPinching.current) {
                const dx      = touches[0].pageX - tapStartPos.current.x;
                const dy      = touches[0].pageY - tapStartPos.current.y;
                const clamped = clampTranslation(panStartTx.current + dx, panStartTy.current + dy, scaleVal.current);
                translateX.setValue(clamped.x); translateY.setValue(clamped.y);
                txVal.current = clamped.x; tyVal.current = clamped.y;
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
                // ── FIXED: measure body wrapper to get exact image bounds ──
                if (bodyWrapperRef.current) {
                    bodyWrapperRef.current.measure(
                        (_fx: number, _fy: number, _fw: number, _fh: number, px: number, py: number) => {
                            // The image is centered inside the wrapper
                            const imgLeft   = px;
                            const imgTop    = py;
                            const imgCenterX = imgLeft + BODY_IMG_W / 2;
                            const imgCenterY = imgTop  + BODY_IMG_H / 2;

                            // Reverse the transform applied to the image
                            const touchRelX = (touch.pageX - txVal.current - imgCenterX) / scaleVal.current + BODY_IMG_W / 2;
                            const touchRelY = (touch.pageY - tyVal.current - imgCenterY) / scaleVal.current + BODY_IMG_H / 2;

                            const nx = touchRelX / BODY_IMG_W;
                            const ny = touchRelY / BODY_IMG_H;

                            if (checkBodyHit(nx, ny)) {
                                router.push({
                                    pathname: '/Screensbar/Camera',
                                    params: {
                                        tapX: touchRelX.toFixed(2),
                                        tapY: touchRelY.toFixed(2),
                                        bodyView: bodyViewRef.current,
                                    },
                                });
                            }
                        }
                    );
                }
            }
        },
    })).current;

    // ── Real-time unread notifications listener ───────────────
    useFocusEffect(
        React.useCallback(() => {
            let unsubscribe: () => void = () => {};
            const startListener = async () => {
                const user = auth.currentUser;
                const enabledVal = await AsyncStorage.getItem('notificationsEnabled');
                const isEnabled = enabledVal === null ? true : enabledVal === 'true';
                setNotificationsEnabled(isEnabled);
                if (isEnabled && user) {
                    const q = query(
                        collection(db, 'users', user.uid, 'notifications'),
                        where('isRead', '==', false)
                    );
                    unsubscribe = onSnapshot(q, (snapshot) => {
                        setUnreadCount(snapshot.docs.length);
                    });
                } else {
                    setUnreadCount(0);
                }
            };
            startListener();
            return () => unsubscribe();
        }, [])
    );

    useFocusEffect(
        React.useCallback(() => {
            const loadUserData = async () => {
                try {
                    const saved = await AsyncStorage.getItem(STORAGE_KEY);
                    if (saved) {
                        const data = JSON.parse(saved);
                        setUserName(`${data.firstName || ''} ${data.lastName || ''}`.trim());
                        setPhotoUri(data.photoUri || null);
                    }
                    const currentUser = auth.currentUser;
                    if (currentUser) {
                        const docSnap = await getDoc(firestoreDoc(db, 'users', currentUser.uid));
                        if (docSnap.exists()) {
                            const data = docSnap.data();
                            setUserName(`${data.firstName || ''} ${data.lastName || ''}`.trim());
                        }
                    }
                } catch (err) {
                    console.log('Error loading user data:', err);
                }
            };
            loadUserData();
        }, [])
    );

    useFocusEffect(
        React.useCallback(() => {
            const loadMoles = async () => {
                try {
                    const saved = await AsyncStorage.getItem(MOLES_STORAGE_KEY);
                    if (saved) setMoles(JSON.parse(saved));
                } catch (err) { console.log('Error loading moles:', err); }
            };
            loadMoles();
        }, [])
    );

    const currentMoles = moles.filter((m) => m.bodyView === bodyView);

    const deleteMole = async (moleId: string) => {
        const moleToDelete = moles.find((m) => m.id === moleId);
        const updated = moles.filter((m) => m.id !== moleId);
        setMoles(updated);
        try {
            await AsyncStorage.setItem(MOLES_STORAGE_KEY, JSON.stringify(updated));
        } catch (err) { console.log('Error deleting mole from AsyncStorage:', err); }
        try {
            const user = auth.currentUser;
            if (user && moleToDelete?.firestoreId) {
                await deleteDoc(firestoreDoc(db, 'users', user.uid, 'scans', moleToDelete.firestoreId));
            }
        } catch (err) { console.log('Error deleting mole from Firestore:', err); }
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
            case 'Camera':   router.push('/Screensbar/Camera');           break;
            case 'History':  router.push('/Screensbar/History');          break;
            case 'Reports':  router.push('/Screensbar/Reports');          break;
            case 'Settings': router.push('/Screensbar/Setting');          break;
        }
    };

    return (
      <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
        <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

        {/* ── Header ── */}
        <View style={[styles.headerCard, { backgroundColor: colors.card }]}>
          <View style={[styles.headerContent, { flexDirection: isArabic ? "row-reverse" : "row" }]}>
            <TouchableOpacity
              style={[styles.profileIconContainer, { backgroundColor: "#E8F4F8", borderColor: isDark ? "#00A3A3" : "#C5E3ED" }]}
              onPress={() => router.push("/Settingsoptions/Editprofile")}
            >
              {photoUri ? (
                <Image source={{ uri: photoUri }} style={styles.profilePhoto} resizeMode="cover" />
              ) : (
                <Image source={Icons.person} style={styles.headerIconImg} resizeMode="contain" />
              )}
            </TouchableOpacity>

            {/* ── FIXED: Welcome + name on same row, same baseline ── */}
            <View style={[
              styles.welcomeContainer,
              {
                marginLeft: isArabic ? 0 : 12,
                marginRight: isArabic ? 12 : 0,
                flexDirection: isArabic ? "row-reverse" : "row",
                alignItems: "center",
              }
            ]}>
              <Text style={[
                styles.welcomeLabel,
                { color: "#00A3A3", fontFamily: fontsLoaded ? "DancingScript_700Bold" : undefined }
              ]}>
                {isArabic ? "أهلاً،" : "Welcome,"}
              </Text>
              <Text style={[
                styles.userName,
                customText,
                { marginLeft: isArabic ? 0 : 5, marginRight: isArabic ? 5 : 0 }
              ]}>
                {userName}
              </Text>
            </View>

            <TouchableOpacity
              style={[styles.notificationButton, { backgroundColor: isDark ? "#1E2A35" : "#F9FAFB" }]}
              onPress={() => router.push("/Screensbar/Notifications")}
            >
              <View style={{ position: 'relative' }}>
                <Image
                  source={Icons.notification}
                  style={[
                    styles.notifIconImg,
                    !notificationsEnabled && { tintColor: '#9CA3AF', opacity: 0.6 }
                  ]}
                  resizeMode="contain"
                />
                {!notificationsEnabled && <View style={styles.disabledLine} />}
                {notificationsEnabled && unreadCount > 0 && (
                  <View style={styles.notifBadge}>
                    <Text style={styles.notifBadgeText}>{unreadCount > 99 ? "99+" : unreadCount}</Text>
                  </View>
                )}
              </View>
            </TouchableOpacity>
          </View>
        </View>

        {/* ── Title ── */}
        <View style={styles.titleContainer}>
          <Text style={[styles.title, customText]}>
            <Text>{isArabic ? "دعنا نفحص " : "Let's Check your "}</Text>
            <Text style={[{ fontFamily: FONT_FAMILY_MAP[settings.fontFamily] }, styles.titleBold, { color: "#00A3A3" }]}>
              {isArabic ? "جلدك" : "Skin"}
            </Text>
          </Text>
        </View>

        {/* ── FIXED: Body map — zoom is contained, moles placed correctly ── */}
        <View style={[styles.bodyMainContainer, { backgroundColor: pageBg }]}>
          {/* outer view clips the zoomed content */}
          <View
            style={[styles.bodyClipWrapper, { backgroundColor: pageBg }]}
            {...panResponder.panHandlers}
            ref={(r) => { bodyWrapperRef.current = r; }}
          >
            <Animated.View
              style={[
                styles.bodyImageWrapper,
                { backgroundColor: pageBg, transform: [{ scale }, { translateX }, { translateY }] },
              ]}
            >
              <Image
                source={bodyView === "front"
                  ? require("../../assets/images/body-front.png")
                  : require("../../assets/images/body-back.png")}
                style={[styles.bodyImage, { backgroundColor: pageBg }]}
                resizeMode="contain"
              />

              {/* ── FIXED: moles placed at exact (x,y) within the image ── */}
              {currentMoles.map((mole) => {
                const MARKER_SIZE = 28;
                return (
                  <View
                    key={mole.id}
                    style={[
                      styles.moleContainer,
                      {
                        left: mole.x - MARKER_SIZE / 2,
                        top:  mole.y - MARKER_SIZE / 2,
                      },
                    ]}
                    pointerEvents="box-none"
                  >
                    <TouchableOpacity
                      activeOpacity={0.8}
                      delayLongPress={500}
                      style={{ flexDirection: "row", alignItems: "center", gap: 4 }}
                      onPress={() =>
                        router.push({
                          pathname: "/Screensbar/Camera",
                          params: {
                            tapX: mole.x.toFixed(2),
                            tapY: mole.y.toFixed(2),
                            bodyView: mole.bodyView,
                            moleId: mole.id,
                            existingPhotoUri: mole.photoUri || "",
                            firestoreId: mole.firestoreId || "",
                          },
                        })
                      }
                      onLongPress={() =>
                        Alert.alert("Delete Point", "Are you sure?", [
                          { text: "Cancel", style: "cancel" },
                          { text: "Delete", style: "destructive", onPress: () => deleteMole(mole.id) },
                        ])
                      }
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

        {/* ── Toggle Front/Back ── */}
        <View style={styles.bottomControls}>
          <View style={[styles.toggleWrapper, { backgroundColor: isDark ? "#1E2A35" : "#B8D4DE" }]}>
            <TouchableOpacity
              onPress={() => toggleBodyView("front")}
              style={[styles.toggleButton, bodyView === "front" && styles.toggleButtonActive]}
            >
              <Text style={[styles.toggleText, { color: bodyView === "front" ? "#FFFFFF" : colors.subText }]}>
                {isArabic ? "أمامي" : "Front"}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => toggleBodyView("back")}
              style={[styles.toggleButton, bodyView === "back" && styles.toggleButtonActive]}
            >
              <Text style={[styles.toggleText, { color: bodyView === "back" ? "#FFFFFF" : colors.subText }]}>
                {isArabic ? "خلفي" : "Back"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* ── FIXED: Floating bottom nav (not stuck to edge) ── */}
        <View style={styles.bottomNavContainer}>
          <View style={[
            styles.bottomNav,
            {
              backgroundColor: colors.navBg,
              borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
            }
          ]}>
            {["Home", "Reports"].map((tabName) => {
              const tab = bottomTabs.find((t) => t.name === tabName)!;
              const isActive = activeTab === tab.name;
              return (
                <TouchableOpacity key={tab.name} style={styles.navItem} onPress={() => handleTabPress(tab.name)}>
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                    isActive && { backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8", borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#C5E3ED" },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[
                    styles.navText,
                    { color: isActive ? colors.navActive : colors.navText },
                    isActive && { fontWeight: "700" },
                  ]}>
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
                  <View style={[
                    styles.navIcon,
                    { backgroundColor: isDark ? "#152030" : "#F9FAFB" },
                    isActive && { backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8", borderWidth: 2, borderColor: isDark ? "#00A3A3" : "#C5E3ED" },
                  ]}>
                    <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
                  </View>
                  <Text style={[
                    styles.navText,
                    { color: isActive ? colors.navActive : colors.navText },
                    isActive && { fontWeight: "700" },
                  ]}>
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
              { backgroundColor: colors.navBg, borderColor: isDark ? "#374151" : "#C5E3ED" },
              activeTab === "Camera" && { borderColor: colors.navActive, backgroundColor: isDark ? "#1E3A4A" : "#E8F4F8" },
            ]}
            onPress={() => handleTabPress("Camera")}
            activeOpacity={0.85}
          >
            <Ionicons name="camera-outline" size={30} color={activeTab === "Camera" ? colors.navActive : colors.navText} />
          </TouchableOpacity>
        </View>

        {/* ── Onboarding Modal ── */}
        {showOnboarding && (
          <View style={styles.onboardingOverlay}>
            <View style={[styles.onboardingBox, { backgroundColor: colors.card }]}>
              <Ionicons name="hand-left-outline" size={40} color="#004F7F" style={{ marginBottom: 12 }} />
              <Text style={[styles.onboardingTitle, { color: isDark ? '#fff' : '#1F2937' }]}>Tips</Text>
              <View style={styles.onboardingRow}>
                <Ionicons name="search-outline" size={20} color="#00A3A3" />
                <Text style={[styles.onboardingText, { color: isDark ? '#ccc' : '#374151' }]}>
                  You can pinch to zoom in/out on the body map.
                </Text>
              </View>
              <View style={styles.onboardingRow}>
                <Ionicons name="time-outline" size={20} color="#00A3A3" />
                <Text style={[styles.onboardingText, { color: isDark ? '#ccc' : '#374151' }]}>
                  Long press (2 seconds) on a point to delete it.
                </Text>
              </View>
              <TouchableOpacity style={styles.onboardingBtn} onPress={dismissOnboarding} activeOpacity={0.85}>
                <Text style={styles.onboardingBtnText}>Got it!</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:            { flex: 1 },
    titleContainer:       { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 4 },
    title:                { fontSize: 20, textAlign: 'center' },
    titleBold:            { fontWeight: '700' },

    // ── FIXED: body area clips overflow so zoom stays inside ──
    bodyMainContainer:    { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 16, marginBottom: 110 },
    bodyClipWrapper: {
        width: BODY_IMG_W,
        height: BODY_IMG_H,
        overflow: 'hidden',   // ← clips the zoomed image within bounds
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 12,
    },
    bodyImageWrapper: {
        width: BODY_IMG_W,
        height: BODY_IMG_H,
        alignItems: 'center',
        justifyContent: 'center',
        // no overflow:hidden here — overflow is handled by the clip wrapper
    },
    bodyImage:            { width: '100%', height: '100%' },

    // ── FIXED: mole positioned absolutely inside bodyImageWrapper ──
    moleContainer:        { position: 'absolute', flexDirection: 'row', alignItems: 'center' },
    moleInner:            { width: 28, height: 28, borderRadius: 14, backgroundColor: '#004F7F', alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#FFFFFF' },
    moleIcon:             { color: '#FFFFFF', fontSize: 18, fontWeight: '700', lineHeight: 22 },
    moleThumbnail:        { width: 38, height: 38, borderRadius: 8, borderWidth: 2, borderColor: '#FFFFFF', backgroundColor: '#ccc' },

    bottomControls:       { position: 'absolute', bottom: 110, left: 0, right: 0, alignItems: 'center' },
    toggleWrapper:        { flexDirection: 'row', borderRadius: 25, padding: 4, width: width * 0.45 },
    toggleButton:         { flex: 1, paddingVertical: 8, alignItems: 'center', justifyContent: 'center', borderRadius: 20 },
    toggleButtonActive:   { backgroundColor: '#004F7F' },
    toggleText:           { fontSize: 14, fontWeight: '600' },

    // ── FIXED: floating bottom nav ──
    bottomNavContainer: {
        position: 'absolute',
        bottom: 16,              // ← lifted off the edge
        left: 16,                // ← side margins so it floats
        right: 16,
        alignItems: 'center',
    },
    bottomNav: {
        flexDirection: 'row',
        paddingVertical: 10,
        paddingBottom: 14,
        borderRadius: 28,        // ← fully rounded pill shape
        borderWidth: 1,
        width: '100%',
        // shadow for floating effect
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.12,
        shadowRadius: 12,
        elevation: 8,
    },
    navCenterSpacer:      { flex: 1 },
    navItem:              { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navIcon:              { width: 42, height: 42, borderRadius: 22, justifyContent: 'center', alignItems: 'center', marginBottom: 4 },
    navIconImg:           { width: 42, height: 42 },
    notifIconImg:         { width: 56, height: 56 },
    headerIconImg:        { width: 45, height: 45 },
    navText:              { fontSize: 11, fontWeight: '500' },
    cameraButton: {
        position: 'absolute',
        top: -26,
        alignSelf: 'center',
        width: 60,
        height: 60,
        borderRadius: 30,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        elevation: 8,
    },

    // ── Header ──
    headerCard: {
        marginHorizontal: 16, marginTop: 12, marginBottom: 8,
        borderRadius: 20, paddingVertical: 14, paddingHorizontal: 16,
        shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
    },
    headerContent:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    profileIconContainer: { width: 46, height: 46, borderRadius: 26, justifyContent: 'center', alignItems: 'center', borderWidth: 2, overflow: 'hidden' },
    profilePhoto:         { width: 52, height: 52, borderRadius: 26 },

    // ── FIXED: welcome + name on same line, same vertical center ──
    welcomeContainer: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        flexWrap: 'nowrap',
    },
    welcomeLabel:         { fontSize: 18 },
    userName:             { fontSize: 16, fontWeight: '700', flexShrink: 1 },

    notificationButton:   { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
    notifBadge: {
        position: 'absolute', top: 4, right: 4,
        backgroundColor: '#EF4444', borderRadius: 9,
        minWidth: 18, height: 18,
        alignItems: 'center', justifyContent: 'center',
        paddingHorizontal: 4, borderWidth: 1.5, borderColor: '#FFFFFF',
    },
    notifBadgeText:       { color: '#FFFFFF', fontSize: 10, fontWeight: '800', lineHeight: 13 },
    disabledLine: {
        position: 'absolute', top: '45%', left: '34%',
        width: '60%', height: 2,
        backgroundColor: '#9CA3AF',
        transform: [{ rotate: '45deg' }],
        borderRadius: 1,
    },

    // ── Onboarding ──
    onboardingOverlay:    { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'center', alignItems: 'center', zIndex: 9999 },
    onboardingBox:        { width: '80%', borderRadius: 20, padding: 24, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.2, shadowRadius: 12, elevation: 10 },
    onboardingTitle:      { fontSize: 18, fontWeight: '700', marginBottom: 16 },
    onboardingRow:        { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 12, width: '100%' },
    onboardingText:       { flex: 1, fontSize: 14, lineHeight: 20 },
    onboardingBtn:        { marginTop: 8, backgroundColor: '#004F7F', borderRadius: 12, paddingVertical: 12, paddingHorizontal: 32 },
    onboardingBtnText:    { color: '#fff', fontWeight: '700', fontSize: 15 },
});