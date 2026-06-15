// app/Guest/GuestBottomNav.tsx
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTheme } from '../ThemeContext';

const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

export const GUEST_FREE_SCAN_KEY = 'guestFreeScanUsed';

export type GuestTab = 'Home' | 'Reports' | 'History' | 'Settings' | 'Camera';

interface Props {
  activeTab: GuestTab;
  onCameraPress?: () => void;
}

const TAB_ROUTES: Record<string, string> = {
  Home:     '/Guest/Guest',
  Reports:  '/Guest/reportguest',
  History:  '/Guest/histroyguest',
  Settings: '/Guest/settingsguest',
};

const bottomTabs = [
  { name: 'Home',     iconImg: Icons.home     },
  { name: 'Reports',  iconImg: Icons.reports  },
  { name: 'History',  iconImg: Icons.history  },
  { name: 'Settings', iconImg: Icons.settings },
];

export default function GuestBottomNav({ activeTab, onCameraPress }: Props) {
  const router = useRouter();
  const { isDark, colors } = useTheme();
  const [freeScanUsed, setFreeScanUsed] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(GUEST_FREE_SCAN_KEY).then(val => {
      setFreeScanUsed(val === 'true');
    });
  }, []);

  const scaleAnims = useRef<Record<string, Animated.Value>>({
    Home:     new Animated.Value(1),
    Reports:  new Animated.Value(1),
    History:  new Animated.Value(1),
    Settings: new Animated.Value(1),
    Camera:   new Animated.Value(1),
  }).current;

  const animatePress = (tabName: string) => {
    Animated.sequence([
      Animated.timing(scaleAnims[tabName], { toValue: 0.88, duration: 80, useNativeDriver: true }),
      Animated.spring(scaleAnims[tabName], { toValue: 1, tension: 200, friction: 7, useNativeDriver: true }),
    ]).start();
  };

const handleTabPress = (tabName: string) => {
  animatePress(tabName);
  if (tabName === 'Camera') {
    onCameraPress?.();
    return;
  }
  if (tabName === activeTab) return;
  
  // بدل router.replace استخدم:
  router.navigate(TAB_ROUTES[tabName] as any);
};

  return (
    <View style={styles.bottomNavContainer}>
      <View
        style={[
          styles.bottomNav,
          {
            backgroundColor: colors.navBg,
            borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
          },
        ]}
      >
        {['Home', 'Reports'].map((tabName) => {
          const tab      = bottomTabs.find((t) => t.name === tabName)!;
          const isActive = activeTab === tab.name;
          return (
            <TouchableOpacity
              key={tab.name}
              style={styles.navItem}
              onPress={() => handleTabPress(tab.name)}
              activeOpacity={1}
            >
              <Animated.View
                style={[
                  styles.navIcon,
                  { backgroundColor: isDark ? '#152030' : '#F9FAFB' },
                  isActive && {
                    backgroundColor: isDark ? '#1E3A4A' : '#E8F4F8',
                    borderWidth: 2,
                    borderColor: isDark ? '#00A3A3' : '#C5E3ED',
                  },
                  { transform: [{ scale: scaleAnims[tab.name] }] },
                ]}
              >
                <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
              </Animated.View>
              <Text
                style={[
                  styles.navText,
                  { color: isActive ? colors.navActive : isDark ? '#FFFFFF' : '#6B7280' },
                  isActive && { fontWeight: '700' },
                ]}
              >
                {tab.name}
              </Text>
            </TouchableOpacity>
          );
        })}

        <View style={styles.navCenterSpacer} />

        {['History', 'Settings'].map((tabName) => {
          const tab      = bottomTabs.find((t) => t.name === tabName)!;
          const isActive = activeTab === tab.name;
          return (
            <TouchableOpacity
              key={tab.name}
              style={styles.navItem}
              onPress={() => handleTabPress(tab.name)}
              activeOpacity={1}
            >
              <Animated.View
                style={[
                  styles.navIcon,
                  { backgroundColor: isDark ? '#152030' : '#F9FAFB' },
                  isActive && {
                    backgroundColor: isDark ? '#1E3A4A' : '#E8F4F8',
                    borderWidth: 2,
                    borderColor: isDark ? '#00A3A3' : '#C5E3ED',
                  },
                  { transform: [{ scale: scaleAnims[tab.name] }] },
                ]}
              >
                <Image source={tab.iconImg} style={styles.navIconImg} resizeMode="contain" />
              </Animated.View>
              <Text
                style={[
                  styles.navText,
                  { color: isActive ? colors.navActive : isDark ? '#FFFFFF' : '#6B7280' },
                  isActive && { fontWeight: '700' },
                ]}
              >
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
            borderColor: freeScanUsed
              ? '#EF4444'
              : activeTab === 'Camera'
                ? '#004F7F'
                : isDark ? '#374151' : '#C5E3ED',
          },
          activeTab === 'Camera' && {
            backgroundColor: isDark ? '#1A3A4A' : '#E8F4F8',
          },
        ]}
        onPress={() => handleTabPress('Camera')}
        activeOpacity={0.85}
      >
        <Animated.View style={{ transform: [{ scale: scaleAnims['Camera'] }] }}>
          <Ionicons
            name={freeScanUsed ? 'lock-closed-outline' : 'camera-outline'}
            size={30}
            color={
              freeScanUsed
                ? '#EF4444'
                : activeTab === 'Camera'
                  ? '#004F7F'
                  : isDark ? '#FFFFFF' : '#6B7280'
            }
          />
        </Animated.View>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  bottomNavContainer: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    alignItems: 'center',
  },
  bottomNav: {
    flexDirection: 'row',
    paddingVertical: 10,
    paddingBottom: 14,
    borderRadius: 28,
    borderWidth: 1,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 8,
  },
  navCenterSpacer: { flex: 1 },
  navItem:         { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navIcon: {
    width: 42,
    height: 42,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 4,
  },
  navIconImg: { width: 42, height: 42 },
  navText:    { fontSize: 11, fontWeight: '500' },
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
});