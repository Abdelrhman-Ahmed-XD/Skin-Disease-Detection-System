import React from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../ThemeContext';

const Icons = {
  home:     require('../../assets/Icons/home.png'),
  reports:  require('../../assets/Icons/Reports.png'),
  history:  require('../../assets/Icons/history.png'),
  settings: require('../../assets/Icons/setting.png'),
};

export type NavTab = 'Home' | 'Reports' | 'History' | 'Settings' | 'Camera';

interface Props {
  activeTab: NavTab;
  onTabPress: (tab: NavTab) => void;
  fontFamily?: string;
  tabLabels?: Partial<Record<NavTab, string>>;
}

const LEFT_TABS:  { name: NavTab; iconImg: any }[] = [
  { name: 'Home',    iconImg: Icons.home    },
  { name: 'Reports', iconImg: Icons.reports },
];
const RIGHT_TABS: { name: NavTab; iconImg: any }[] = [
  { name: 'History',  iconImg: Icons.history  },
  { name: 'Settings', iconImg: Icons.settings },
];

export default function FloatingNavBar({
  activeTab,
  onTabPress,
  fontFamily,
  tabLabels,
}: Props) {
  const { colors, isDark } = useTheme();

  const label = (name: NavTab) => tabLabels?.[name] ?? name;

  const renderTab = (tab: { name: NavTab; iconImg: any }) => {
    const isActive = activeTab === tab.name;
    return (
      <TouchableOpacity
        key={tab.name}
        style={styles.navItem}
        onPress={() => onTabPress(tab.name)}
        activeOpacity={0.75}
      >
        <View
          style={[
            styles.navIcon,
            { backgroundColor: isDark ? '#152030' : '#F9FAFB' },
            isActive && {
              backgroundColor: isDark ? '#1E3A4A' : '#E8F4F8',
              borderWidth: 2,
              borderColor: isDark ? '#00A3A3' : '#C5E3ED',
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
              fontFamily,
              color: isActive ? colors.navActive : colors.navText,
            },
            isActive && { fontWeight: '700' },
          ]}
        >
          {label(tab.name)}
        </Text>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container} pointerEvents="box-none">
      {/* الـ pill */}
      <View
        style={[
          styles.bar,
          {
            backgroundColor: colors.navBg,
            borderColor: isDark
              ? 'rgba(255,255,255,0.08)'
              : 'rgba(0,0,0,0.06)',
          },
        ]}
      >
        {LEFT_TABS.map(renderTab)}
        <View style={styles.spacer} />
        {RIGHT_TABS.map(renderTab)}
      </View>

      {/* Camera FAB */}
      <TouchableOpacity
        style={[
          styles.fab,
          {
            backgroundColor: colors.navBg,
            borderColor: isDark ? '#374151' : '#C5E3ED',
          },
          activeTab === 'Camera' && {
            borderColor: colors.navActive,
            backgroundColor: isDark ? '#1E3A4A' : '#E8F4F8',
          },
        ]}
        onPress={() => onTabPress('Camera')}
        activeOpacity={0.85}
      >
        <Ionicons
          name="camera-outline"
          size={30}
          color={
            activeTab === 'Camera' ? colors.navActive : colors.navText
          }
        />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    alignItems: 'center',
  },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingBottom: Platform.OS === 'ios' ? 20 : 14,
    borderRadius: 28,
    borderWidth: 1,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 8,
  },
  spacer:     { flex: 1 },
  navItem:    { flex: 1, alignItems: 'center', justifyContent: 'center' },
  navIcon: {
    width: 42,
    height: 42,
    borderRadius: 21,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 3,
  },
  navIconImg: { width: 38, height: 38 },
  navText:    { fontSize: 11, fontWeight: '500' },
  fab: {
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