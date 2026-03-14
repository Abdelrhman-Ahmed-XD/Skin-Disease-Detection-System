import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
  Alert, LayoutChangeEvent, PanResponder, ScrollView,
  StatusBar, StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  CustomizeSettings, DEFAULT_SETTINGS, FontFamily, Language, useCustomize,
} from '../Customize/Customizecontext';
import { useTranslation } from '../Customize/translations';
import { useTheme } from '../ThemeContext';

const LANGUAGES: Language[] = ['English', 'Arabic'];
const FONT_FAMILIES: FontFamily[] = ['System', 'Inter', 'SpaceMono', 'Roboto'];

const TEXT_COLORS = [
  { label: 'Black', value: '#1F2937' },
  { label: 'Red',   value: '#F87171' },
  { label: 'Green', value: '#86EFAC' },
  { label: 'Blue',  value: '#7DD3FC' },
  { label: 'White', value: '#FFFFFF' },
];

const BG_COLORS = [
  { label: 'Default', value: '#D8E9F0' },
  { label: 'Red',     value: '#FECACA' },
  { label: 'Green',   value: '#BBF7D0' },
  { label: 'Blue',    value: '#BAE6FD' },
  { label: 'White',   value: '#FFFFFF' },
];

const MIN_FONT = 12;
const MAX_FONT = 36;

function FontSlider({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const trackWidth = useRef(0);
  const [trackW, setTrackW] = useState(0);
  const THUMB_SIZE = 28;
  const clamp = (v: number) => Math.min(MAX_FONT, Math.max(MIN_FONT, v));
  const xToValue = (x: number) => {
    const w = trackWidth.current;
    if (w === 0) return value;
    return Math.round(MIN_FONT + Math.min(1, Math.max(0, x / w)) * (MAX_FONT - MIN_FONT));
  };
  const panResponder = useRef(PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderGrant: (evt) => onChange(clamp(xToValue(evt.nativeEvent.locationX))),
    onPanResponderMove: (evt) => onChange(clamp(xToValue(evt.nativeEvent.locationX))),
  })).current;
  const progress = (value - MIN_FONT) / (MAX_FONT - MIN_FONT);
  const fillWidth = trackW * progress;
  const thumbLeft = trackW * progress - THUMB_SIZE / 2;
  return (
    <View style={sliderSt.wrapper}>
      <View
        style={sliderSt.trackContainer}
        onLayout={(e: LayoutChangeEvent) => {
          trackWidth.current = e.nativeEvent.layout.width;
          setTrackW(e.nativeEvent.layout.width);
        }}
        {...panResponder.panHandlers}
      >
        <View style={sliderSt.trackBg} />
        <View style={[sliderSt.trackFill, { width: fillWidth }]} />
        <View style={[sliderSt.thumb, { left: thumbLeft }]}>
          <Text style={sliderSt.thumbLabel}>{value}</Text>
        </View>
      </View>
      <View style={sliderSt.labels}>
        <Text style={sliderSt.labelText}>{MIN_FONT}</Text>
        <Text style={sliderSt.labelText}>{MAX_FONT}</Text>
      </View>
    </View>
  );
}

const sliderSt = StyleSheet.create({
  wrapper:        { width: '100%', paddingTop: 18, paddingBottom: 4 },
  trackContainer: { width: '100%', height: 30, justifyContent: 'center', position: 'relative' },
  trackBg:        { position: 'absolute', left: 0, right: 0, height: 4, borderRadius: 2, backgroundColor: '#C5E3ED' },
  trackFill:      { position: 'absolute', left: 0, height: 4, borderRadius: 2, backgroundColor: '#004F7F' },
  thumb:          { position: 'absolute', width: 28, height: 28, borderRadius: 14, backgroundColor: '#004F7F', marginLeft: -14, alignItems: 'center', justifyContent: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 4, elevation: 4 },
  thumbLabel:     { color: '#FFFFFF', fontSize: 10, fontWeight: '800' },
  labels:         { flexDirection: 'row', justifyContent: 'space-between', marginTop: 4 },
  labelText:      { fontSize: 11, color: '#9CA3AF' },
});

function Dropdown<T extends string>({ options, value, onChange, colors, customText }: { options: T[]; value: T; onChange: (v: T) => void; colors: any; customText: any }) {
  const [open, setOpen] = useState(false);
  return (
    <View style={dd.wrapper}>
      <TouchableOpacity style={[dd.button, { backgroundColor: colors.primary }]} onPress={() => setOpen((o) => !o)} activeOpacity={0.85}>
        <Text style={[dd.buttonText, { fontFamily: customText.fontFamily }]}>{value}</Text>
        <Ionicons name={open ? 'chevron-up' : 'chevron-down'} size={16} color="#FFFFFF" />
      </TouchableOpacity>
      {open && (
        <View style={[dd.menu, { backgroundColor: colors.card, borderColor: colors.border }]}>
          {options.map((opt) => (
            <TouchableOpacity
              key={opt}
              style={[dd.menuItem, opt === value && { backgroundColor: colors.border }]}
              onPress={() => { onChange(opt); setOpen(false); }}
            >
              <Text style={[dd.menuText, customText, opt === value && { color: colors.primary, fontWeight: '700' }]}>
                {opt}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </View>
  );
}

const dd = StyleSheet.create({
  wrapper:    { position: 'relative', zIndex: 9999, elevation: 9999 },
  button:     { flexDirection: 'row', alignItems: 'center', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8, gap: 8, minWidth: 130 },
  buttonText: { color: '#FFFFFF', fontWeight: '600', fontSize: 14, flex: 1, textAlign: 'center' },
  menu:       { position: 'absolute', top: 42, right: 0, borderRadius: 12, borderWidth: 1, paddingVertical: 4, minWidth: 140, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 8, elevation: 9999, zIndex: 9999 },
  menuItem:   { paddingHorizontal: 16, paddingVertical: 10 },
  menuText:   { fontSize: 14, fontWeight: '500' },
});

function ColorSwatch({ color, selected, onPress, dark }: { color: string; selected: boolean; onPress: () => void; dark?: boolean }) {
  return (
    <TouchableOpacity onPress={onPress} style={[sw.swatch, { backgroundColor: color }, color === '#FFFFFF' && sw.swatchBorder]} activeOpacity={0.8}>
      {selected && <Ionicons name="checkmark" size={16} color={dark || color === '#1F2937' ? '#FFFFFF' : '#1F2937'} />}
    </TouchableOpacity>
  );
}
const sw = StyleSheet.create({
  swatch:       { width: 28, height: 28, borderRadius: 7, alignItems: 'center', justifyContent: 'center' },
  swatchBorder: { borderWidth: 1.5, borderColor: '#D1D5DB' },
});

export default function CustomizePage() {
  const { colors, isDark } = useTheme();
  const { settings, saveSettings } = useCustomize();
  const [draft, setDraft] = useState<CustomizeSettings>({ ...settings });
  const { t, isArabic } = useTranslation(draft.language);

  const customText = {
    fontSize: settings.fontSize,
    color: isDark ? "#FFFFFF" : settings.textColor,
    fontFamily:
      settings.fontFamily === "System" ? undefined : settings.fontFamily,
  };

  const pageBg = isDark ? colors.background : settings.backgroundColor;

  const handleConfirm = async () => {
    await saveSettings(draft);
    Alert.alert(t('customizeSaved'), t('customizeSavedMsg'), [
      { text: t('ok'), onPress: () => router.back() },
    ]);
  };

  const previewFont = draft.fontFamily === 'System' ? undefined : draft.fontFamily;

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={['top']}>
      <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.card }]}>
        <TouchableOpacity style={[styles.backButton, { borderColor: colors.border }]} onPress={() => router.back()}>
          <Ionicons name={isArabic ? 'chevron-back' : 'chevron-back'} size={24} color={colors.text} />
        </TouchableOpacity>
        <View style={styles.headerTitleRow}>
          <Text style={[styles.headerTitle, customText]}>{t('customize')}</Text>
        </View>
        <TouchableOpacity style={[styles.resetButton, { borderColor: colors.border }]} onPress={() => setDraft({ ...DEFAULT_SETTINGS })}>
          <Ionicons name="refresh-outline" size={18} color={colors.subText} />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        nestedScrollEnabled={true}
      >
        {/* Live Preview */}
        <View style={[styles.previewCard, { backgroundColor: draft.backgroundColor }]}>
          <Text style={[styles.previewLabel, { color: draft.textColor, fontSize: 11, fontFamily: previewFont }]}>{t('preview')}</Text>
          <Text style={{ fontSize: draft.fontSize, color: draft.textColor, fontFamily: previewFont, fontWeight: '500', textAlign: 'center', marginTop: 6 }}>
            {t('previewText')}
          </Text>
        </View>

        {/* Settings Card */}
        <View style={[styles.card, { backgroundColor: colors.card }]}>

          {/* Language - zIndex عالي */}
          <View style={[styles.row, styles.rowBorder, {
            borderBottomColor: colors.border,
            flexDirection: isArabic ? 'row-reverse' : 'row',
            zIndex: 999,
            elevation: 999,
          }]}>
            <View style={[styles.labelRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
              <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1A3040' : '#EAF4FB' }]}>
                <Ionicons name="language-outline" size={20} color={colors.primary} />
              </View>
              <Text style={[styles.label, customText]}>{t('language')}</Text>
            </View>
            <Dropdown<Language>
              options={LANGUAGES}
              value={draft.language}
              onChange={(v) => setDraft((d) => ({ ...d, language: v }))}
              colors={colors}
              customText={customText}
            />
          </View>

          {/* Font Type - zIndex أقل */}
          <View style={[styles.row, styles.rowBorder, {
            borderBottomColor: colors.border,
            flexDirection: isArabic ? 'row-reverse' : 'row',
            zIndex: 998,
            elevation: 998,
          }]}>
            <View style={[styles.labelRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
              <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1A3040' : '#EAF4FB' }]}>
                <Text style={[styles.iconText, customText, { color: colors.primary }]}>T</Text>
              </View>
              <Text style={[styles.label, customText]}>{t('fontType')}</Text>
            </View>
            <Dropdown<FontFamily>
              options={FONT_FAMILIES}
              value={draft.fontFamily}
              onChange={(v) => setDraft((d) => ({ ...d, fontFamily: v }))}
              colors={colors}
              customText={customText}
            />
          </View>

          {/* Font Size */}
          <View style={[styles.row, styles.rowBorder, {
            borderBottomColor: colors.border,
            zIndex: 1,
            elevation: 1,
          }]}>
            <View style={styles.fullWidth}>
              <View style={[styles.labelRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1A3040' : '#EAF4FB' }]}>
                  <Text style={[styles.iconTextSm, customText, { color: colors.primary }]}>Aa</Text>
                </View>
                <Text style={[styles.label, customText]}>{t('fontSize')}</Text>
              </View>
              <FontSlider value={draft.fontSize} onChange={(v) => setDraft((d) => ({ ...d, fontSize: v }))} />
            </View>
          </View>

          {/* Text Color */}
          <View style={[styles.colorBlock, styles.rowBorder, {
            borderBottomColor: colors.border,
            zIndex: 1,
            elevation: 1,
          }]}>
            <View style={[styles.labelRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
              <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1A3040' : '#EAF4FB' }]}>
                <Ionicons name="brush-outline" size={18} color={colors.primary} />
              </View>
              <Text style={[styles.label, customText]}>{t('textColor')}</Text>
            </View>
            <View style={[styles.swatchRow, { marginLeft: 44, marginTop: 10 }]}>
              {TEXT_COLORS.map((c) => (
                <ColorSwatch key={c.value} color={c.value} selected={draft.textColor === c.value} onPress={() => setDraft((d) => ({ ...d, textColor: c.value, textColorCustomized: true }))} dark={c.value === '#1F2937'} />
              ))}
            </View>
          </View>

          {/* Background Color */}
          <View style={[styles.colorBlock, { zIndex: 1, elevation: 1 }]}>
            <View style={[styles.labelRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
              <View style={[styles.iconWrap, { backgroundColor: isDark ? '#1A3040' : '#EAF4FB' }]}>
                <Ionicons name="color-fill-outline" size={18} color={colors.primary} />
              </View>
              <Text style={[styles.label, customText]}>{t('backgroundColor')}</Text>
            </View>
            <View style={[styles.swatchRow, { marginLeft: 44, marginTop: 10 }]}>
              {BG_COLORS.map((c) => (
                <ColorSwatch key={c.value} color={c.value} selected={draft.backgroundColor === c.value} onPress={() => setDraft((d) => ({ ...d, backgroundColor: c.value }))} />
              ))}
            </View>
          </View>

        </View>

        {/* Confirm */}
        <TouchableOpacity style={[styles.confirmBtn, { backgroundColor: colors.primary }]} onPress={handleConfirm} activeOpacity={0.85}>
          <Text style={[styles.confirmBtnText, { fontFamily: customText.fontFamily, fontSize: settings.fontSize }]}>{t('confirm')}</Text>
        </TouchableOpacity>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:      { flex: 1 },
  scrollView:     { flex: 1 },
  scrollContent:  { paddingHorizontal: 16, paddingBottom: 20 },
  header:         { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
  backButton:     { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  resetButton:    { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  headerTitleRow: { flexDirection: 'row', alignItems: 'center' },
  headerTitle:    { fontSize: 22, fontWeight: 'bold' },
  previewCard:    { borderRadius: 14, padding: 16, marginBottom: 14, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2 },
  previewLabel:   { fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1 },
  card:           { borderRadius: 16, marginBottom: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.07, shadowRadius: 6, elevation: 2, overflow: 'visible', zIndex: 10 },
  row:            { paddingHorizontal: 16, paddingVertical: 14, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rowBorder:      { borderBottomWidth: 1 },
  fullWidth:      { width: '100%' },
  labelRow:       { flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1 },
  iconWrap:       { width: 34, height: 34, borderRadius: 9, alignItems: 'center', justifyContent: 'center' },
  iconText:       { fontSize: 18, fontWeight: '800' },
  iconTextSm:     { fontSize: 12, fontWeight: '700' },
  label:          { fontSize: 15, fontWeight: '600' },
  swatchRow:      { flexDirection: 'row', gap: 8 },
  colorBlock:     { paddingHorizontal: 16, paddingVertical: 14 },
  confirmBtn:     { borderRadius: 16, paddingVertical: 16, alignItems: 'center', shadowColor: '#004F7F', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.25, shadowRadius: 8, elevation: 4 },
  confirmBtnText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
});