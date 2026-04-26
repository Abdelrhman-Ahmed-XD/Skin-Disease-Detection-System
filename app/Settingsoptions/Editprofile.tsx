import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import DateTimePicker, { DateTimePickerEvent } from "@react-native-community/datetimepicker";
import * as ImagePicker from "expo-image-picker";
import { router, useFocusEffect } from "expo-router";
import React, { useCallback, useState } from "react";
import {
    ActivityIndicator, Alert, Image, Modal, Platform,
    ScrollView, StatusBar, StyleSheet, Text, TextInput,
    TouchableOpacity, View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useCustomize } from "../Customize/Customizecontext";
import { useTranslation } from "../Customize/translations";
import { useTheme } from "../ThemeContext";
import { loadProfileFromFirestore, saveProfileToFirestore } from "../../Firebase/firestoreProfileService";

const STORAGE_KEY       = "signupDraft";
const CLOUDINARY_CLOUD  = "dignpxpgy";
const CLOUDINARY_PRESET = "skinsight_uploads";

const uploadPhotoToCloudinary = async (localUri: string): Promise<string> => {
    const formData = new FormData();
    formData.append('file', { uri: localUri, type: 'image/jpeg', name: 'profile.jpg' } as any);
    formData.append('upload_preset', CLOUDINARY_PRESET);
    formData.append('folder', 'skinsight_profiles');
    const response = await fetch(
        `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD}/image/upload`,
        { method: 'POST', body: formData }
    );
    if (!response.ok) throw new Error('Cloudinary upload failed');
    const data = await response.json();
    return data.secure_url as string;
};

const skinColors = [
    { label: "Very Light", color: "#F5E0D3" }, { label: "Light", color: "#EACAA7" },
    { label: "Medium", color: "#D1A67A" },     { label: "Tan", color: "#B57D50" },
    { label: "Brown", color: "#A05C38" },      { label: "Dark Brown", color: "#8B4513" },
    { label: "Deep", color: "#7A3E11" },       { label: "Ebony", color: "#603311" },
];
const eyeColorOptions  = [{ name: "Black", color: "#000000" }, { name: "Brown", color: "#7B4B1A" }, { name: "Light Blue", color: "#6EB6FF" }, { name: "Light Green", color: "#6EDB8F" }, { name: "Grey", color: "#9AA0A6" }];
const hairColorOptions = [
  { name: "Black", color: "#000000" },
  { name: "Brown", color: "#7B4B1A" },
  { name: "Gold", color: "#D4A853" },
  { name: "Red", color: "#C0392B" },
  { name: "Grey", color: "#9AA0A6" },
];

export default function EditProfile() {
    const { colors, isDark } = useTheme();
    const { settings } = useCustomize();
    const { t, isArabic } = useTranslation(settings.language);

    const customText = {
      fontSize:   settings.fontSize,
      color:      isDark ? "#FFFFFF" : settings.textColor,
      fontFamily: settings.fontFamily === "System" ? undefined : settings.fontFamily,
    };

    const pageBg = isDark ? colors.background : settings.backgroundColor;

    const [loading, setLoading]       = useState(true);
    const [saving, setSaving]         = useState(false);
    const [isDirty, setIsDirty]       = useState(false);
    const [photoUri, setPhotoUri]     = useState<string | null>(null);
    const [firstName, setFirstName]   = useState("");
    const [lastName, setLastName]     = useState("");
    const [email, setEmail]           = useState("");
    const [birthDay, setBirthDay]     = useState<number | null>(null);
    const [birthMonth, setBirthMonth] = useState<number | null>(null);
    const [birthYear, setBirthYear]   = useState<number | null>(null);
    const [gender, setGender]         = useState<"male" | "female" | null>(null);
    const [genderOpen, setGenderOpen] = useState(false);
    const [isEmailVerified, setIsEmailVerified] = useState(false);
    const [originalEmail, setOriginalEmail]     = useState("");
    const [emailError, setEmailError]           = useState("");
    const [showPicker, setShowPicker] = useState(false);
    const [pickerDate, setPickerDate] = useState(new Date(2000, 0, 1));
    const [skinColor, setSkinColor]   = useState<string | null>(null);
    const [eyeColor, setEyeColor]     = useState<string | null>(null);
    const [hairColor, setHairColor]   = useState<string | null>(null);
    const [skinOpen, setSkinOpen]     = useState(false);
    const [eyeOpen, setEyeOpen]       = useState(false);
    const [hairOpen, setHairOpen]     = useState(false);

    useFocusEffect(
        useCallback(() => {
            const load = async () => {
                setLoading(true);
                try {
                    const data = await loadProfileFromFirestore();
                    if (data) {
                        setFirstName(data.firstName || ""); setLastName(data.lastName || "");
                        setEmail(data.email || ""); setOriginalEmail(data.email || "");
                        setBirthDay(data.birthDay ?? null); setBirthMonth(data.birthMonth ?? null); setBirthYear(data.birthYear ?? null);
                        setGender((data.gender as "male" | "female" | null) ?? null); setIsEmailVerified(data.isEmailVerified || false);
                        setPhotoUri(data.photoUri || null);
                        setSkinColor(data.skinColor || null); setEyeColor(data.eyeColor || null); setHairColor(data.hairColor || null);
                        if (data.birthYear && data.birthMonth && data.birthDay)
                            setPickerDate(new Date(data.birthYear, data.birthMonth - 1, data.birthDay));
                    }
                } catch (e) {
                    console.warn('EditProfile load error:', e);
                } finally { setLoading(false); setIsDirty(false); }
            };
            load();
        }, [])
    );

    const validateEmail = (text: string) => {
        if (!text) setEmailError("");
        else if (!/^\S+@\S+\.\S+$/.test(text)) setEmailError(t('validEmail'));
        else setEmailError("");
    };

    const handleEmailChange = (text: string) => {
        setEmail(text); validateEmail(text); setIsDirty(true);
        const isSame = text.trim().toLowerCase() === originalEmail.trim().toLowerCase();
        setIsEmailVerified(isSame);
    };

    const handleSendVerifyOtp = async () => {
        if (!email || !!emailError) return;
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        const data  = saved ? JSON.parse(saved) : {};
        const name  = `${data.firstName || ''} ${data.lastName || ''}`.trim() || 'User';
        router.push({ pathname: "/VerifyEmailChange", params: { newEmail: email, userName: name } });
    };

    const onDateChange = (_: DateTimePickerEvent, selected?: Date) => {
        if (Platform.OS === "android") setShowPicker(false);
        if (!selected) return;
        setPickerDate(selected);
        setBirthDay(selected.getDate()); setBirthMonth(selected.getMonth() + 1); setBirthYear(selected.getFullYear());
        setIsDirty(true);
    };

    const saveGender = async (value: "male" | "female") => {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        const data = saved ? JSON.parse(saved) : {};
        await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, gender: value }));
    };

    const handlePickImage = () => {
        Alert.alert(t('profilePhoto'), t('chooseOption'), [
            {
                text: t('camera'), onPress: async () => {
                    const perm = await ImagePicker.requestCameraPermissionsAsync();
                    if (!perm.granted) return;
                    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: true, aspect: [1, 1], quality: 0.8 });
                    if (!result.canceled && result.assets[0].uri) await handlePhotoSelected(result.assets[0].uri);
                }
            },
            {
                text: t('gallery'), onPress: async () => {
                    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
                    if (!perm.granted) return;
                    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, allowsEditing: true, aspect: [1, 1], quality: 0.8 });
                    if (!result.canceled && result.assets[0].uri) await handlePhotoSelected(result.assets[0].uri);
                }
            },
            { text: t('cancel'), style: "cancel" },
        ]);
    };

    const handlePhotoSelected = async (localUri: string) => {
        setPhotoUri(localUri); setIsDirty(true); setSaving(true);
        try {
            const cloudUrl = await uploadPhotoToCloudinary(localUri);
            setPhotoUri(cloudUrl);
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...(saved ? JSON.parse(saved) : {}), photoUri: cloudUrl }));
            await saveProfileToFirestore({ photoUri: cloudUrl });
        } catch (err) {
            console.warn('⚠️ Cloudinary upload failed, using local URI:', err);
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...(saved ? JSON.parse(saved) : {}), photoUri: localUri }));
        } finally { setSaving(false); }
    };

    const formatDOB = () => {
        if (!birthDay || !birthMonth || !birthYear) return t('notSet');
        return `${String(birthDay).padStart(2, "0")} / ${String(birthMonth).padStart(2, "0")} / ${birthYear}`;
    };

    const handleSave = async () => {
        if (!firstName.trim() || !lastName.trim()) { Alert.alert(t('error'), t('firstLastRequired')); return; }
        if (!isEmailVerified) { Alert.alert(t('emailNotVerified'), t('verifyEmailFirst')); return; }
        try {
            setSaving(true);
            const profileUpdate = { firstName, lastName, email, gender, birthDay, birthMonth, birthYear, skinColor, eyeColor, hairColor, photoUri, isEmailVerified };
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            const data = saved ? JSON.parse(saved) : {};
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, ...profileUpdate }));
            await saveProfileToFirestore(profileUpdate);
            Alert.alert(t('profileSaved'), t('profileUpdated'), [{ text: t('ok'), onPress: () => router.back() }]);
        } catch (err) {
            console.error('handleSave error:', err);
            Alert.alert(t('error'), t('tryAgain'));
        } finally { setSaving(false); }
    };

    // ── التعديل: لون النص داخل الـ TextInput حسب الثيم ──
    const inputStyle = [
        styles.input,
        customText,
        {
            backgroundColor: colors.input,
            borderColor:     colors.border,
            color:           isDark ? '#FFFFFF' : '#000000',  // ← هنا
        }
    ];

    const inputTouchStyle = [
        styles.input,
        {
            fontSize:   customText.fontSize,
            fontFamily: customText.fontFamily,
            backgroundColor: colors.input,
            borderColor:     colors.border,
        }
    ];

    const dropdownCardStyle = [styles.dropdownCard, { backgroundColor: colors.card, borderColor: colors.border }];

    if (loading) {
        return (
            <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
                <View style={styles.loadingWrap}><ActivityIndicator size="large" color={colors.primary} /></View>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: pageBg }]} edges={["top"]}>
            <StatusBar barStyle={colors.statusBar} backgroundColor={pageBg} />

            <View style={[styles.header, { backgroundColor: colors.card }]}>
                <TouchableOpacity style={[styles.backBtn, { borderColor: colors.border }]} onPress={() => router.back()}>
                    <Ionicons name="chevron-back" size={24} color={colors.text} />
                </TouchableOpacity>
                <Text style={[styles.headerTitle, customText]}>{t('editProfile')}</Text>
                <View style={{ width: 40 }} />
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">

                <TouchableOpacity style={styles.avatarWrap} onPress={handlePickImage} activeOpacity={0.8}>
                    <View style={[styles.avatar, { backgroundColor:"#E8F4F8", borderColor: isDark ? '#00A3A3' : '#004f7f' }]}>
                        {photoUri ? <Image source={{ uri: photoUri }} style={styles.avatarImage} resizeMode="cover" /> : <Ionicons name="person-outline" size={44} color={colors.primary} />}
                    </View>
                    <View style={[styles.avatarEditBtn, { backgroundColor: "#E8F4F8", borderColor: isDark ? '#00A3A3' : '#004f7f' }]}>
                        <Ionicons name="create-outline" size={14} color={"#004f7f"} />
                    </View>
                </TouchableOpacity>

                {/* First Name */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('firstName')}</Text>
                <TextInput
                    style={inputStyle}
                    value={firstName}
                    onChangeText={(v) => { setFirstName(v); setIsDirty(true); }}
                    placeholder={t('enterFirstName')}
                    placeholderTextColor={colors.subText}
                    textAlign={isArabic ? 'right' : 'left'}
                />

                {/* Last Name */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('lastName')}</Text>
                <TextInput
                    style={inputStyle}
                    value={lastName}
                    onChangeText={(v) => { setLastName(v); setIsDirty(true); }}
                    placeholder={t('enterLastName')}
                    placeholderTextColor={colors.subText}
                    textAlign={isArabic ? 'right' : 'left'}
                />

                {/* Email */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('email')}</Text>
                <View style={[styles.emailRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                    <TextInput
                        style={[inputStyle, { flex: 1, marginBottom: 0 }]}
                        value={email}
                        onChangeText={handleEmailChange}
                        placeholder={t('enterEmail')}
                        placeholderTextColor={colors.subText}
                        keyboardType="email-address"
                        autoCapitalize="none"
                        textAlign={isArabic ? 'right' : 'left'}
                    />
                    {!isEmailVerified ? (
                        <TouchableOpacity
                            style={[styles.verifyBtn, { backgroundColor: email && !emailError ? colors.primary : colors.subText }]}
                            disabled={!email || !!emailError}
                            onPress={handleSendVerifyOtp}
                        >
                            <Text style={[styles.verifyBtnText, { fontFamily: customText.fontFamily }]}>{t('verify')}</Text>
                        </TouchableOpacity>
                    ) : (
                        <View style={[styles.verifyBtn, { backgroundColor: "#28A745", flexDirection: 'row', gap: 4 }]}>
                            <Ionicons name="checkmark-circle" size={14} color="#fff" />
                            <Text style={[styles.verifyBtnText, { fontFamily: customText.fontFamily }]}>{t('verified')}</Text>
                        </View>
                    )}
                </View>
                {!!emailError && <Text style={[styles.errorText, customText, { color: 'red', textAlign: isArabic ? 'right' : 'left' }]}>{emailError}</Text>}

                {/* Age */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('age')}</Text>
                <TouchableOpacity style={inputTouchStyle} onPress={() => setShowPicker(true)} activeOpacity={0.8}>
                    <View style={[styles.dobRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                        <Text style={[styles.inputText, customText, { color: birthDay ? (isDark ? '#FFFFFF' : '#000000') : colors.subText }]}>
                            {formatDOB()}
                        </Text>
                        <Ionicons name="calendar-outline" size={20} color={colors.subText} />
                    </View>
                </TouchableOpacity>

                {Platform.OS === "android" && showPicker && (
                    <DateTimePicker value={pickerDate} mode="date" display="default" maximumDate={new Date()} onChange={onDateChange} />
                )}
                {Platform.OS === "ios" && (
                    <Modal visible={showPicker} transparent animationType="slide" onRequestClose={() => setShowPicker(false)}>
                        <View style={styles.modalOverlay}>
                            <View style={[styles.modalCard, { backgroundColor: colors.card }]}>
                                <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
                                    <TouchableOpacity onPress={() => setShowPicker(false)}>
                                        <Text style={[styles.modalCancel, customText]}>{t('cancel')}</Text>
                                    </TouchableOpacity>
                                    <Text style={[styles.modalTitle, customText]}>{t('selectDate')}</Text>
                                    <TouchableOpacity onPress={() => setShowPicker(false)}>
                                        <Text style={[styles.modalDone, customText, { color: colors.primary }]}>{t('done')}</Text>
                                    </TouchableOpacity>
                                </View>
                                <DateTimePicker value={pickerDate} mode="date" display="spinner" maximumDate={new Date()} onChange={onDateChange} style={{ height: 200 }} />
                            </View>
                        </View>
                    </Modal>
                )}

                {/* Gender */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('gender')}</Text>
                <TouchableOpacity style={[inputTouchStyle, { marginBottom: 0 }]} onPress={() => setGenderOpen(!genderOpen)} activeOpacity={0.8}>
                    <View style={[styles.dobRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                        {gender ? (
                            <View style={[styles.genderValueRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                                <Ionicons name={gender === "male" ? "male" : "female"} size={18} color={gender === "male" ? colors.primary : "#E6007A"} />
                                <Text style={[styles.inputText, customText, { color: isDark ? '#FFFFFF' : '#000000' }]}>
                                    {gender === "male" ? t('male') : t('female')}
                                </Text>
                            </View>
                        ) : (
                            <Text style={[styles.genderPlaceholder, customText, { color: colors.subText }]}>{t('chooseGender')}</Text>
                        )}
                        <Ionicons name={genderOpen ? "chevron-up" : "chevron-down"} size={16} color={colors.subText} />
                    </View>
                </TouchableOpacity>
                {genderOpen && (
                    <View style={dropdownCardStyle}>
                        {["male", "female"].map((g, i) => (
                            <React.Fragment key={g}>
                                <TouchableOpacity
                                    style={[styles.dropdownItem, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                                    onPress={() => { setGender(g as any); setGenderOpen(false); setIsDirty(true); saveGender(g as any); }}
                                >
                                    <Ionicons name={g === "male" ? "male" : "female"} size={18} color={g === "male" ? colors.primary : "#E6007A"} />
                                    <Text style={[styles.dropdownItemText, customText]}>{g === "male" ? t('male') : t('female')}</Text>
                                    {gender === g && <Ionicons name="checkmark" size={18} color={colors.primary} style={{ marginLeft: isArabic ? 0 : "auto", marginRight: isArabic ? "auto" : 0 }} />}
                                </TouchableOpacity>
                                {i === 0 && <View style={[styles.dropdownDivider, { backgroundColor: colors.border }]} />}
                            </React.Fragment>
                        ))}
                    </View>
                )}

                {/* Skin Tone */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('skinTone')}</Text>
                <TouchableOpacity style={[inputTouchStyle, { marginBottom: 0 }]} onPress={() => setSkinOpen(!skinOpen)} activeOpacity={0.8}>
                    <View style={[styles.dobRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                        {skinColor ? (
                            <View style={[styles.genderValueRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                                <View style={[styles.colorCircle, { backgroundColor: skinColor }]} />
                                <Text style={[styles.inputText, customText, { color: isDark ? '#FFFFFF' : '#000000' }]}>
                                    {skinColors.find(s => s.color === skinColor)?.label || skinColor}
                                </Text>
                            </View>
                        ) : (
                            <Text style={[styles.genderPlaceholder, customText, { color: colors.subText }]}>{t('chooseSkinTone')}</Text>
                        )}
                        <Ionicons name={skinOpen ? "chevron-up" : "chevron-down"} size={16} color={colors.subText} />
                    </View>
                </TouchableOpacity>
                {skinOpen && (
                    <View style={dropdownCardStyle}>
                        {skinColors.map((item, index) => (
                            <React.Fragment key={item.label}>
                                <TouchableOpacity
                                    style={[styles.dropdownItem, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                                    onPress={() => { setSkinColor(item.color); setSkinOpen(false); setIsDirty(true); }}
                                >
                                    <View style={[styles.colorCircle, { backgroundColor: item.color }]} />
                                    <Text style={[styles.dropdownItemText, customText]}>{item.label}</Text>
                                    {skinColor === item.color && <Ionicons name="checkmark" size={18} color={colors.primary} style={{ marginLeft: isArabic ? 0 : "auto", marginRight: isArabic ? "auto" : 0 }} />}
                                </TouchableOpacity>
                                {index < skinColors.length - 1 && <View style={[styles.dropdownDivider, { backgroundColor: colors.border }]} />}
                            </React.Fragment>
                        ))}
                    </View>
                )}

                {/* Eye Color */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('eyeColor')}</Text>
                <TouchableOpacity style={[inputTouchStyle, { marginBottom: 0 }]} onPress={() => setEyeOpen(!eyeOpen)} activeOpacity={0.8}>
                    <View style={[styles.dobRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                        {eyeColor ? (
                            <View style={[styles.genderValueRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                                <View style={[styles.colorCircle, { backgroundColor: eyeColorOptions.find(e => e.name === eyeColor)?.color }]} />
                                <Text style={[styles.inputText, customText, { color: isDark ? '#FFFFFF' : '#000000' }]}>
                                    {eyeColor}
                                </Text>
                            </View>
                        ) : (
                            <Text style={[styles.genderPlaceholder, customText, { color: colors.subText }]}>{t('chooseEyeColor')}</Text>
                        )}
                        <Ionicons name={eyeOpen ? "chevron-up" : "chevron-down"} size={16} color={colors.subText} />
                    </View>
                </TouchableOpacity>
                {eyeOpen && (
                    <View style={dropdownCardStyle}>
                        {eyeColorOptions.map((item, index) => (
                            <React.Fragment key={item.name}>
                                <TouchableOpacity
                                    style={[styles.dropdownItem, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                                    onPress={() => { setEyeColor(item.name); setEyeOpen(false); setIsDirty(true); }}
                                >
                                    <View style={[styles.colorCircle, { backgroundColor: item.color }]} />
                                    <Text style={[styles.dropdownItemText, customText]}>{item.name}</Text>
                                    {eyeColor === item.name && <Ionicons name="checkmark" size={18} color={colors.primary} style={{ marginLeft: isArabic ? 0 : "auto", marginRight: isArabic ? "auto" : 0 }} />}
                                </TouchableOpacity>
                                {index < eyeColorOptions.length - 1 && <View style={[styles.dropdownDivider, { backgroundColor: colors.border }]} />}
                            </React.Fragment>
                        ))}
                    </View>
                )}

                {/* Hair Color */}
                <Text style={[styles.label, customText, { textAlign: isArabic ? 'right' : 'left' }]}>{t('hairColor')}</Text>
                <TouchableOpacity style={[inputTouchStyle, { marginBottom: 0 }]} onPress={() => setHairOpen(!hairOpen)} activeOpacity={0.8}>
                    <View style={[styles.dobRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                        {hairColor ? (
                            <View style={[styles.genderValueRow, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}>
                                <View style={[styles.colorCircle, { backgroundColor: hairColorOptions.find(h => h.name === hairColor)?.color }]} />
                                <Text style={[styles.inputText, customText, { color: isDark ? '#FFFFFF' : '#000000' }]}>
                                    {hairColor}
                                </Text>
                            </View>
                        ) : (
                            <Text style={[styles.genderPlaceholder, customText, { color: colors.subText }]}>{t('chooseHairColor')}</Text>
                        )}
                        <Ionicons name={hairOpen ? "chevron-up" : "chevron-down"} size={16} color={colors.subText} />
                    </View>
                </TouchableOpacity>
                {hairOpen && (
                    <View style={dropdownCardStyle}>
                        {hairColorOptions.map((item, index) => (
                            <React.Fragment key={item.name}>
                                <TouchableOpacity
                                    style={[styles.dropdownItem, { flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                                    onPress={() => { setHairColor(item.name); setHairOpen(false); setIsDirty(true); }}
                                >
                                    <View style={[styles.colorCircle, { backgroundColor: item.color }]} />
                                    <Text style={[styles.dropdownItemText, customText]}>{item.name}</Text>
                                    {hairColor === item.name && <Ionicons name="checkmark" size={18} color={colors.primary} style={{ marginLeft: isArabic ? 0 : "auto", marginRight: isArabic ? "auto" : 0 }} />}
                                </TouchableOpacity>
                                {index < hairColorOptions.length - 1 && <View style={[styles.dropdownDivider, { backgroundColor: colors.border }]} />}
                            </React.Fragment>
                        ))}
                    </View>
                )}

                <TouchableOpacity
                    style={[styles.changePasswordBtn, { backgroundColor: colors.primary, alignSelf: isArabic ? 'flex-start' : 'flex-end', flexDirection: isArabic ? 'row-reverse' : 'row' }]}
                    onPress={() => router.push('/Settingsoptions/Changepassword')}
                    activeOpacity={0.85}
                >
                    <Ionicons name="lock-closed-outline" size={12} color="#fff" />
                    <Text style={[styles.changePasswordText, { fontFamily: customText.fontFamily }]}>{t('changePassword')}</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={[styles.confirmBtn, isDirty ? { backgroundColor: colors.primary } : { backgroundColor: colors.subText }, saving && { opacity: 0.7 }]}
                    onPress={handleSave}
                    disabled={!isDirty || saving}
                    activeOpacity={0.85}
                >
                    {saving ? <ActivityIndicator color="#fff" /> : (
                        <Text style={[styles.confirmText, { fontFamily: customText.fontFamily }]}>{t('confirm')}</Text>
                    )}
                </TouchableOpacity>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container:          { flex: 1 },
    loadingWrap:        { flex: 1, justifyContent: "center", alignItems: "center" },
    header:             { flexDirection: "row", alignItems: "center", justifyContent: "space-between", paddingHorizontal: 16, paddingVertical: 12, borderRadius: 15, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, shadowRadius: 4, elevation: 2, margin: 15 },
    backBtn:            { width: 40, height: 40, borderRadius: 12, borderWidth: 1, alignItems: "center", justifyContent: "center" },
    headerTitle:        { fontSize: 20, fontWeight: "bold" },
    scrollContent:      { paddingHorizontal: 20, paddingBottom: 40 },
    avatarWrap:         { alignSelf: "center", marginBottom: 24, position: "relative" },
    avatar:             { width: 90, height: 90, borderRadius: 45, borderWidth: 2, justifyContent: "center", alignItems: "center", overflow: "hidden" },
    avatarEditBtn:      { position: "absolute", bottom: 0, right: 0, width: 30, height: 30, borderRadius: 15, borderWidth: 1.5, justifyContent: "center", alignItems: "center" },
    avatarImage:        { width: 90, height: 90, borderRadius: 45 },
    label:              { fontSize: 14, fontWeight: "600", marginBottom: 6, marginTop: 14 },
    input:              { borderRadius: 12, paddingHorizontal: 14, paddingVertical: 13, fontSize: 15, borderWidth: 1, marginBottom: 2 },
    inputText:          { fontSize: 15 },
    errorText:          { fontSize: 13, marginTop: 4 },
    emailRow:           { flexDirection: "row", alignItems: "center", gap: 8 },
    verifyBtn:          { paddingHorizontal: 14, paddingVertical: 12, borderRadius: 10, justifyContent: "center", alignItems: "center" },
    verifyBtnText:      { color: "#fff", fontWeight: "600", fontSize: 13 },
    dobRow:             { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
    modalOverlay:       { flex: 1, backgroundColor: "rgba(0,0,0,0.4)", justifyContent: "flex-end" },
    modalCard:          { borderTopLeftRadius: 20, borderTopRightRadius: 20, paddingBottom: 30 },
    modalHeader:        { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1 },
    modalTitle:         { fontSize: 16, fontWeight: "700" },
    modalCancel:        { fontSize: 15 },
    modalDone:          { fontSize: 15, fontWeight: "700" },
    genderPlaceholder:  { fontSize: 15 },
    genderValueRow:     { flexDirection: "row", alignItems: "center", gap: 8 },
    dropdownCard:       { borderRadius: 12, borderWidth: 1, marginTop: 4, overflow: "hidden", elevation: 4 },
    dropdownItem:       { flexDirection: "row", alignItems: "center", gap: 10, paddingHorizontal: 16, paddingVertical: 14 },
    dropdownItemText:   { fontSize: 15, fontWeight: "500" },
    dropdownDivider:    { height: 1, marginHorizontal: 12 },
    colorCircle:        { width: 22, height: 22, borderRadius: 11, borderWidth: 1, borderColor: "#E5E7EB" },
    confirmBtn:         { borderRadius: 14, paddingVertical: 16, marginTop: 16, alignItems: "center" },
    confirmText:        { color: "#fff", fontSize: 16, fontWeight: "700" },
    changePasswordBtn:  { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8, borderRadius: 14, paddingVertical: 10, paddingHorizontal: 14, marginTop: 16 },
    changePasswordText: { color: "#fff", fontSize: 10, fontWeight: "600" },
});