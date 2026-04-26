import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import React, { useState } from "react";
import {
    ActivityIndicator,
    Pressable,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { flushOnboardingToFirestore } from "../Firebase/firestoreProfileService";

const STORAGE_KEY = "signupDraft";

const hairColors = [
  { name: "Black", border: "#000", color: "#000000" },
  { name: "Brown", border: "#7B4B1A", color: "#7B4B1A" },
  { name: "Light Blue", border: "#D4A853", color: "#D4A853" },
  { name: "Light Green", border: "#C0392B", color: "#C0392B" },
  { name: "Grey", border: "#9AA0A6", color: "#9AA0A6" },
];

export default function Hair() {
    const [selectedColor, setSelectedColor] = useState("");
    const [saving, setSaving] = useState(false);

    const handleSelect = async (name: string) => {
        setSelectedColor(name);
        try {
            const saved = await AsyncStorage.getItem(STORAGE_KEY);
            const data  = saved ? JSON.parse(saved) : {};
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, hairColor: name }));
        } catch (err) {
            console.log("Hair save error:", err);
        }
    };

    // ── Called when user taps Continue ────────────────────────────────────────
    const handleContinue = async () => {
        setSaving(true);
        try {
            // Flush ALL onboarding data (gender, age, skin, eye, hair) to Firestore
            await flushOnboardingToFirestore();
            console.log("✅ Onboarding data saved to Firestore");
        } catch (err) {
            // Don't block navigation — data is already in AsyncStorage
            console.warn("Firestore flush failed (will sync later):", err);
        } finally {
            setSaving(false);
            router.push("/Continue");
        }
    };

    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: "#DDF0F6" }}>
            <View style={{ margin: 20 }}>
                <Pressable onPress={() => router.back()} style={styles.backBtn}>
                    <Ionicons name="arrow-back" size={28} color="black" />
                </Pressable>

                <View style={styles.topRow}>
                    {!selectedColor && (
                        <TouchableOpacity onPress={handleContinue}>
                            <Text style={styles.skip}>Skip</Text>
                        </TouchableOpacity>
                    )}
                    <Text style={styles.percent}>{selectedColor ? "100%" : "80%"}</Text>
                </View>

                <View style={styles.progressBg}>
                    <View style={[styles.progressFill, { width: selectedColor ? "100%" : "80%" }]} />
                </View>

                <Text style={styles.title}>Your hair color is ?</Text>
                <Text style={styles.subtitle}>Select Your Hair color from the list below.</Text>

                {hairColors.map((item, index) => (
                    <TouchableOpacity
                        key={index}
                        onPress={() => handleSelect(item.name)}
                        style={styles.optionWrapper}
                    >
                        <View style={[styles.topColorBar, { backgroundColor: item.border }]} />
                        <View style={[styles.optionCard, selectedColor === item.name && styles.optionSelected]}>
                            <Text style={styles.optionText}>{item.name}</Text>
                            <View style={[styles.radio, selectedColor === item.name && styles.radioActive]} />
                        </View>
                    </TouchableOpacity>
                ))}

                <TouchableOpacity
                    onPress={handleContinue}
                    disabled={!selectedColor || saving}
                    style={[styles.continueBtn, (!selectedColor || saving) && styles.continueDisabled]}
                >
                    {saving ? (
                        <ActivityIndicator color="#fff" />
                    ) : (
                        <Text style={styles.continueText}>Continue</Text>
                    )}
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    topRow:          { flexDirection: "row", justifyContent: "flex-start", alignItems: "center", marginTop: 30 },
    percent:         { fontWeight: "600", marginLeft: "auto" },
    skip:            { color: "#0B4F6C", fontWeight: "600" },
    progressBg:      { height: 6, backgroundColor: "#ccc", borderRadius: 10, marginVertical: 10 },
    progressFill:    { height: "100%", backgroundColor: "#0B4F6C", borderRadius: 10 },
    title:           { fontSize: 24, fontWeight: "bold", textAlign: "center", marginTop: 40, marginBottom: 40 },
    subtitle:        { textAlign: "center", marginTop: 5, color: "#000000", marginBottom: 20 },
    backBtn:         { width: 42, height: 42, backgroundColor: "#D8E9F0", borderRadius: 15, borderColor: "#000", borderWidth: 0.5, justifyContent: "center", alignItems: "center", elevation: 3 },
    continueBtn:     { backgroundColor: "#004F7F", marginTop: 50, padding: 16, borderRadius: 12 },
    continueDisabled:{ backgroundColor: "#aaa" },
    continueText:    { color: "#fff", textAlign: "center", fontWeight: "bold" },
    optionWrapper:   { marginBottom: 18, borderRadius: 16, overflow: "hidden" },
    topColorBar:     { height: 6, width: "100%" },
    optionCard:      { backgroundColor: "#F2F4F5", padding: 16, borderBottomLeftRadius: 16, borderBottomRightRadius: 16, flexDirection: "row", justifyContent: "space-between", alignItems: "center", shadowColor: "#000", shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 6, elevation: 6 },
    optionSelected:  { backgroundColor: "#FFFFFF" },
    optionText:      { fontSize: 16, fontWeight: "500" },
    radio:           { width: 20, height: 20, borderRadius: 10, borderWidth: 2, borderColor: "#333" },
    radioActive:     { backgroundColor: "#004F7F", borderColor: "#004F7F" },
});