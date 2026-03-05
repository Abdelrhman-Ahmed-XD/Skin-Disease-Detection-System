import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import React, { useState } from "react";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const STORAGE_KEY = "signupDraft";

const colors = [
  { label: "Very Light", color: "#F5E0D3" },
  { label: "Light",      color: "#EACAA7" },
  { label: "Medium",     color: "#D1A67A" },
  { label: "Tan",        color: "#B57D50" },
  { label: "Brown",      color: "#A05C38" },
  { label: "Dark Brown", color: "#8B4513" },
  { label: "Deep",       color: "#7A3E11" },
  { label: "Ebony",      color: "#603311" },
];

export default function Skin() {
  const [selectedColor, setSelectedColor] = useState("");

  const handleSelect = async (color: string) => {
    setSelectedColor(color);
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      const data = saved ? JSON.parse(saved) : {};
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, skinColor: color }));
    } catch (err) {
      console.log("Skin save error:", err);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#DDF0F6" }}>
      <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 20 }}>
        <Pressable onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={28} color="black" />
        </Pressable>

        <View style={styles.topRow}>
          {!selectedColor && (
            <TouchableOpacity onPress={() => router.push("/EyeScreen")}>
              <Text style={styles.skip}>Skip</Text>
            </TouchableOpacity>
          )}
          <Text style={styles.percent}>{selectedColor ? "60%" : "40%"}</Text>
        </View>

        <View style={styles.progressBg}>
          <View style={[styles.progressFill, { width: selectedColor ? "60%" : "40%" }]} />
        </View>

        <Text style={styles.title}>Choose your skin tone</Text>
        <Text style={styles.subtitle}>Select the skin tone that best matches you.</Text>

        <View style={styles.centerArea}>
          <View style={[styles.selectedBox, { backgroundColor: selectedColor || "#eee" }]} />

          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.palette}>
            {colors.map((item, index) => (
              <TouchableOpacity
                key={index}
                style={[
                  styles.colorBox,
                  { backgroundColor: item.color },
                  selectedColor === item.color && styles.selectedBorder,
                ]}
                onPress={() => handleSelect(item.color)}
              />
            ))}
          </ScrollView>
        </View>

        <TouchableOpacity
          onPress={() => router.push("/EyeScreen")}
          disabled={!selectedColor}
          style={[styles.continueBtn, !selectedColor && styles.continueDisabled]}
        >
          <Text style={styles.continueText}>Continue</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  topRow: { flexDirection: "row", alignItems: "center", marginTop: 20 },
  percent: { fontWeight: "600", marginLeft: "auto" },
  skip: { color: "#0B4F6C", fontWeight: "600" },
  progressBg: { height: 6, backgroundColor: "#ccc", borderRadius: 10, marginVertical: 12 },
  progressFill: { height: "100%", backgroundColor: "#0B4F6C", borderRadius: 10 },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginTop: 30 },
  subtitle: { textAlign: "center", marginTop: 10, color: "#000" },
  centerArea: { alignItems: "center", marginTop: 40 },
  selectedBox: { width: 200, height: 150, borderRadius: 12, marginBottom: 20 },
  palette: { paddingHorizontal: 10 },
  colorBox: { width: 50, height: 50, borderRadius: 25, marginHorizontal: 6 },
  selectedBorder: { borderWidth: 3, borderColor: "#007AFF" },
  continueBtn: { backgroundColor: "#004F7F", marginTop: 50, padding: 16, borderRadius: 12, marginBottom: 30 },
  continueDisabled: { backgroundColor: "#aaa" },
  continueText: { color: "#fff", textAlign: "center", fontWeight: "bold" },
  backBtn: {
    width: 42, height: 42, backgroundColor: "#D8E9F0",
    borderRadius: 15, borderColor: "#000", borderWidth: 0.5,
    justifyContent: "center", alignItems: "center", elevation: 3,
  },
});