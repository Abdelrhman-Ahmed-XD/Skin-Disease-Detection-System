import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import React, { useState } from "react";
import {
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const STORAGE_KEY = "signupDraft";

export default function GenderScreen() {
  const [gender, setGender] = useState<"male" | "female" | null>(null);

  const handleSelect = async (value: "male" | "female") => {
    setGender(value);
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      const data = saved ? JSON.parse(saved) : {};
      await AsyncStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ ...data, gender: value })
      );
    } catch (err) {
      console.log("Gender save error:", err);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#DDF0F6" }}>
      <View style={{ margin: 20 }}>
        <Pressable onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={28} color="black" />
        </Pressable>

        {/* TOP ROW */}
        <View style={styles.topRow}>
          {!gender && (
            <TouchableOpacity onPress={() => router.push("/Age")}>
              <Text style={styles.skip}>Skip</Text>
            </TouchableOpacity>
          )}
          <Text style={styles.percent}>{gender ? "20%" : "0%"}</Text>
        </View>

        {/* PROGRESS */}
        <View style={styles.progressBg}>
          <View
            style={[styles.progressFill, { width: gender ? "20%" : "0%" }]}
          />
        </View>

        {/* TITLE */}
        <Text style={styles.title}>What's your gender?</Text>
        <Text style={styles.subtitle}>
          Adding your gender helps us maintain{"\n"}
          complete and accurate information.
        </Text>

        {/* OPTIONS */}
        <View style={styles.optionsRow}>
          {/* MALE */}
          <TouchableOpacity
            onPress={() => handleSelect("male")}
            style={[
              styles.card,
              gender === null || gender === "male"
                ? styles.maleCard
                : styles.inactiveCard,
            ]}
          >
            <Ionicons name="male" size={26} color="#fff" />
            <Text style={styles.cardText}>Male</Text>
          </TouchableOpacity>

          {/* FEMALE */}
          <TouchableOpacity
            onPress={() => handleSelect("female")}
            style={[
              styles.card,
              gender === null || gender === "female"
                ? styles.femaleCard
                : styles.inactiveCard,
            ]}
          >
            <Ionicons name="female" size={26} color="#fff" />
            <Text style={styles.cardText}>Female</Text>
          </TouchableOpacity>
        </View>

        {/* CONTINUE */}
        <TouchableOpacity
          onPress={() => router.push("/Age")}
          disabled={!gender}
          style={[styles.continueBtn, !gender && styles.continueDisabled]}
        >
          <Text style={styles.continueText}>Continue</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  backBtn: {
    width: 42,
    height: 42,
    backgroundColor: "#D8E9F0",
    borderRadius: 15,
    borderColor: "#000",
    borderWidth: 0.5,
    justifyContent: "center",
    alignItems: "center",
    elevation: 3,
  },
  topRow: {
    flexDirection: "row",
    justifyContent: "flex-start",
    alignItems: "center",
    marginTop: 30,
  },
  percent: { fontWeight: "600", marginLeft: "auto" },
  skip: { color: "#0B4F6C", fontWeight: "600" },
  progressBg: {
    height: 6,
    backgroundColor: "#ccc",
    borderRadius: 10,
    marginVertical: 10,
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#0B4F6C",
    borderRadius: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    textAlign: "center",
    marginTop: 30,
  },
  subtitle: { textAlign: "center", marginTop: 10, color: "#555" },
  optionsRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: 40,
  },
  card: {
    width: 110,
    height: 110,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    elevation: 4,
  },
  maleCard: { backgroundColor: "#004F7F" },
  femaleCard: { backgroundColor: "#E6007A" },
  inactiveCard: { backgroundColor: "#aeaeae" },
  cardText: { marginTop: 8, color: "#fff", fontWeight: "600" },
  continueBtn: {
    backgroundColor: "#004F7F",
    marginTop: 50,
    padding: 16,
    borderRadius: 12,
  },
  continueDisabled: { backgroundColor: "#aaa" },
  continueText: { color: "#fff", textAlign: "center", fontWeight: "bold" },
});