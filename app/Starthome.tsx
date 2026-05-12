import { router } from "expo-router";
import React, { useState } from "react";
import { View, Image, StyleSheet, TouchableOpacity, Text } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import AsyncStorage from "@react-native-async-storage/async-storage";

export default function StartUp() {
  const [showImage] = useState(true);

  const handleProceedHome = async () => {
    await AsyncStorage.setItem("hasSeenOnboarding", "true");
    router.replace("/Screensbar/Nextscreens");
  };

  return (
    <View style={styles.container}>
      {/* الطبقة الأولى - مالية الصفحة كلها */}
      {showImage && (
        <Image
          source={require("../assets/images/Starthome.png")}
          style={styles.backgroundImage}
        />
      )}

      {/* الطبقة الثانية - فوق الأولى */}
      <Image
        source={require("../assets/images/textstarthome.png")}
        style={styles.overlayImage}
      />

      {/* الزرار */}
      <TouchableOpacity style={styles.button1} onPress={handleProceedHome}>
        <Text style={styles.text1}>Home</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundImage: {
    position: "absolute", // ← مالية الصفحة كلها
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },
  overlayImage: {
    position: "absolute",
    top: 0,
    width: "100%",
    height: "65%", // ← نص الشاشة
    alignSelf: "center",
    resizeMode: "contain",
  },
  button1: {
    position: "absolute",
    bottom: "20%",
    alignSelf: "center",
    width: "80%",
    alignItems: "center",
    backgroundColor: "#004F7F",
    paddingVertical: 15,
    paddingHorizontal: 32,
    borderRadius: 8,
    shadowColor: "#000",
    shadowOffset: { width: 2, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  text1: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
});
