import { router } from "expo-router";
import React, { useState } from "react";
import { View, Image, StyleSheet, TouchableOpacity, Text } from 'react-native'
import { SafeAreaView } from "react-native-safe-area-context";
import AsyncStorage from "@react-native-async-storage/async-storage"; // <-- Added this

export default function StartUp() {
  const [showImage] = useState(true);

  // <-- We added an async function to permanently save the flag when they click the button
  const handleProceedHome = async () => {
    await AsyncStorage.setItem("hasSeenOnboarding", "true");
    router.replace("/Screensbar/Nextscreens");
  };

  return (
    <>
      <SafeAreaView style={{flex:1}}>
          <View style={styles.container}>
        {showImage && (
          <Image
            source={require("../assets/images/Starthome.png")}
            style={styles.image}
          />
        )}
        <TouchableOpacity style={styles.button1} onPress={handleProceedHome}>
          <Text style={styles.text1}>Home</Text>
        </TouchableOpacity >
      </View>
    </SafeAreaView>
    </>
  )
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
  },
  image: {
    resizeMode:"cover",
  },
    button1: {
    position: 'absolute',
    bottom: '30%',
    width: '80%',
    alignItems: "center",
    backgroundColor: '#004F7F',
    paddingVertical: 15,
    paddingHorizontal: 32,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 2, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
      text1: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});