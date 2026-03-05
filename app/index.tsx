import StartUp from "@/app/StartUp";
import * as Updates from "expo-updates";
import React, { useEffect, useState } from "react";
import { I18nManager, Image, StyleSheet, View } from "react-native";

// Disable RTL
if (I18nManager.isRTL) {
  I18nManager.allowRTL(false);
  I18nManager.forceRTL(false);
  Updates.reloadAsync(); // ← ده المهم! بيعمل restart لو الـ RTL اتغير
}

export default function Index() {
  const [showImage, setShowImage] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowImage(false);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      <View style={styles.container}>
        {showImage && (
          <Image
            source={require("../assets/images/splash-screen.png")}
            style={styles.image}
            resizeMode="cover"
          />
        )}
        {!showImage && <StartUp />}
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#D8E9F0",
  },
  image: {
    resizeMode: "cover",
  },
});