import StartUp from "@/app/StartUp";
import * as Updates from "expo-updates";
import React, { useEffect, useState } from "react";
import { I18nManager, Image, StyleSheet, View } from "react-native";

// Disable RTL
if (I18nManager.isRTL) {
  I18nManager.allowRTL(false);
  I18nManager.forceRTL(false);
  Updates.reloadAsync();
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
          <View style={{ flex: 1, width: "100%" }}>
            {/* Background Image */}
            <Image
              source={require("../assets/images/splash-screen.png")}
              style={{ width: "100%", height: "100%", position: "absolute" }}
              resizeMode="cover"
            />
            {/* Logo in center */}
            <Image
              source={require("../assets/images/Logo4.png")}
              style={{
                position: "absolute",
                top: "50%",
                left: "45%",
                transform: [{ translateX: -150 }, { translateY: -75 }],
                width: 380,
                height: 150,
                resizeMode: "contain",
              }}
            />
          </View>
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