import StartUp from "@/app/StartUp";
import * as SplashScreen from "expo-splash-screen";
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
    // Dismiss the native splash now that React is rendered — seamless handoff
    SplashScreen.hideAsync().catch(() => {});

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
            {/* Background Image — contain keeps full image visible, no cropping */}
            <Image
              source={require("../assets/images/splash-screen.png")}
              style={{ width: "100%", height: "100%", position: "absolute" }}
              resizeMode="contain"
            />
            {/* Logo in center with rounded corners */}
            <View
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: [{ translateX: -150 }, { translateY: -75 }],
                width: 300,
                height: 150,
                borderRadius: 20,
                overflow: "hidden",
              }}
            >
              <Image
                source={require("../assets/images/Logo4.png")}
                style={{ width: "100%", height: "100%" }}
                resizeMode="contain"
              />
            </View>
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
    backgroundColor: "#000000",
  },
  image: {
    resizeMode: "cover",
  },
});