import { router } from "expo-router";
import React, { useState } from "react";
import { View, Image, StyleSheet, TouchableOpacity, Text } from 'react-native'
import { SafeAreaView } from "react-native-safe-area-context";
export default function StartUp() {
  const [showImage] = useState(true);
  return (
    <>
      
      <SafeAreaView style={{flex:1}}>
          <View style={styles.container}>
        {showImage && (
          <Image
            source={require("../assets/images/Starthome.png")}
            style={styles.image}
            resizeMode="cover"
          />
        )}
        <TouchableOpacity style={styles.button1} onPress={()=>router.push("/Screensbar/Nextscreens")}>
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
