import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import React from 'react'
import { Image, Pressable, StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'

export default function Resetsign() {
  return (
      <>
      <SafeAreaView style={{ flex: 1, backgroundColor: "#D8E9F0" }}>
        <View style={{ margin: 20 }}>
        <Pressable onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={28} color="black" />
          </Pressable>
          <View style={{ margin: 20, justifyContent: "center", alignItems: "center" }}>
            <Image
              source={require("../assets/images/checkmark.png")}
              style={{ width: 250, height: 250, marginBottom: 5, marginTop:30}}
            />
            <Text style={{ fontSize: 24, fontWeight: "bold", color: "#000", textAlign:"center" }}>
              Password is Changed Succsessfuly !
            </Text>
              <TouchableOpacity onPress={() => router.push("/Login1")} style={{ width: "100%" }}>
                <Text style={styles.Continue}>Continue</Text>
            </TouchableOpacity>
            </View>
        </View>
        </SafeAreaView>
      </>
  )
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
    Continue: {
    marginTop: "50%",
    padding: 14,
    borderRadius: 8,
    marginBottom: 5,
    backgroundColor: "#004F7F",
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
    textAlign: "center",
  },
});