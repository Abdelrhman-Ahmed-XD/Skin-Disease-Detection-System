import React from 'react'
import { View ,Text ,Image ,TouchableOpacity,StyleSheet, Pressable} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { router } from 'expo-router'
import { Ionicons } from '@expo/vector-icons'

export default function Accountcreated() {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#D8E9F0" }}>
        <Pressable onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={28} color="black" />
        </Pressable>
      <View style={{ margin: 20, justifyContent: "center", alignItems: "center" }}>

        <Image
          source={require("../assets/images/cloud 1.png")}
          style={{ width: 250, height: 250, marginBottom: 5 }}
        />
        <Text style={{ fontSize: 24, fontWeight: "bold", color: "#000" }}>
          Account is created!
        </Text>
          <TouchableOpacity onPress={() => router.push("/Login1")} style={{ width: "100%" }}>
            <Text style={styles.Continue}>Continue</Text>
          </TouchableOpacity>
      </View>
    </SafeAreaView>
  )
}
const styles = StyleSheet.create({
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
    margin: 20,
  },
})
