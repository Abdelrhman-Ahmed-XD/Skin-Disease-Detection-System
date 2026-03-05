import { Ionicons } from '@expo/vector-icons'
import { router } from 'expo-router'
import React from 'react'
import { Image, Pressable, StyleSheet, Text, TouchableOpacity, View } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'


export default function Continue() {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#D8E9F0" }}>
        <Pressable onPress={() => router.back()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={28} color="black" />
        </Pressable>
      <View style={{ margin: 20, justifyContent: "center", alignItems: "center" }}>

        <Image
          source={require("../assets/images/Home.png")}
          style={{ width: 250, height: 250, marginBottom: 5 ,marginRight:60}}
        />
        <Text style={{ fontSize: 24, fontWeight: "bold", color: "#000" ,marginTop:20,textAlign:"center",flexDirection:"row"}}>
          Finish your profile in the profile section to 
        </Text>
        <Text style={{ fontWeight: "bold", color: "#000" ,marginTop:20,textAlign:"center",flexDirection:"row"}}>
          get the best experience, or tap Continue to proceed.
        </Text>
      <View style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 40 ,width:"80%"}}>
          <TouchableOpacity onPress={() => router.push("/Settingsoptions/Editprofile")}
            style={{
              backgroundColor: "#000",
              padding: 12,
              borderRadius: 8,
              flex: 1,
              marginRight: 10, // مسافة بين الزرين
            }}
          
          >
            <Text style={{ color: "#fff", textAlign: "center", fontWeight: "bold" }}>
              Go to Profile
            </Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={() => router.push("/Screensbar/Nextscreens")}
            style={{
              backgroundColor: "#0B4F6C",
              padding: 12,
              borderRadius: 8,
              flex: 1,
              marginLeft: 10, // مسافة بين الزرين
            }}
            
          >
          <Text style={{ color: "#fff", textAlign: "center", fontWeight: "bold" }}>
            Continue
          </Text>
        </TouchableOpacity>
      </View>

      </View>
    </SafeAreaView>
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
    margin: 20,
  },
})
