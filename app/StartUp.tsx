import { router } from "expo-router";
import React, { useState } from "react";
import { Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
export default function StartUp() {
  const [showImage] = useState(true);
  return (
    <>
      
      <View style={styles.container}>
        {showImage && (
          <Image
            source={require("../assets/images/The Buzz 1.png")}
            style={styles.image}
            resizeMode="cover"
          />
        )}
        <TouchableOpacity style={styles.button1} onPress={()=>router.push("/Login1")}>
          <Text style={styles.text1}>Login</Text>
        </TouchableOpacity >
        <TouchableOpacity style={styles.button2 } onPress={()=>router.push("/SignUp")}>
          <Text style={styles.text2}>Sign UP</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button3} onPress={()=>router.push("/Guest/Guest")}>
          <Text style={styles.text3}>Guest</Text>
        </TouchableOpacity>
      </View>
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
    backgroundColor: '#fff',
    paddingVertical: 15,
    paddingHorizontal: 32,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 2, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
    button2: {
      position: 'absolute',
    bottom: '20%',
    width: '80%',
    alignItems: "center",
    backgroundColor: '#004F80',
    paddingVertical: 15,
    paddingHorizontal: 32,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 2, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  button3: {
        
    position: 'absolute',
    bottom: '10%',
    width: '80%',
    alignItems: "center",
    backgroundColor: '#454553',
    paddingVertical: 15,
    paddingHorizontal:32,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 2, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  text1: {
    color: '#000',
    fontSize: 16,
    fontWeight: 'bold',
  },
  text2: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  text3: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
// socialBtn: {
//     flex: 1,
//     height: 55,
//     backgroundColor: "#fff",
//     borderRadius: 12,
//     alignItems: "center",
//     justifyContent: "center",
//     marginHorizontal: 5,
//     elevation: 2,
//   },