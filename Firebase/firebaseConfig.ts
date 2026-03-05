// firebaseConfig.ts
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyCWkWoXSd7RXUvZJ2wk_W010KyV1TPp60U",
    authDomain: "skin-disease-detection-s-9b419.firebaseapp.com",
    projectId: "skin-disease-detection-s-9b419",
    storageBucket: "skin-disease-detection-s-9b419.firebasestorage.app",
    messagingSenderId: "257266533414",
    appId: "1:257266533414:web:761e3ef59136de75d1d2e7",
    measurementId: "G-YL03BPWEXF"

};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export what you need
export const auth = getAuth(app);
export const db = getFirestore(app);

export default app;