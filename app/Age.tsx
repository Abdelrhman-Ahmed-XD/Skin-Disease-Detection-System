import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
import {
  FlatList,
  NativeScrollEvent,
  NativeSyntheticEvent,
  Pressable,
  StyleSheet,
  Text, TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const STORAGE_KEY = "signupDraft";
const ITEM_HEIGHT = 50;

export default function Age() {
  const [dateSelected, setDateSelected] = useState(false);

  // نستخدم ref للقيم الحالية عشان نحفظها صح فوراً
  const dayIdxRef   = useRef(0);
  const monthIdxRef = useRef(0);
  const yearIdxRef  = useRef(21);

  const [dayIndex,   setDayIndex]   = useState(0);
  const [monthIndex, setMonthIndex] = useState(0);
  const [yearIndex,  setYearIndex]  = useState(21);

  const dayRef   = useRef<FlatList>(null);
  const monthRef = useRef<FlatList>(null);
  const yearRef  = useRef<FlatList>(null);

  const days   = Array.from({ length: 31 }, (_, i) => i + 1);
  const months = Array.from({ length: 12 }, (_, i) => i + 1);
  const years  = Array.from({ length: 111 }, (_, i) => 1970 + i);

  // ── Load saved date ───────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (saved) {
          const data = JSON.parse(saved);
          if (data.birthDay !== undefined) {
            const dIdx = days.indexOf(data.birthDay);
            const mIdx = months.indexOf(data.birthMonth);
            const yIdx = years.indexOf(data.birthYear);
            if (dIdx >= 0) {
              setDayIndex(dIdx); dayIdxRef.current = dIdx;
              setTimeout(() => dayRef.current?.scrollToIndex({ index: dIdx, animated: false }), 150);
            }
            if (mIdx >= 0) {
              setMonthIndex(mIdx); monthIdxRef.current = mIdx;
              setTimeout(() => monthRef.current?.scrollToIndex({ index: mIdx, animated: false }), 150);
            }
            if (yIdx >= 0) {
              setYearIndex(yIdx); yearIdxRef.current = yIdx;
              setTimeout(() => yearRef.current?.scrollToIndex({ index: yIdx, animated: false }), 150);
            }
            setDateSelected(true);
          }
        }
      } catch (err) {
        console.log("Age load error:", err);
      }
    };
    load();
  }, []);

  // ── Save date — يستخدم الـ ref مباشرة ────────────────────
  const saveDate = async () => {
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      const data  = saved ? JSON.parse(saved) : {};
      await AsyncStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            ...data,
            birthDay:   days[dayIdxRef.current],
            birthMonth: months[monthIdxRef.current],
            birthYear:  years[yearIdxRef.current],
          })
      );
    } catch (err) {
      console.log("Age save error:", err);
    }
  };

  const handleScroll = (
      e: NativeSyntheticEvent<NativeScrollEvent>,
      type: "day" | "month" | "year"
  ) => {
    const raw = Math.round(e.nativeEvent.contentOffset.y / ITEM_HEIGHT);

    // Clamp index to valid bounds — prevents crash when scrolling past end
    if (type === "day") {
      const index = Math.min(Math.max(raw, 0), days.length - 1);
      setDayIndex(index); dayIdxRef.current = index;
    } else if (type === "month") {
      const index = Math.min(Math.max(raw, 0), months.length - 1);
      setMonthIndex(index); monthIdxRef.current = index;
    } else if (type === "year") {
      const index = Math.min(Math.max(raw, 0), years.length - 1);
      setYearIndex(index); yearIdxRef.current = index;
    }

    setDateSelected(true);
    saveDate();
  };

  const renderItem = (item: number, index: number, selectedIndex: number) => (
      <View style={styles.item}>
        <Text style={[styles.itemText, index === selectedIndex && styles.selectedText]}>
          {item}
        </Text>
      </View>
  );

  return (
      <SafeAreaView style={{ flex: 1, backgroundColor: "#DDF0F6" }}>
        <View style={{ margin: 20 }}>
          <Pressable onPress={() => router.back()} style={styles.backBtn}>
            <Ionicons name="arrow-back" size={28} color="black" />
          </Pressable>

          <View style={styles.topRow}>
            {!dateSelected && (
                <TouchableOpacity onPress={() => router.push("/Skin")}>
                  <Text style={styles.skip}>Skip</Text>
                </TouchableOpacity>
            )}
            <Text style={styles.percent}>{dateSelected ? "40%" : "20%"}</Text>
          </View>

          <View style={styles.progressBg}>
            <View style={[styles.progressFill, { width: dateSelected ? "40%" : "20%" }]} />
          </View>

          <Text style={styles.title}>How old are you?</Text>
          <Text style={styles.subtitle}>Providing your age helps us keep your information complete</Text>

          <View style={styles.header}>
            <Text style={styles.headerText}>Day</Text>
            <Text style={styles.headerText}>Month</Text>
            <Text style={styles.headerText}>Year</Text>
          </View>

          <View style={styles.highlight} />

          <View style={styles.pickerContainer}>
            <FlatList
                ref={dayRef}
                data={days}
                keyExtractor={(item) => item.toString()}
                showsVerticalScrollIndicator={false}
                snapToInterval={ITEM_HEIGHT}
                decelerationRate="fast"
                contentContainerStyle={{ paddingVertical: ITEM_HEIGHT }}
                onMomentumScrollEnd={(e) => handleScroll(e, "day")}
                getItemLayout={(_, index) => ({ length: ITEM_HEIGHT, offset: ITEM_HEIGHT * index, index })}
                renderItem={({ item, index }) => renderItem(item, index, dayIndex)}
            />
            <FlatList
                ref={monthRef}
                data={months}
                keyExtractor={(item) => item.toString()}
                showsVerticalScrollIndicator={false}
                snapToInterval={ITEM_HEIGHT}
                decelerationRate="fast"
                contentContainerStyle={{ paddingVertical: ITEM_HEIGHT }}
                onMomentumScrollEnd={(e) => handleScroll(e, "month")}
                getItemLayout={(_, index) => ({ length: ITEM_HEIGHT, offset: ITEM_HEIGHT * index, index })}
                renderItem={({ item, index }) => renderItem(item, index, monthIndex)}
            />
            <FlatList
                ref={yearRef}
                data={years}
                keyExtractor={(item) => item.toString()}
                showsVerticalScrollIndicator={false}
                snapToInterval={ITEM_HEIGHT}
                decelerationRate="fast"
                contentContainerStyle={{ paddingVertical: ITEM_HEIGHT }}
                onMomentumScrollEnd={(e) => handleScroll(e, "year")}
                getItemLayout={(_, index) => ({ length: ITEM_HEIGHT, offset: ITEM_HEIGHT * index, index })}
                renderItem={({ item, index }) => renderItem(item, index, yearIndex)}
            />
          </View>

          <TouchableOpacity
              onPress={() => router.push("/Skin")}
              disabled={!dateSelected}
              style={[styles.continueBtn, !dateSelected && styles.continueDisabled]}
          >
            <Text style={styles.continueText}>Continue</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  backBtn: {
    width: 42, height: 42, backgroundColor: "#D8E9F0",
    borderRadius: 15, borderWidth: 0.5,
    justifyContent: "center", alignItems: "center", elevation: 3,
  },
  topRow: { flexDirection: "row", marginTop: 40 },
  skip: { color: "#0B4F6C", fontWeight: "600" },
  percent: { marginLeft: "auto", fontWeight: "600" },
  progressBg: { height: 6, backgroundColor: "#ccc", borderRadius: 10, marginVertical: 10 },
  progressFill: { height: "100%", backgroundColor: "#0B4F6C", borderRadius: 10 },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginTop: 30 },
  subtitle: { textAlign: "center", marginTop: 10, color: "#000" },
  header: {
    flexDirection: "row", backgroundColor: "#1976A3",
    borderRadius: 8, marginTop: 30, paddingVertical: 10,
  },
  headerText: { flex: 1, textAlign: "center", color: "#fff", fontWeight: "bold" },
  pickerContainer: {
    flexDirection: "row", height: ITEM_HEIGHT * 3,
    backgroundColor: "#D8E9F0", marginTop: 10, zIndex: 1, position: "relative",
  },
  highlight: {
    position: "absolute", top: "64%", left: 0, right: 0,
    height: ITEM_HEIGHT, backgroundColor: "#000", opacity: 0.1,
    borderRadius: 10, elevation: 4, shadowOpacity: 1,
    shadowColor: "#000", shadowRadius: 10, zIndex: 2,
  },
  item: { height: ITEM_HEIGHT, justifyContent: "center", alignItems: "center", zIndex: 3 },
  itemText: { fontSize: 18, color: "#000" },
  selectedText: { fontSize: 20, color: "#000", fontWeight: "bold" },
  continueBtn: { backgroundColor: "#004F7F", marginTop: 50, padding: 16, borderRadius: 12 },
  continueDisabled: { backgroundColor: "#aaa" },
  continueText: { color: "#fff", textAlign: "center", fontWeight: "bold" },
});