// ════════════════════════════════════════════════════════════
// HOW TO USE CustomizeContext IN YOUR APP
// ════════════════════════════════════════════════════════════

// ─── STEP 1: Wrap your root layout with CustomizeProvider ────
// File: app/_layout.tsx  (or wherever your root layout is)

/*
import { CustomizeProvider } from '../Settingsoptions/CustomizeContext';

export default function RootLayout() {
  return (
    <CustomizeProvider>
      <ThemeProvider>
        <Stack />
      </ThemeProvider>
    </CustomizeProvider>
  );
}
*/

// ─── STEP 2: Use the hook in ANY screen ──────────────────────
// Example: How to apply font size + text color + bg color in FirstHomePage

/*
import { useCustomize } from '../Settingsoptions/CustomizeContext';

export default function FirstHomePage() {
  const { settings } = useCustomize();

  // Use settings.fontSize    → for Text fontSize
  // Use settings.textColor   → for Text color
  // Use settings.backgroundColor → for View/SafeAreaView background
  // Use settings.fontFamily  → for Text fontFamily

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: settings.backgroundColor }}>
      <Text style={{
        fontSize:   settings.fontSize,
        color:      settings.textColor,
        fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
      }}>
        Hello World
      </Text>
    </SafeAreaView>
  );
}
*/

// ─── STEP 3: Helper style function (optional) ────────────────
// You can create a helper to quickly get text styles anywhere:

/*
import { useCustomize } from '../Settingsoptions/CustomizeContext';

export function useAppStyles() {
  const { settings } = useCustomize();
  return {
    text: {
      fontSize:   settings.fontSize,
      color:      settings.textColor,
      fontFamily: settings.fontFamily === 'System' ? undefined : settings.fontFamily,
    },
    container: {
      backgroundColor: settings.backgroundColor,
    },
  };
}

// Then in any component:
const { text, container } = useAppStyles();
<SafeAreaView style={[styles.container, container]}>
  <Text style={[styles.title, text]}>Welcome</Text>
</SafeAreaView>
*/

export { };
