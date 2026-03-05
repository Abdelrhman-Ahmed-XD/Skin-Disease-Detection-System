import { useCallback } from 'react';

export type Language = 'English' | 'Arabic';

const dict = {
  English: {
    // Common
    back: 'Back', save: 'Save', cancel: 'Cancel', confirm: 'Confirm',
    continue: 'Continue', change: 'Change', delete: 'Delete',
    yes: 'Yes', no: 'No', ok: 'OK', error: 'Error', loading: 'Loading...',
    done: 'Done', verify: 'Verify', verified: 'Verified ✓',

    // Navigation tabs
    home: 'Home', reportsTab: 'Reports', historyTab: 'History',
    settingsTab: 'Settings',

    // Settings
    settings: 'Settings', preferences: 'Preferences', app: 'App',
    notifications: 'Notifications', darkMode: 'Dark Mode',
    customize: 'Customize', about: 'About', helpSupport: 'Help & Support',
    logout: 'Logout', logoutConfirmTitle: 'Are you sure you want to logout?',
    stay: 'Stay',

    // Customize
    language: 'Language', fontType: 'Font Type', fontSize: 'Font Size',
    textColor: 'Text Color', backgroundColor: 'Background Color',
    preview: 'PREVIEW', previewText: 'This is how your text will look.',
    customizeSaved: 'Saved!', customizeSavedMsg: 'Your preferences have been updated.',

    // Password
    changePassword: 'Change Password',
    currentPassword: 'Current Password',
    newPassword: 'New Password',
    confirmNewPassword: 'Confirm New Password',
    enterCurrentPassword: 'Enter current password',
    enterNewPassword: 'Enter new password',
    confirmYourPassword: 'Confirm your password',
    passwordChangedSuccess: 'Password Changed Successfully!',
    changePasswordConfirm: 'Are you sure you want to change your password?',
    passwordMin: 'Password must be at least 8 characters',
    passwordUppercase: 'Must contain an uppercase letter',
    passwordLowercase: 'Must contain a lowercase letter',
    passwordNumber: 'Must contain a number',
    passwordSpecial: 'Must contain a special character',
    passwordsNotMatch: 'Passwords do not match',
    eightChars: 'At least 8 characters',
    uppercaseLetter: 'Uppercase letter',
    lowercaseLetter: 'Lowercase letter',
    numberDigit: 'Number (0-9)',
    specialChar: 'Special character (!@#$...)',

    // Profile
    editProfile: 'Edit Profile',
    firstName: 'First Name', lastName: 'Last Name',
    email: 'Email', age: 'Date of Birth', gender: 'Gender',
    skinTone: 'Skin Tone', eyeColor: 'Eye Color', hairColor: 'Hair Color',
    male: 'Male', female: 'Female',
    enterFirstName: 'Enter first name', enterLastName: 'Enter last name',
    enterEmail: 'Enter email address',
    chooseGender: 'Choose gender',
    chooseSkinTone: 'Choose skin tone',
    chooseEyeColor: 'Choose eye color',
    chooseHairColor: 'Choose hair color',
    notSet: 'Not set',
    selectDate: 'Select Date',
    validEmail: 'Please enter a valid email',
    profilePhoto: 'Profile Photo',
    chooseOption: 'Choose an option',
    camera: 'Camera', gallery: 'Gallery',
    profileSaved: 'Profile Saved!',
    profileUpdated: 'Your profile has been updated successfully.',
    firstLastRequired: 'First and last name are required.',
    emailNotVerified: 'Email Not Verified',
    verifyEmailFirst: 'Please verify your email before saving.',
    tryAgain: 'Something went wrong. Please try again.',

    // Home
    welcome: 'Welcome,', letsCheck: "Let's check your",
    skin: 'Skin', front: 'Front', Back: 'Back',
    deletePoint: 'Delete Point', areYouSure: 'Are you sure?',

    // History
    history: 'History', noHistoryYet: 'No History Yet',
    noHistorySubtitle: 'Start scanning your skin to build your history.',
    entry: 'entry', entriesFound: 'entries found',
    entryNum: 'Entry #', analyzed: 'Analyzed',
    frontBody: 'Front', backBody: 'Back',
    deleteEntry: 'Delete Entry',
    deleteEntryConfirm: 'Are you sure you want to delete this entry?',

    // Reports
    reports: 'Reports', noReportsYet: 'No Reports Yet',
    noReportsSubtitle: 'Complete a skin scan to generate your first report.',
    noReportsToDownload: 'No reports available to download.',
    reportNum: 'Report #', downloadPDF: 'Download PDF',
    downloadAll: 'Download All Reports',
    loadingReports: 'Loading reports...',
    analysisInProgress: 'Analysis in progress...',

    // Report Details
    reportDetails: 'Report Details',
    location: 'Location', coordinates: 'Coordinates', reportId: 'Report ID',
    analysisResults: 'Analysis Results',
    downloadAsPDF: 'Download as PDF',
    downloadedSuccess: 'Downloaded Successfully',
    medicalDisclaimer: 'This report is for informational purposes only and does not replace professional medical advice.',

    // Notifications
    notificationsDisabled: 'Notifications Disabled',
    notificationsDisabledSub: 'Enable notifications in Settings to receive skin analysis alerts and reminders.',
    goToSettings: 'Go to Settings',
    markAllRead: 'Mark all as read',
    skinDiseaseDetected: 'Skin Disease Detected',
    aiConfidence: 'AI Confidence:',
    aboutCondition: 'About this condition',
    recommendation: 'Recommendation',
    consultDoctor: 'Consult a Doctor',

    // About
    aboutUs: 'About Us', appName: 'Skin Disease Detection',
    usingAI: 'Using Artificial Intelligence',
    keyFeatures: 'Key Features',

    // Help
    help: 'Help & Support',
    faqSubtitle: 'Frequently asked questions — tap a question to see the answer.',
  },

  Arabic: {
    // Common
    back: 'رجوع', save: 'حفظ', cancel: 'إلغاء', confirm: 'تأكيد',
    continue: 'متابعة', change: 'تغيير', delete: 'حذف',
    yes: 'نعم', no: 'لا', ok: 'حسناً', error: 'خطأ', loading: 'جارٍ التحميل...',
    done: 'تم', verify: 'تحقق', verified: 'تم التحقق ✓',

    // Navigation tabs
    home: 'الرئيسية', reportsTab: 'التقارير', historyTab: 'السجل',
    settingsTab: 'الإعدادات',

    // Settings
    settings: 'الإعدادات', preferences: 'التفضيلات', app: 'التطبيق',
    notifications: 'الإشعارات', darkMode: 'الوضع الداكن',
    customize: 'تخصيص', about: 'حول', helpSupport: 'المساعدة والدعم',
    logout: 'تسجيل الخروج',
    logoutConfirmTitle: 'هل أنت متأكد أنك تريد تسجيل الخروج؟',
    stay: 'ابقَ',

    // Customize
    language: 'اللغة', fontType: 'نوع الخط', fontSize: 'حجم الخط',
    textColor: 'لون النص', backgroundColor: 'لون الخلفية',
    preview: 'معاينة', previewText: 'هكذا سيبدو النص.',
    customizeSaved: 'تم الحفظ!', customizeSavedMsg: 'تم تحديث تفضيلاتك.',

    // Password
    changePassword: 'تغيير كلمة المرور',
    currentPassword: 'كلمة المرور الحالية',
    newPassword: 'كلمة المرور الجديدة',
    confirmNewPassword: 'تأكيد كلمة المرور الجديدة',
    enterCurrentPassword: 'أدخل كلمة المرور الحالية',
    enterNewPassword: 'أدخل كلمة المرور الجديدة',
    confirmYourPassword: 'أكد كلمة مرورك',
    passwordChangedSuccess: 'تم تغيير كلمة المرور بنجاح!',
    changePasswordConfirm: 'هل أنت متأكد أنك تريد تغيير كلمة المرور؟',
    passwordMin: 'كلمة المرور يجب أن تكون 8 أحرف على الأقل',
    passwordUppercase: 'يجب أن تحتوي على حرف كبير',
    passwordLowercase: 'يجب أن تحتوي على حرف صغير',
    passwordNumber: 'يجب أن تحتوي على رقم',
    passwordSpecial: 'يجب أن تحتوي على رمز خاص',
    passwordsNotMatch: 'كلمتا المرور غير متطابقتين',
    eightChars: '8 أحرف على الأقل',
    uppercaseLetter: 'حرف كبير',
    lowercaseLetter: 'حرف صغير',
    numberDigit: 'رقم (0-9)',
    specialChar: 'رمز خاص (!@#$...)',

    // Profile
    editProfile: 'تعديل الملف الشخصي',
    firstName: 'الاسم الأول', lastName: 'اسم العائلة',
    email: 'البريد الإلكتروني', age: 'تاريخ الميلاد', gender: 'الجنس',
    skinTone: 'لون البشرة', eyeColor: 'لون العيون', hairColor: 'لون الشعر',
    male: 'ذكر', female: 'أنثى',
    enterFirstName: 'أدخل الاسم الأول', enterLastName: 'أدخل اسم العائلة',
    enterEmail: 'أدخل البريد الإلكتروني',
    chooseGender: 'اختر الجنس',
    chooseSkinTone: 'اختر لون البشرة',
    chooseEyeColor: 'اختر لون العيون',
    chooseHairColor: 'اختر لون الشعر',
    notSet: 'غير محدد',
    selectDate: 'اختر التاريخ',
    validEmail: 'أدخل بريداً إلكترونياً صحيحاً',
    profilePhoto: 'صورة الملف الشخصي',
    chooseOption: 'اختر خياراً',
    camera: 'الكاميرا', gallery: 'المعرض',
    profileSaved: 'تم حفظ الملف!',
    profileUpdated: 'تم تحديث ملفك الشخصي بنجاح.',
    firstLastRequired: 'الاسم الأول واسم العائلة مطلوبان.',
    emailNotVerified: 'البريد غير محقق',
    verifyEmailFirst: 'يرجى التحقق من بريدك قبل الحفظ.',
    tryAgain: 'حدث خطأ. يرجى المحاولة مرة أخرى.',

    // Home
    welcome: 'مرحباً،', letsCheck: 'دعنا نفحص',
    skin: 'جلدك', front: 'الأمام', Back: 'الخلف',
    deletePoint: 'حذف النقطة', areYouSure: 'هل أنت متأكد؟',

    // History
    history: 'السجل', noHistoryYet: 'لا يوجد سجل بعد',
    noHistorySubtitle: 'ابدأ بفحص جلدك لبناء سجلك.',
    entry: 'إدخال', entriesFound: 'إدخالات',
    entryNum: 'إدخال #', analyzed: 'تم التحليل',
    frontBody: 'الأمام', backBody: 'الخلف',
    deleteEntry: 'حذف الإدخال',
    deleteEntryConfirm: 'هل أنت متأكد أنك تريد حذف هذا الإدخال؟',

    // Reports
    reports: 'التقارير', noReportsYet: 'لا توجد تقارير بعد',
    noReportsSubtitle: 'أكمل فحص الجلد لإنشاء أول تقرير.',
    noReportsToDownload: 'لا توجد تقارير متاحة للتنزيل.',
    reportNum: 'تقرير #', downloadPDF: 'تنزيل PDF',
    downloadAll: 'تنزيل جميع التقارير',
    loadingReports: 'جارٍ تحميل التقارير...',
    analysisInProgress: 'التحليل قيد التنفيذ...',

    // Report Details
    reportDetails: 'تفاصيل التقرير',
    location: 'الموقع', coordinates: 'الإحداثيات', reportId: 'رقم التقرير',
    analysisResults: 'نتائج التحليل',
    downloadAsPDF: 'تنزيل بصيغة PDF',
    downloadedSuccess: 'تم التنزيل بنجاح',
    medicalDisclaimer: 'هذا التقرير لأغراض إعلامية فقط ولا يغني عن الاستشارة الطبية.',

    // Notifications
    notificationsDisabled: 'الإشعارات معطلة',
    notificationsDisabledSub: 'فعّل الإشعارات في الإعدادات لتلقي تنبيهات تحليل الجلد.',
    goToSettings: 'اذهب إلى الإعدادات',
    markAllRead: 'تحديد الكل كمقروء',
    skinDiseaseDetected: 'تم اكتشاف مرض جلدي',
    aiConfidence: 'ثقة الذكاء الاصطناعي:',
    aboutCondition: 'عن هذه الحالة',
    recommendation: 'التوصية',
    consultDoctor: 'استشر طبيباً',

    // About
    aboutUs: 'حول التطبيق', appName: 'تطبيق كشف أمراض الجلد',
    usingAI: 'باستخدام الذكاء الاصطناعي',
    keyFeatures: 'المميزات الرئيسية',

    // Help
    help: 'المساعدة والدعم',
    faqSubtitle: 'الأسئلة الشائعة — اضغط على السؤال لرؤية الإجابة.',
  },
} as const;

type Dict = typeof dict.English;
type TranslationKey = keyof Dict;

export function useTranslation(language: Language) {
  const isArabic = language === 'Arabic';
  const translations = isArabic ? dict.Arabic : dict.English;

  const t = useCallback(
    (key: TranslationKey): string => {
      return (translations as Dict)[key] ?? key;
    },
    [translations]
  );

  return { t, isArabic };
}