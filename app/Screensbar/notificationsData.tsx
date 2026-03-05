// ─── Shared Notifications Storage Keys ────────────────────────────────────────
export const NOTIFICATIONS_STORAGE_KEY = 'appNotifications';
export const NOTIFICATIONS_ENABLED_KEY = 'notificationsEnabled'; // true by default

// ─── Type ──────────────────────────────────────────────────────────────────────
export type AppNotification = {
  id: number;
  disease: string;
  time: string;
  image: string;
  read: boolean;
  details: {
    confidence: string;
    description: string;
    recommendation: string;
  };
};

// ─── Default Notifications (used on first load) ────────────────────────────────
export const defaultNotifications: AppNotification[] = [
  {
    id: 1,
    disease: 'Possible Melanoma',
    time: '10 minutes ago',
    image: 'https://www.dermnet.com/images/slides/melanoma/11.jpg',
    read: false,
    details: {
      confidence: '87%',
      description:
        'Melanoma is the most serious type of skin cancer. It develops in the cells (melanocytes) that produce melanin. Early detection is critical for successful treatment.',
      recommendation:
        'Please consult a dermatologist as soon as possible. Avoid sun exposure on the affected area and do not attempt self-treatment.',
    },
  },
  {
    id: 2,
    disease: 'Possible Psoriasis',
    time: '1 hour ago',
    image: 'https://www.dermnet.com/images/slides/psoriasis-of-the-scalp/3.jpg',
    read: false,
    details: {
      confidence: '79%',
      description:
        'Psoriasis is a chronic autoimmune condition that causes rapid skin cell buildup, resulting in scaling, inflammation, and redness.',
      recommendation:
        'Use prescribed topical creams and moisturizers. Avoid triggers such as stress and certain medications. Schedule a follow-up with your doctor.',
    },
  },
  {
    id: 3,
    disease: 'Possible Skin Ulcer',
    time: '2 hours ago',
    image: 'https://www.dermnet.com/images/slides/leg-ulcer/3.jpg',
    read: true,
    details: {
      confidence: '72%',
      description:
        'A skin ulcer is an open sore caused by poor circulation, pressure, or infection. It requires proper wound care to prevent further complications.',
      recommendation:
        'Keep the area clean and covered. Seek immediate medical attention to prevent infection or worsening of the wound.',
    },
  },
  {
    id: 4,
    disease: 'Possible Eczema',
    time: '5 hours ago',
    image: 'https://www.dermnet.com/images/slides/atopic-dermatitis-or-eczema/6.jpg',
    read: true,
    details: {
      confidence: '83%',
      description:
        'Eczema (atopic dermatitis) causes itchy, inflamed skin. It is common in children but can occur at any age and is often associated with allergies.',
      recommendation:
        'Apply fragrance-free moisturizer regularly. Avoid scratching, harsh soaps, and known allergens. A dermatologist can prescribe antihistamines or steroid creams.',
    },
  },
  {
    id: 5,
    disease: 'Possible Wart',
    time: 'Yesterday',
    image: 'https://www.dermnet.com/images/slides/common-wart/6.jpg',
    read: true,
    details: {
      confidence: '91%',
      description:
        'Warts are benign skin growths caused by the human papillomavirus (HPV). They are contagious through direct contact but are generally harmless.',
      recommendation:
        'Avoid picking or scratching the wart. Over-the-counter salicylic acid treatments are available. A doctor can perform cryotherapy for faster removal.',
    },
  },
  {
    id: 6,
    disease: 'Possible Ringworm',
    time: '2 days ago',
    image: 'https://www.dermnet.com/images/slides/tinea-corporis/6.jpg',
    read: true,
    details: {
      confidence: '76%',
      description:
        'Ringworm (tinea corporis) is a fungal infection causing a ring-shaped rash. Despite its name, it is not caused by a worm.',
      recommendation:
        'Apply antifungal cream as directed. Keep the area dry and clean. Avoid sharing towels or clothing with others to prevent spreading.',
    },
  },
];