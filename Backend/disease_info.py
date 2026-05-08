# ==============================================================================
# SKINSIGHT — DISEASE_INFO.PY
# Refined for UI Readability (with full medical sources restored)
# ==============================================================================

DISEASE_INFO = {

    # ──────────────────────────────────────────────────────────────────────────
    # NV — Melanocytic Nevus (Common Mole)
    # ──────────────────────────────────────────────────────────────────────────
    "NV": {
        "full_name": "Melanocytic Nevus (NV)",

        "description": (
            "A melanocytic nevus (commonly known as Common mole) is a harmless, non-cancerous "
            "cluster of pigment-producing skin cells. They are extremely common and "
            "typically appear as small, round, or oval spots with a uniform brown or "
            "tan color. While almost always benign, any mole that rapidly changes should "
            "be checked by a professional."
        ),

        "tips": [
            "Apply broad-spectrum SPF 30+ sunscreen daily; UV radiation drives mole changes.",
            "Check your skin monthly using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution).",
            "Take periodic photos of your moles next to a ruler to easily track any changes over time.",
            "Wear UV-protective clothing and seek shade during peak sun hours (10 AM to 4 PM)."
        ],

        "precautions": [
            "See a dermatologist if any mole bleeds, itches, develops a crust, or changes rapidly.",
            "Schedule an annual skin exam if you have more than 50 moles or a family history of skin cancer.",
            "Never attempt to remove a mole at home using over-the-counter products or remedies."
        ],

        "sources": [
            "Mayo Clinic: Moles: Symptoms & Causes (https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200)",
            "NIH National Cancer Institute: Melanoma Treatment (https://www.cancer.gov/types/melanoma/patient/melanoma-treatment-pdq)",
            "American Academy of Dermatology: Moles (https://www.aad.org/public/diseases/a-z/moles-overview)",
            "WHO Skin Cancer Guidelines 2023 (https://www.who.int/health-topics/skin-cancer)"
        ],
    },


    # ──────────────────────────────────────────────────────────────────────────
    # MEL — Melanoma
    # ──────────────────────────────────────────────────────────────────────────
    "MEL": {
        "full_name": "Melanoma (MEL)",

        "description": (
            "Melanoma is a serious and potentially life-threatening form of skin cancer. "
            "It can develop within an existing mole or appear as a new, unusual dark lesion "
            "anywhere on the body. Early detection is absolutely critical; when caught "
            "early, it is highly curable, but it can spread to other organs if ignored."
        ),

        "tips": [
            "Perform a thorough monthly self-examination of all skin, including your scalp and the soles of your feet.",
            "Apply SPF 50+ sunscreen daily, reapplying every 2 hours when outdoors.",
            "Never use tanning beds, which drastically increase the risk of developing melanoma.",
            "Report any lesion that follows the ABCDE criteria to a doctor without delay."
        ],

        "precautions": [
            "⚠️ URGENT EVALUATION REQUIRED: Please consult a board-certified dermatologist immediately.",
            "Do not delay seeking medical care. Early-stage melanoma is highly curable with minor surgery.",
            "Document the lesion's history and bring clear, dated photographs to your appointment.",
            "Inform close family members, as melanoma risk can be hereditary."
        ],

        "sources": [
            "WHO Skin Cancer Guidelines 2023 (https://www.who.int/health-topics/skin-cancer)",
            "American Cancer Society: Melanoma Skin Cancer (https://www.cancer.org/cancer/types/melanoma-skin-cancer.html)",
            "NIH National Cancer Institute: Melanoma Treatment (https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq)",
            "Harvard Medical School: Understanding Melanoma (https://www.health.harvard.edu/cancer/melanoma-overview)",
            "IARC: Radiation: A Review of Human Carcinogens (https://publications.iarc.fr/118)",
            "PubMed: JAMA Dermatology Melanoma Epidemiology (https://jamanetwork.com/journals/jamadermatology)"
        ],
    },


    # ──────────────────────────────────────────────────────────────────────────
    # BKL — Benign Keratosis (Seborrheic Keratosis)
    # ──────────────────────────────────────────────────────────────────────────
    "BKL": {
        "full_name": "Benign Keratosis (BKL)",

        "description": (
            "Seborrheic keratosis( also known as Seborrheic Keratosis ) is a very common, entirely harmless non-cancerous skin "
            "growth that frequently appears in adults over 40. These growths look like "
            "waxy, slightly raised, 'stuck-on' patches ranging from tan to black. They "
            "do not turn into cancer and require no medical treatment."
        ),

        "tips": [
            "No medical treatment or lifestyle changes are required for these benign lesions.",
            "Avoid scratching or picking at the patches to prevent irritation or secondary infection.",
            "If friction from clothing causes discomfort, protective bandaging may help.",
            "Continue using SPF 30+ daily to protect your overall skin health."
        ],

        "precautions": [
            "Consult a dermatologist if a lesion bleeds without trauma or changes color significantly.",
            "Always have a professional confirm the diagnosis, as some cancers can mimic this appearance.",
            "If a growth becomes unsightly or bothersome, a dermatologist can safely remove it in-office."
        ],

        "sources": [
            "Mayo Clinic: Seborrheic Keratosis (https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878)",
            "American Academy of Dermatology: Seborrheic Keratoses (https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview)",
            "NIH MedlinePlus: Seborrheic Keratosis (https://medlineplus.gov/ency/article/000842.htm)",
            "PubMed: British Journal of Dermatology Seborrheic Keratosis Review (https://academic.oup.com/bjd)"
        ],
    },


    # ──────────────────────────────────────────────────────────────────────────
    # BCC — Basal Cell Carcinoma
    # ──────────────────────────────────────────────────────────────────────────
    "BCC": {
        "full_name": "Basal Cell Carcinoma (BCC)",

        "description": (
            "Basal cell carcinoma (BCC) is the most common type of skin cancer, usually "
            "developing on sun-damaged areas like the face, neck, and hands. It often "
            "appears as a pearly bump, a pink patch with raised edges, or a shiny scar-like "
            "lesion. It grows very slowly and rarely spreads, but must be treated to prevent "
            "local tissue damage."
        ),

        "tips": [
            "Apply broad-spectrum SPF 30+ sunscreen daily to all exposed skin to prevent further damage.",
            "Wear long-sleeved clothing, wide-brimmed hats, and UV-blocking sunglasses.",
            "Avoid outdoor sun exposure during peak UV hours and completely avoid indoor tanning.",
            "Perform regular skin checks for pearly bumps or sores that repeatedly heal and re-open."
        ],

        "precautions": [
            "⚠️ REQUIRES MEDICAL EVALUATION: Please see a board-certified dermatologist to confirm.",
            "Do not delay treatment. Untreated BCCs can grow deeper and require complex surgical reconstruction later.",
            "Standard treatments (like minor surgery or topical creams) are highly effective with a 95%+ cure rate.",
            "Maintain annual skin exams, as having one BCC increases your risk of developing another."
        ],

        "sources": [
            "WHO Skin Cancer Guidelines 2023 (https://www.who.int/health-topics/skin-cancer)",
            "Skin Cancer Foundation: Basal Cell Carcinoma (https://www.skincancer.org/skin-cancer-information/basal-cell-carcinoma/)",
            "Harvard Medical School: Skin Cancer (https://www.health.harvard.edu/cancer/skin-cancer-overview)",
            "NIH National Cancer Institute: Skin Cancer Treatment (https://www.cancer.gov/types/skin/patient/skin-cancer-treatment-pdq)",
            "PubMed: New England Journal of Medicine BCC Management (https://www.nejm.org/doi/full/10.1056/NEJMra1804933)",
            "American Academy of Dermatology: Basal Cell Carcinoma (https://www.aad.org/public/diseases/skin-cancer/basal-cell-carcinoma)"
        ],
    },
}