"""Benchmark dataset generator for Phase 1.

Synthesizes 5 English and 5 Hindi scanned document pages with ground-truth transcripts,
metadata records, rendered high-resolution images, and independently authored evaluation questions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.core.schemas import DocumentMetadata, Question, Language, QuestionType
from src.core.logging import get_logger

logger = get_logger("dataset.generator")

# Visual parameters for document rendering (Standard A4 @ 150 DPI = 1240 x 1754 px)
PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MARGIN_X = 90
MARGIN_Y = 100

# Font resolution on Windows
ENGLISH_FONT_PATH = "C:/Windows/Fonts/arial.ttf"
DEV_FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"


def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Safely load TrueType font with fallback."""
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def render_document_image(
    title: str,
    paragraphs: List[str],
    doc_id: str,
    page_num: int,
    is_hindi: bool,
    output_path: Path,
) -> None:
    """Render a clean scanned document page with margins, header, and structured text."""
    font_file = DEV_FONT_PATH if is_hindi else ENGLISH_FONT_PATH
    title_font = get_font(font_file, 34)
    heading_font = get_font(font_file, 22)
    body_font = get_font(font_file, 20)
    footer_font = get_font(font_file, 15)

    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    # Header / Title Bar
    y_cursor = MARGIN_Y
    draw.text((MARGIN_X, y_cursor), title, fill=(20, 30, 60), font=title_font)
    y_cursor += 50

    # Decorative header rule
    draw.line([(MARGIN_X, y_cursor), (PAGE_WIDTH - MARGIN_X, y_cursor)], fill=(180, 190, 205), width=2)
    y_cursor += 35

    # Document details subtitle
    lang_label = "हिंदी (Hindi)" if is_hindi else "English"
    meta_line = f"Ref: {doc_id} | Page: {page_num} | Language: {lang_label} | Classification: Official Research Bulletin"
    draw.text((MARGIN_X, y_cursor), meta_line, fill=(90, 100, 115), font=footer_font)
    y_cursor += 45

    # Paragraph rendering with word wrap
    max_text_width = PAGE_WIDTH - 2 * MARGIN_X

    for para in paragraphs:
        if para.startswith("## "):
            # Sub-heading
            y_cursor += 15
            heading_text = para.replace("## ", "")
            draw.text((MARGIN_X, y_cursor), heading_text, fill=(30, 45, 80), font=heading_font)
            y_cursor += 40
            continue

        words = para.split(" ")
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            # Estimate width
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_text_width:
                current_line.append(word)
            else:
                if current_line:
                    draw.text((MARGIN_X, y_cursor), " ".join(current_line), fill=(25, 25, 25), font=body_font)
                    y_cursor += 34
                current_line = [word]

        if current_line:
            draw.text((MARGIN_X, y_cursor), " ".join(current_line), fill=(25, 25, 25), font=body_font)
            y_cursor += 48

    # Footer
    footer_text = f"CONFIDENTIAL RESEARCH DATASET • {doc_id} • PAGE {page_num}"
    draw.line([(MARGIN_X, PAGE_HEIGHT - 80), (PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 80)], fill=(200, 205, 215), width=1)
    draw.text((MARGIN_X, PAGE_HEIGHT - 65), footer_text, fill=(120, 130, 140), font=footer_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")


SAMPLE_DOCUMENTS = [
    # ================= ENGLISH DOCUMENTS =================
    {
        "doc_id": "doc_en_001",
        "page_id": "doc_en_001_page_001",
        "page_num": 1,
        "language": Language.ENGLISH,
        "doc_type": "government_report",
        "layout_type": "single_column",
        "title": "National Renewable Energy Transition Report 2025",
        "paragraphs": [
            "## 1. Executive Summary and Strategic Targets",
            "The Federal Ministry for Energy has formalized the National Renewable Transition Roadmap, aiming to achieve a cumulative solar energy generation capacity of 300 gigawatts by the year 2030. Rapid infrastructure expansion throughout western corridors has accelerated utility-scale grid modernization.",
            "## 2. Capital Expenditure and Regional Deployment",
            "During the fiscal year 2024, institutional capital investment reached $42.5 billion across utility-scale photovoltaic solar parks and battery storage facilities. The western desert states of Rajasthan and Gujarat accounted for over 62 percent of new generation permits.",
            "## 3. Storage Infrastructure & Transmission",
            "To mitigate diurnal intermittency, the Energy Commission has mandated the deployment of grid-scale battery energy storage systems with an aggregate capacity of 25 gigawatt-hours by 2027. High-voltage transmission lines are currently being constructed to evacuate power to central industrial zones.",
        ],
        "questions": [
            {
                "qid": "q_en_001_01",
                "question": "What is the targeted total solar energy capacity by the year 2030?",
                "answer": "300 gigawatts",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_001_02",
                "question": "How much capital investment was allocated for renewable infrastructure in 2024?",
                "answer": "$42.5 billion",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_001_03",
                "question": "Which two states accounted for over 62 percent of new solar generation permits?",
                "answer": "Rajasthan and Gujarat",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_en_001_04",
                "question": "What is the planned aggregate capacity for grid-scale battery energy storage systems?",
                "answer": "25 gigawatt-hours",
                "type": QuestionType.FACTUAL,
            },
        ],
    },
    {
        "doc_id": "doc_en_001",
        "page_id": "doc_en_001_page_002",
        "page_num": 2,
        "language": Language.ENGLISH,
        "doc_type": "government_report",
        "layout_type": "single_column",
        "title": "Offshore Wind and Grid Modernization Initiatives",
        "paragraphs": [
            "## 1. Maritime Wind Project Approvals",
            "Complementing terrestrial solar infrastructure, the Maritime Energy Directorate has approved an initial tranche of offshore wind turbines targeting a total capacity of 12.8 gigawatts. The coastal waters off Tamil Nadu and Gujarat have been prioritized for bathymetric seabed mapping.",
            "## 2. Equipment Contracts and Industrial Partnerships",
            "Major original equipment manufacturers Siemens Energy and Vestas have secured engineering contracts to manufacture 14-megawatt marine wind turbines. The project specifications require corrosion-resistant nacelles designed to withstand severe cyclonic weather systems.",
            "## 3. High-Voltage Direct Current Corridor",
            "Power generated from offshore platforms will be synchronized via undersea cables to coastal converter substations. The National Power Corporation will construct a 3,400 kilometers high-voltage direct current transmission corridor scheduled for final commissioning by December 2028.",
        ],
        "questions": [
            {
                "qid": "q_en_001_05",
                "question": "What is the targeted total capacity for offshore wind turbine generation?",
                "answer": "12.8 gigawatts",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_001_06",
                "question": "Which equipment manufacturers secured engineering contracts for marine wind turbines?",
                "answer": "Siemens Energy and Vestas",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_en_001_07",
                "question": "What is the total length of the planned high-voltage direct current transmission corridor?",
                "answer": "3,400 kilometers",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_001_08",
                "question": "By what month and year is the HVDC transmission corridor scheduled for final commissioning?",
                "answer": "December 2028",
                "type": QuestionType.FACTUAL,
            },
        ],
    },
    {
        "doc_id": "doc_en_002",
        "page_id": "doc_en_002_page_001",
        "page_num": 1,
        "language": Language.ENGLISH,
        "doc_type": "agricultural_bulletin",
        "layout_type": "single_column",
        "title": "Sustainable Agriculture & Micro-Irrigation Bulletin",
        "paragraphs": [
            "## 1. Water Conservation and Agronomic Outcomes",
            "A comprehensive field study across arid agricultural belts demonstrates that precision micro-irrigation systems have achieved a documented 44 percent reduction in total water consumption. Concurrently, average crop yield across wheat and legumes expanded by 28 percent.",
            "## 2. State Subsidies and Smallholder Coverage",
            "To incentivize technology adoption among smallholders, the National Water Mission provides a financial subsidy of 55 percent for drip and sprinkler installations to farmers owning less than two hectares. The total agricultural area converted under this policy has surpassed 1.6 million hectares.",
            "## 3. Soil Moisture Automation",
            "Participating agricultural cooperatives have integrated low-cost capacitive soil moisture telemetry. Sensors communicate via cellular telemetry to automated drip manifolds, delivering calibrated nutrient fertigation cycles directly to root systems.",
        ],
        "questions": [
            {
                "qid": "q_en_002_01",
                "question": "What percentage reduction in water consumption was achieved under micro-irrigation?",
                "answer": "44 percent",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_002_02",
                "question": "What subsidy rate is granted to farmers owning under two hectares for drip installations?",
                "answer": "55 percent",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_002_03",
                "question": "Which nodal body coordinates the financial subsidies for micro-irrigation systems?",
                "answer": "National Water Mission",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_en_002_04",
                "question": "How many hectares of agricultural land have been converted to micro-irrigation?",
                "answer": "1.6 million hectares",
                "type": QuestionType.NUMERICAL,
            },
        ],
    },
    {
        "doc_id": "doc_en_002",
        "page_id": "doc_en_002_page_002",
        "page_num": 2,
        "language": Language.ENGLISH,
        "doc_type": "agricultural_bulletin",
        "layout_type": "single_column",
        "title": "Precision Soil Health and Drone Spraying Standards",
        "paragraphs": [
            "## 1. Nationwide Soil Profiling Initiatives",
            "The Department of Agricultural Cooperation has distributed over 23 million soil health cards to farming families. Systematic laboratory soil chemistry testing indicates an average baseline organic carbon content of 0.42 percent in intensively cultivated northern plains.",
            "## 2. Nano-Urea Efficacy and Runoff Mitigation",
            "Extensive agronomical trials show that foliar application of nano-urea improves crop nitrogen absorption efficiency by 35 percent compared to traditional granular urea. This technique significantly curtails chemical runoff into municipal groundwater aquifers.",
            "## 3. Drone Subsidy and Certification Framework",
            "Under the Agricultural Modernization Scheme, registered farmer cooperatives are eligible for a financial subsidy cap of Rs 500,000 for acquiring certified spraying drones. Drones must comply with civil aviation safety protocols and operate with calibrated ultra-low volume nozzles.",
        ],
        "questions": [
            {
                "qid": "q_en_002_05",
                "question": "How many soil health cards have been issued to farming families?",
                "answer": "23 million",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_002_06",
                "question": "By what percentage does foliar nano-urea application enhance nitrogen absorption efficiency?",
                "answer": "35 percent",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_002_07",
                "question": "What is the maximum subsidy granted to cooperatives for purchasing spraying drones?",
                "answer": "Rs 500,000",
                "type": QuestionType.FACTUAL,
            },
        ],
    },
    {
        "doc_id": "doc_en_003",
        "page_id": "doc_en_003_page_001",
        "page_num": 1,
        "language": Language.ENGLISH,
        "doc_type": "scientific_report",
        "layout_type": "single_column",
        "title": "Space Research Organization Geosynchronous Telemetry Briefing",
        "paragraphs": [
            "## 1. Launch Mission Profile and Trajectory",
            "The Space Research Organization successfully placed the advanced telecommunication satellite into geostationary transfer orbit utilizing the heavy-lift GSLV Mk III launch vehicle. The spacecraft payload mass totaled exactly 4,150 kilograms at launch liftoff.",
            "## 2. Orbital Parameters and Transponder Configuration",
            "Following circularization maneuvers, the satellite established stable orbital equilibrium at an apogee altitude of 35,786 kilometers. The communications bus carries 24 active Ku-band transponders designed to deliver high-throughput broadband to isolated archipelago communities.",
            "## 3. Ground Stations and Tracking Network",
            "Primary flight telemetry and health monitoring are managed from the Satish Dhawan Space Centre in Sriharikota, supported by international deep-space ground tracking stations situated in Brunei and Indonesia.",
        ],
        "questions": [
            {
                "qid": "q_en_003_01",
                "question": "Which heavy-lift launch vehicle deployed the geostationary communication satellite?",
                "answer": "GSLV Mk III",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_en_003_02",
                "question": "What was the total spacecraft payload mass at the time of liftoff?",
                "answer": "4,150 kilograms",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_003_03",
                "question": "How many active Ku-band transponders are operational on the satellite bus?",
                "answer": "24",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_en_003_04",
                "question": "Where is the primary launch tracking and telemetry centre located?",
                "answer": "Sriharikota",
                "type": QuestionType.ENTITY,
            },
        ],
    },

    # ================= HINDI DOCUMENTS =================
    {
        "doc_id": "doc_hi_001",
        "page_id": "doc_hi_001_page_001",
        "page_num": 1,
        "language": Language.HINDI,
        "doc_type": "government_report",
        "layout_type": "single_column",
        "title": "राष्ट्रीय ग्रामीण आजीविका मिशन वार्षिक समीक्षा २०२४",
        "paragraphs": [
            "## १. मिशन का अवलोकन एवं सामाजिक प्रगति",
            "ग्रामीण विकास मंत्रालय द्वारा संचालित राष्ट्रीय ग्रामीण आजीविका मिशन के अंतर्गत अब तक देश भर में ८८ लाख महिला स्वयं सहायता समूहों का गठन किया जा चुका है। इस कार्यक्रम के माध्यम से ९.८ करोड़ ग्रामीण परिवारों को वित्तीय समावेशन के दायरे में लाया गया है।",
            "## २. संस्थागत बैंक ऋण एवं ब्याज अनुदान",
            "महिला स्वयं सहायता समूहों को आर्थिक रूप से सशक्त बनाने हेतु वाणिज्यिक बैंकों के माध्यम से कुल ₹२.४ लाख करोड़ का संस्थागत ऋण वितरित किया गया है। समय पर ऋण चुकाने वाले महिला समूहों को केवल ७ प्रतिशत की रियायती वार्षिक ब्याज दर पर ऋण उपलब्ध कराया जाता है।",
            "## ३. आजीविका संवर्धन एवं सूक्ष्म उद्यम",
            "मिशन के तहत कृषि और गैर-कृषि क्षेत्रों में मूल्य संवर्धन हेतु ग्राम स्तर पर सामुदायिक निवेश कोष की स्थापना की गई है। महिला उद्यमियों द्वारा उत्पादित स्थानीय उत्पादों को राष्ट्रीय स्तर पर विपणन के लिए सरस मेलों का नियमित आयोजन किया जाता है।",
        ],
        "questions": [
            {
                "qid": "q_hi_001_01",
                "question": "राष्ट्रीय ग्रामीण आजीविका मिशन के अंतर्गत कुल कितने महिला स्वयं सहायता समूहों का गठन किया गया है?",
                "answer": "८८ लाख",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_001_02",
                "question": "इस आजीविका योजना का संचालन भारत सरकार के किस मंत्रालय द्वारा किया जा रहा है?",
                "answer": "ग्रामीण विकास मंत्रालय",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_hi_001_03",
                "question": "महिला स्वयं सहायता समूहों को कुल कितना बैंक ऋण वितरित किया जा चुका है?",
                "answer": "₹२.४ लाख करोड़",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_001_04",
                "question": "समय पर ऋण चुकाने वाले महिला स्वयं सहायता समूहों के लिए रियायती ब्याज दर कितनी निर्धारित है?",
                "answer": "७ प्रतिशत",
                "type": QuestionType.FACTUAL,
            },
        ],
    },
    {
        "doc_id": "doc_hi_001",
        "page_id": "doc_hi_001_page_002",
        "page_num": 2,
        "language": Language.HINDI,
        "doc_type": "government_report",
        "layout_type": "single_column",
        "title": "प्रधानमंत्री कुसुम सौर पम्प एवं ग्रामीण विद्युतीकरण योजना",
        "paragraphs": [
            "## १. योजना के उद्देश्य एवं लक्ष्य",
            "नवीन और नवीकरणीय ऊर्जा मंत्रालय द्वारा शुरू की गई प्रधानमंत्री कुसुम योजना का मुख्य लक्ष्य देश भर के किसानों को सिंचाई हेतु ३५ लाख सौर कृषि पम्प उपलब्ध कराना है। इससे डीजल पम्पों पर किसानों की निर्भरता समाप्त होगी।",
            "## २. वित्तीय सहायता एवं सरकारी अनुदान",
            "इस सौर कृषि योजना के तहत केंद्र एवं राज्य सरकारें मिलकर कुल लागत का ६० प्रतिशत प्रत्यक्ष अनुदान (सब्सिडी) वहन करती हैं। बैंक द्वारा ३० प्रतिशत ऋण दिया जाता है और किसान को केवल १० प्रतिशत प्रारंभिक लागत का भुगतान करना पड़ता है।",
            "## ३. पर्यावरण संरक्षण एवं कार्बन बचत",
            "कृषि क्षेत्र में सौर ऊर्जा के प्रयोग से प्रति वर्ष लगभग ३२ लाख टन कार्बन डाइऑक्साइड उत्सर्जन में कमी दर्ज की गई है। इसके अतिरिक्त अतिरिक्त विद्युत ग्रिड को बेचकर किसान नियमित अतिरिक्त आय अर्जित कर रहे हैं।",
        ],
        "questions": [
            {
                "qid": "q_hi_001_05",
                "question": "प्रधानमंत्री कुसुम योजना के अंतर्गत कुल कितने सौर कृषि पम्प स्थापित करने का लक्ष्य निर्धारित है?",
                "answer": "३५ लाख",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_001_06",
                "question": "सौर कृषि पम्प स्थापना हेतु सरकारों द्वारा कुल कितने प्रतिशत प्रत्यक्ष अनुदान दिया जाता है?",
                "answer": "६० प्रतिशत",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_001_07",
                "question": "कुसुम योजना के कार्यान्वयन के लिए कौन सा केंद्रीय मंत्रालय उत्तरदायी है?",
                "answer": "नवीन और नवीकरणीय ऊर्जा मंत्रालय",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_hi_001_08",
                "question": "सौर पम्पों के प्रयोग से प्रति वर्ष कार्बन डाइऑक्साइड उत्सर्जन में कितनी कमी आंकी गई है?",
                "answer": "३२ लाख टन",
                "type": QuestionType.FACTUAL,
            },
        ],
    },
    {
        "doc_id": "doc_hi_002",
        "page_id": "doc_hi_002_page_001",
        "page_num": 1,
        "language": Language.HINDI,
        "doc_type": "agricultural_bulletin",
        "layout_type": "single_column",
        "title": "प्रधानमंत्री फसल बीमा योजना एवं डिजिटल दावा निपटान",
        "paragraphs": [
            "## १. फसल बीमा एवं प्रीमियम दरें",
            "कृषि एवं किसान कल्याण मंत्रालय द्वारा संचालित प्रधानमंत्री फसल बीमा योजना में किसानों के लिए न्यूनतम प्रीमियम दरें लागू हैं। खरीफ फसलों के लिए किसानों को २.० प्रतिशत तथा रबी फसलों के लिए केवल १.५ प्रतिशत प्रीमियम देना होता है। वाणिज्यिक व बागवानी फसलों के लिए यह दर ५.० प्रतिशत है।",
            "## २. डिजिटल सर्वेक्षण एवं त्वरित निपटान",
            "प्राकृतिक आपदाओं अथवा बेमौसम वर्षा से फसलों को होने वाले नुकसान का सटीक आकलन उपग्रह इमेजरी और रिमोट सेंसिंग द्वारा किया जाता है। सर्वेक्षण पूरा होने के पश्चात अधिकतम २१ दिन के भीतर बीमा दावों का निपटान सीधे किसान के बैंक खाते में किया जाता है।",
            "## ३. संचयी दावा भुगतान",
            "वर्ष २०१६ से अब तक प्राकृतिक आपदाओं से प्रभावित किसानों को कुल ₹१.२५ लाख करोड़ से अधिक मूल्य के बीमा दावों का सफल भुगतान डीबीटी (प्रत्यक्ष लाभ अंतरण) प्रणाली के माध्यम से किया जा चुका है।",
        ],
        "questions": [
            {
                "qid": "q_hi_002_01",
                "question": "खरीफ फसलों के लिए किसानों द्वारा देय फसल बीमा प्रीमियम की दर कितनी है?",
                "answer": "२.० प्रतिशत",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_002_02",
                "question": "रबी फसलों के लिए किसानों हेतु निर्धारित बीमा प्रीमियम दर क्या है?",
                "answer": "१.५ प्रतिशत",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_002_03",
                "question": "उपग्रह सर्वेक्षण के उपरांत बीमा दावा राशि के निपटान की अधिकतम समय सीमा क्या है?",
                "answer": "२१ दिन",
                "type": QuestionType.FACTUAL,
            },
            {
                "qid": "q_hi_002_04",
                "question": "योजना के तहत अब तक कुल कितने मूल्य के फसल दावों का भुगतान किया जा चुका है?",
                "answer": "₹१.२५ लाख करोड़",
                "type": QuestionType.NUMERICAL,
            },
        ],
    },
    {
        "doc_id": "doc_hi_002",
        "page_id": "doc_hi_002_page_002",
        "page_num": 2,
        "language": Language.HINDI,
        "doc_type": "education_report",
        "layout_type": "single_column",
        "title": "राष्ट्रीय डिजिटल साक्षरता एवं प्राथमिक शिक्षा अभियान",
        "paragraphs": [
            "## १. प्राथमिक शिक्षा में तकनीकी विस्तार",
            "स्कूली शिक्षा और साक्षरता विभाग द्वारा ग्रामीण प्राथमिक विद्यालयों के आधुनिकीकरण हेतु १.२ लाख विद्यालयों में डिजिटल स्मार्ट क्लासरूम स्थापित किए गए हैं। इन कक्षाओं में इंटरएक्टिव डिजिटल बोर्ड और ई-लर्निंग पाठ्यक्रम उपलब्ध हैं।",
            "## २. भारतनेट कनेक्टिविटी एवं दीक्षा मंच",
            "ग्रामीण क्षेत्रों के विद्यालयों को हाई-स्पीड ब्रॉडबैंड से जोड़ने के उद्देश्य से भारतनेट परियोजना के अंतर्गत ८५,००० ग्राम पंचायतों को ऑप्टिकल फाइबर से जोड़ा गया है। राष्ट्रीय डिजिटल मंच 'दीक्षा' (DIKSHA) पर ३२ विभिन्न भारतीय भाषाओं में उच्च गुणवत्ता की शिक्षण सामग्री निःशुल्क उपलब्ध है।",
            "## ३. बजटीय प्रावधान",
            "राष्ट्रीय शिक्षा नीति के प्रभावी क्रियान्वयन और शिक्षकों के डिजिटल प्रशिक्षण के लिए वार्षिक केंद्रीय बजट में ₹४७,००० करोड़ का विशेष आवंटन सुनिश्चित किया गया है।",
        ],
        "questions": [
            {
                "qid": "q_hi_002_05",
                "question": "देश भर के कितने ग्रामीण प्राथमिक विद्यालयों में डिजिटल स्मार्ट क्लासरूम स्थापित किए गए हैं?",
                "answer": "१.२ लाख",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_002_06",
                "question": "३२ भारतीय भाषाओं में निःशुल्क डिजिटल शिक्षण सामग्री उपलब्ध कराने वाले राष्ट्रीय मंच का क्या नाम है?",
                "answer": "दीक्षा (DIKSHA)",
                "type": QuestionType.ENTITY,
            },
            {
                "qid": "q_hi_002_07",
                "question": "डिजिटल साक्षरता और प्राथमिक शिक्षा अभियान हेतु वार्षिक केंद्रीय बजट में कितना आवंटन किया गया है?",
                "answer": "₹४७,००० करोड़",
                "type": QuestionType.NUMERICAL,
            },
        ],
    },
    {
        "doc_id": "doc_hi_003",
        "page_id": "doc_hi_003_page_001",
        "page_num": 1,
        "language": Language.HINDI,
        "doc_type": "environmental_policy",
        "layout_type": "single_column",
        "title": "भारतीय वन्यजीव संरक्षण एवं हरित आवरण नीति रिपोर्ट",
        "paragraphs": [
            "## १. प्रोजेक्ट टाइगर एवं वन्यजीव स्थिति",
            "पर्यावरण, वन और जलवायु परिवर्तन मंत्रालय की नवीनतम गणना के अनुसार भारत में बाघों की कुल अनुमानित संख्या बढ़कर ३,६८२ हो गई है। देश भर में अब कुल ५४ अधिसूचित बाघ अभयारण्य (टाइगर रिजर्व) कार्यरत हैं जो वैश्विक बाघ आबादी के ७० प्रतिशत हिस्से को आवास प्रदान करते हैं।",
            "## २. राष्ट्रीय हरित आवरण एवं वन क्षेत्र",
            "भारतीय वन सर्वेक्षण रिपोर्ट के अनुसार देश के कुल भौगोलिक क्षेत्रफल का २४.६२ प्रतिशत हिस्सा वनों और वृक्षों से आच्छादित है। वनावरण में निरंतर वृद्धि हेतु राष्ट्रीय वनीकरण कार्यक्रम के तहत सघन पौधरोपण किया जा रहा है।",
            "## ३. पर्यावरण संवेदनशील क्षेत्र एवं एनजीटी नियम",
            "राष्ट्रीय हरित न्यायाधिकरण (NGT) के वैधानिक निर्देशों के अनुसार राष्ट्रीय उद्यानों और अभयारण्यों के चारों ओर १० किलोमीटर तक के क्षेत्र को पर्यावरण संवेदनशील क्षेत्र (Eco-Sensitive Zone) के रूप में विनियमित किया जाता है ताकि अनियंत्रित खनन और वाणिज्यिक निर्माण पर रोक लगाई जा सके।",
        ],
        "questions": [
            {
                "qid": "q_hi_003_01",
                "question": "नवीनतम वन्यजीव गणना रिपोर्ट के अनुसार भारत में बाघों की कुल अनुमानित संख्या कितनी है?",
                "answer": "३,६८२",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_003_02",
                "question": "भारत में वर्तमान में कुल कितने अधिसूचित बाघ अभयारण्य (टाइगर रिजर्व) कार्यरत हैं?",
                "answer": "५४",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_003_03",
                "question": "भारत के कुल भौगोलिक क्षेत्रफल का कितने प्रतिशत भाग वनों और वृक्षों से आच्छादित है?",
                "answer": "२४.६२ प्रतिशत",
                "type": QuestionType.NUMERICAL,
            },
            {
                "qid": "q_hi_003_04",
                "question": "पर्यावरण नियमों के उल्लंघन की निगरानी और रोकथाम कौन सा न्यायिक निकाय करता है?",
                "answer": "राष्ट्रीय हरित न्यायाधिकरण (NGT)",
                "type": QuestionType.ENTITY,
            },
        ],
    },
]


def generate_benchmark_dataset(base_dir: Path | str = ".") -> Tuple[int, int]:
    """Generate all document images, ground-truth text, metadata, and questions."""
    base = Path(base_dir)
    img_dir = base / "data/images"
    gt_dir = base / "data/ground_truth"
    meta_path = base / "data/metadata/documents.json"
    questions_path = base / "data/questions/questions.json"

    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_list: List[DocumentMetadata] = []
    questions_list: List[Question] = []

    for entry in SAMPLE_DOCUMENTS:
        doc_id = entry["doc_id"]
        page_id = entry["page_id"]
        page_num = entry["page_num"]
        lang = entry["language"]
        is_hindi = (lang == Language.HINDI)

        # 1. Construct Ground Truth text
        full_gt_text = entry["title"] + "\n\n" + "\n\n".join(entry["paragraphs"])
        gt_file = gt_dir / f"{page_id}.txt"
        gt_file.write_text(full_gt_text, encoding="utf-8")

        # 2. Render Document Image
        img_file = img_dir / f"{page_id}.png"
        render_document_image(
            title=entry["title"],
            paragraphs=entry["paragraphs"],
            doc_id=doc_id,
            page_num=page_num,
            is_hindi=is_hindi,
            output_path=img_file,
        )

        # 3. Create DocumentMetadata
        rel_img_path = f"data/images/{page_id}.png"
        rel_gt_path = f"data/ground_truth/{page_id}.txt"
        meta = DocumentMetadata(
            document_id=doc_id,
            page_id=page_id,
            language=lang,
            page_number=page_num,
            doc_type=entry["doc_type"],
            layout_type=entry["layout_type"],
            image_quality="clean",
            image_path=rel_img_path,
            ground_truth_path=rel_gt_path,
        )
        metadata_list.append(meta)

        # 4. Create Questions
        for q_data in entry["questions"]:
            q_obj = Question(
                question_id=q_data["qid"],
                document_id=doc_id,
                page_id=page_id,
                language=lang,
                question=q_data["question"],
                expected_answer=q_data["answer"],
                source_page=page_num,
                question_type=q_data["type"],
            )
            questions_list.append(q_obj)

    # Save metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in metadata_list], f, ensure_ascii=False, indent=2)

    # Save questions JSON
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump([q.model_dump() for q in questions_list], f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(metadata_list)} pages and {len(questions_list)} questions.")
    return len(metadata_list), len(questions_list)


if __name__ == "__main__":
    generate_benchmark_dataset()
