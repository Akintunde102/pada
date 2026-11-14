"""Benchmark evaluation scenario suites (dev and heldout)."""

from typing import List, Dict, Any

SCENARIOS: List[Dict[str, Any]] = [
    {
        "title": "Clinical Triage & Surgical Delay (Prefix Collision + Latinate Multi-Token)",
        "situation": (
            "A 52-year-old male was rushed into the emergency room after collapsing at his desk with crushing chest pain. "
            "He signed a medical consent form before surgery, but his family is furious about the surgical delays."
        ),
        "questions": {
            "specialty": {
                "instructions": "Which medical specialty handles this?",
                "candidates": ["cardiac", "cardiology", "orthopedics", "neurology"],
                "expected": "cardiology",
                "kind": "inferential",
            },
            "urgency": {
                "instructions": "What is the clinical urgency level?",
                "expected": "critical",
                "kind": "inferential",
            },
            "facility": {
                "instructions": "Where is the patient right now?",
                "expected": "hospital",
                "kind": "inferential",
            },
            "patient_gender": {
                "instructions": "What is the gender of the patient?",
                "candidates": ["male", "female", "pediatric"],
                "expected": "male",
                "kind": "extractive",
            },
        },
    },
    {
        "title": "Financial Fraud & Banking Dispute (Disjoint Single-Word Candidates)",
        "situation": (
            "Customer reports an unauthorized $8,400 wire transfer executed from an IP in Russia while she was on a flight. "
            "The login bypassed SMS two-factor authentication via SIM swapping."
        ),
        "questions": {
            "fraud_type": {
                "instructions": "What type of attack was this?",
                "candidates": ["wire", "chargeback", "phishing", "theft"],
                "expected": "wire",
                "kind": "extractive",
            },
            "action": {
                "instructions": "What immediate action should the bank take?",
                "candidates": ["freeze", "refund", "investigate", "ignore"],
                "expected": "freeze",
                "kind": "inferential",
            },
            "risk_level": {
                "instructions": "What is the security risk level?",
                "expected": "high",
                "kind": "inferential",
            },
            "vector": {
                "instructions": "What was the technical entry vector?",
                "candidates": ["sim", "password", "malware", "phishing"],
                "expected": "sim",
                "kind": "extractive",
            },
        },
    },
    {
        "title": "E-Commerce Logistics & Defective Merchandise (Physical State + Sentiment)",
        "situation": (
            "My custom ergonomic keyboard arrived with cracked acrylic casing and three keys shattered into pieces. "
            "I need this for my tournament on Saturday and customer support has ignored me for 48 hours."
        ),
        "questions": {
            "item_state": {
                "instructions": "What is the physical condition of the item?",
                "expected": "damaged",
                "kind": "inferential",
            },
            "customer_mood": {
                "instructions": "What is the customer emotion?",
                "expected": "frustrated",
                "kind": "inferential",
            },
            "resolution": {
                "instructions": "What resolution does the customer want?",
                "candidates": ["replacement", "credit", "repair", "refund"],
                "expected": "replacement",
                "kind": "inferential",
            },
            "department": {
                "instructions": "Which department handles this ticket?",
                "candidates": ["shipping", "returns", "hardware", "billing"],
                "expected": "hardware",
                "kind": "inferential",
            },
        },
    },
    {
        "title": "Cybersecurity Incident Response (Malware Subwords + Asset Classification)",
        "situation": (
            "Endpoint detection triggered an alert on the primary domain controller: "
            "an encoded PowerShell script downloaded an obfuscated DLL from an external C2 server and attempted LSASS memory dumping."
        ),
        "questions": {
            "threat_family": {
                "instructions": "What category of threat is this?",
                "candidates": ["malware", "ransomware", "phishing", "spam"],
                "expected": "malware",
                "kind": "inferential",
            },
            "target_asset": {
                "instructions": "What primary asset was targeted?",
                "expected": "domain",
                "kind": "extractive",
            },
            "containment": {
                "instructions": "What immediate containment action should be taken?",
                "candidates": ["isolate", "reboot", "wipe", "ignore"],
                "expected": "isolate",
                "kind": "inferential",
            },
            "priority": {
                "instructions": "What is the triage priority?",
                "candidates": ["critical", "high", "medium", "low"],
                "expected": "high",
                "kind": "inferential",
            },
        },
    },
    {
        "title": "Legal Contract Review & Clause Extraction (Latinate Multi-Token Terms)",
        "situation": (
            "In the event of a material breach, the software provider shall indemnify, defend, and hold harmless the client "
            "from all third-party intellectual property and copyright infringement claims arising from use of the platform."
        ),
        "questions": {
            "clause_type": {
                "instructions": "What specific clause type is this?",
                "candidates": ["indemnification", "liability", "termination", "severability"],
                "expected": "indemnification",
                "kind": "inferential",
            },
            "provider_role": {
                "instructions": "Which entity is the provider?",
                "candidates": ["vendor", "client", "court"],
                "expected": "vendor",
                "kind": "inferential",
            },
            "subject": {
                "instructions": "What legal subject matter is covered?",
                "expected": "intellectual",
                "kind": "extractive",
            },
        },
    },
    {
        "title": "Terse / Telegraphic User Complaint (Ambiguous Short-Input Edge Case)",
        "situation": "Laptop won't turn on. Black screen after update. Need fix ASAP.",
        "questions": {
            "core_problem": {
                "instructions": "Is the fault hardware or software?",
                "candidates": ["hardware", "software"],
                "expected": "software",
                "kind": "inferential",
            },
            "device": {
                "instructions": "What hardware device?",
                "candidates": ["laptop", "phone", "tablet", "desktop"],
                "expected": "laptop",
                "kind": "extractive",
            },
            "urgency": {
                "instructions": "What is the urgency?",
                "candidates": ["high", "medium", "low"],
                "expected": "high",
                "kind": "inferential",
            },
        },
    },
    {
        "title": "Out-of-Distribution & Premise Grounding Test (Hard + Gross Negative Controls)",
        "situation": "The pediatric patient has acute bronchospasm, severe wheezing, and an oxygen saturation of 89%.",
        "questions": {
            "cardiac_check": {
                "instructions": "What cardiovascular emergency is this?",
                "candidates": ["infarction", "stroke", "arrhythmia"],
                "expected": None,
                "expect_ood": True,
                "kind": "inferential",
            },
            "gross_check": {
                "instructions": "What pulmonary condition is this?",
                "candidates": ["mortgage", "parliament", "spreadsheet"],
                "expected": None,
                "expect_ood": True,
                "kind": "inferential",
            },
            "respiratory": {
                "instructions": "What pulmonary condition is this?",
                "candidates": ["asthma", "pneumonia", "embolism"],
                "expected": "asthma",
                "expect_ood": False,
                "kind": "inferential",
            },
        },
    },
]

HELDOUT: List[Dict[str, Any]] = [
    {
        "title": "Hospitality Complaint",
        "situation": (
            "The guest checked into room 412 and found the sheets stained and the bathroom unclean. "
            "She asked the front desk to move her to another room."
        ),
        "questions": {
            "department": {
                "instructions": "Which department should receive this complaint?",
                "candidates": ["housekeeping", "kitchen", "security", "valet"],
                "expected": "housekeeping",
                "kind": "inferential",
            },
            "change": {
                "instructions": "What did she ask to change?",
                "expected": "room",
                "kind": "extractive",
            },
        },
    },
    {
        "title": "Port Logistics Delay",
        "situation": (
            "The container ship was delayed at the port of Lagos for nine days because customs "
            "inspectors found incorrect documentation on forty pallets."
        ),
        "questions": {
            "cause": {
                "instructions": "What caused the delay?",
                "candidates": ["documentation", "weather", "strike", "fuel"],
                "expected": "documentation",
                "kind": "extractive",
            },
            "vessel": {
                "instructions": "What kind of vessel is this?",
                "candidates": ["ship", "aircraft", "truck", "train"],
                "expected": "ship",
                "kind": "extractive",
            },
            "authority": {
                "instructions": "Which authority inspected the cargo?",
                "candidates": ["customs", "police", "army"],
                "expected": "customs",
                "kind": "extractive",
            },
            "mode": {
                "instructions": "What transport mode is this?",
                "candidates": ["maritime", "aerial", "rail", "road"],
                "expected": "maritime",
                "kind": "inferential",
            },
        },
    },
    {
        "title": "University Student Request",
        "situation": (
            "After failing the midterm, the student met her professor to ask whether she could "
            "retake the exam and to request extra tutoring sessions."
        ),
        "questions": {
            "person": {
                "instructions": "Who did she meet?",
                "expected": "professor",
                "kind": "extractive",
            },
            "level": {
                "instructions": "What level of education is this?",
                "candidates": ["university", "primary", "kindergarten"],
                "expected": "university",
                "kind": "inferential",
            },
        },
    },
    {
        "title": "Agricultural Loss",
        "situation": (
            "Farmers in the northern region reported that locusts destroyed most of the maize "
            "crop before the harvest season began."
        ),
        "questions": {
            "pest": {
                "instructions": "What pest caused the damage?",
                "expected": "locust",
                "kind": "extractive",
            },
            "crop": {
                "instructions": "Which crop was affected?",
                "expected": "maize",
                "kind": "extractive",
            },
            "event": {
                "instructions": "What kind of disaster is this?",
                "candidates": ["infestation", "flood", "earthquake", "fire"],
                "expected": "infestation",
                "kind": "inferential",
            },
            "sector": {
                "instructions": "What sector is affected?",
                "candidates": ["agriculture", "mining", "banking"],
                "expected": "agriculture",
                "kind": "inferential",
            },
        },
    },
]
