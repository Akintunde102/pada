# ==============================================================================
# benchmark_v6.py — Held-Out 105-Check Benchmark for v6 (Modes 1, 2, 3 — All Open)
#   - Mode 1 (State-Only): 15 situations (15 checks)
#   - Mode 2 (Tags)      : 12 situations, 1..7 tags each (45 checks)
#   - Mode 3 (Questions) : 12 situations, 1..6 questions each, NO options (45 checks)
#   - Total              : 105 checks + Head-to-Head Model Comparison (1.5B vs 3B)
# ==============================================================================
import gc
import torch
import full

# 1. MODE 1 HELD-OUT SUITE (15 situations -> 15 checks)
MODE1_SUITE = [
    ("The container ship was delayed at the port of Lagos for nine days because customs inspectors found incorrect documentation.",
     {"delay", "delayed", "customs", "shipping", "cargo", "port", "documentation"}),
    ("Laptop won't turn on. Black screen after update. Need fix ASAP.",
     {"fix", "repair", "laptop", "crash", "broken", "update", "failure", "urgent", " malfunction"}),
    ("I hate you",
     {"hate", "hatred", "hostility", "anger", "insult", "dislike"}),
    ("The village council voted to build a new bridge over the river after the flood destroyed the old one.",
     {"bridge", "flood", "rebuild", "vote", "reconstruction", "infrastructure", "council"}),
    ("She practiced the violin for six hours every day before the national competition.",
     {"practice", "violin", "music", "competition", "dedication", "rehearsal", "musician"}),
    ("Surgeons completed a twelve-hour heart transplant on the newborn infant.",
     {"surgery", "transplant", "medical", "heart", "operation", "hospital", "medicine"}),
    ("A massive swarm of locusts devoured three thousand hectares of wheat crops in two days.",
     {"locusts", "famine", "pest", "pests", "crops", "agriculture", "destruction", "swarm", "infestation"}),
    ("The central bank raised interest rates by fifty basis points to curb rising inflation.",
     {"inflation", "finance", "economy", "interest", "banking", "rates", "monetary"}),
    ("Firefighters evacuated two hundred residents after a gas pipeline ruptured beneath the street.",
     {"evacuation", "fire", "emergency", "gas", "leak", "rupture", "hazard", "disaster"}),
    ("The rover drilled into the Martian crater and detected traces of ancient organic molecules.",
     {"mars", "space", "discovery", "astronomy", "science", "rover", "exploration", "planet"}),
    ("Pickpockets stole the tourist's passport and wallet while he was boarding the crowded subway train.",
     {"theft", "robbery", "crime", "stolen", "pickpocket", "stealing", "subway"}),
    ("The airline cancelled forty flights after dense fog reduced visibility at the airport to zero.",
     {"fog", "weather", "cancellation", "cancelled", "flights", "aviation", "delay", "airport"}),
    ("Archaeologists uncovered a bronze chariot buried beside the tomb of an ancient emperor.",
     {"archaeology", "discovery", "excavation", "history", "artifact", "tomb", "ancient"}),
    ("The tenant refused to pay rent until the landlord repaired the leaking roof and broken heater.",
     {"dispute", "rent", "housing", "tenant", "repair", "landlord", "complaint", "maintenance"}),
    ("Three climbers were stranded on a narrow icy ledge after an avalanche blocked their descent.",
     {"avalanche", "rescue", "stranded", "climbing", "mountain", "emergency", "danger", "mountaineering"}),
]

# 2. MODE 2 HELD-OUT SUITE (12 situations, counts: 4, 1, 6, 2, 3, 5, 7, 2, 4, 1, 5, 5 -> 45 checks)
MODE2_SUITE = [
    ("2.01 (4 tags) Hostile short text",
     "I hate you",
     ["tone", "aggression", "f word", "n word"],
     {
         "tone": {"hostile", "negative", "angry", "hateful"},
         "aggression": {"high", "extreme", "severe", "yes"},
         "f word": {"no", "none"},
         "n word": {"no", "none"},
     }),
    ("2.02 (1 tag) Medical emergency",
     "The patient is bleeding heavily from a deep chest wound and losing consciousness.",
     ["urgency"],
     {
         "urgency": {"critical", "high", "emergency", "urgent", "extreme", "yes"},
     }),
    ("2.03 (6 tags) Grateful colleague",
     "Thank you so much for staying late to fix the database server! You saved the launch.",
     ["tone", "gratitude", "urgency", "profanity", "sarcasm", "domain"],
     {
         "tone": {"grateful", "positive", "appreciative", "warm", "friendly"},
         "gratitude": {"high", "yes", "extreme", "strong"},
         "urgency": {"low", "none", "no", "resolved", "high"},
         "profanity": {"no", "none"},
         "sarcasm": {"no", "none"},
         "domain": {"technology", "software", "engineering", "computing", "database", "tech", "it"},
     }),
    ("2.04 (2 tags) Explicit F-word threat",
     "Get out of my way you fucking moron before I smash your phone!",
     ["f word", "aggression"],
     {
         "f word": {"yes"},
         "aggression": {"high", "extreme", "violent", "yes", "severe"},
     }),
    ("2.05 (3 tags) Earnings report",
     "The quarterly report shows revenue grew twelve percent while costs stayed flat.",
     ["sentiment", "topic", "risk"],
     {
         "sentiment": {"positive", "good", "optimistic", "favorable"},
         "topic": {"finance", "revenue", "business", "growth", "earnings", "accounting"},
         "risk": {"low", "none", "no", "minimal"},
     }),
    ("2.06 (5 tags) Polite invoice follow-up",
     "Please send me the invoice again, I think my previous email got lost.",
     ["intent", "politeness", "urgency", "language", "topic"],
     {
         "intent": {"request", "invoice", "billing", "inquiry", "resend"},
         "politeness": {"polite", "high", "yes", "courteous"},
         "urgency": {"low", "medium", "none", "no", "urgent", "moderate"},
         "language": {"english"},
         "topic": {"invoice", "billing", "payment", "finance", "email"},
     }),
    ("2.07 (7 tags) Phishing prize scam",
     "CONGRATULATIONS! Click this link immediately to claim your free million dollar prize!",
     ["spam", "fraud", "urgency", "profanity", "politeness", "topic", "risk"],
     {
         "spam": {"yes", "high", "true"},
         "fraud": {"yes", "high", "scam", "phishing"},
         "urgency": {"high", "urgent", "extreme", "yes"},
         "profanity": {"no", "none"},
         "politeness": {"low", "no", "polite", "yes", "neutral"},
         "topic": {"prize", "lottery", "scam", "money", "fraud", "marketing"},
         "risk": {"high", "extreme", "yes", "severe"},
     }),
    ("2.08 (2 tags) Sincere apology",
     "We deeply regret the delay in shipping your order and have issued a full twenty dollar credit.",
     ["apology", "tone"],
     {
         "apology": {"yes", "sincere", "apologetic", "true"},
         "tone": {"apologetic", "polite", "sincere", "professional", "regretful", "formal"},
     }),
    ("2.09 (4 tags) Sarcastic restaurant review",
     "Oh wonderful, another cold burger and forty minutes waiting in line, truly five star service!",
     ["sarcasm", "complaint", "sentiment", "domain"],
     {
         "sarcasm": {"yes", "high", "true", "sarcastic"},
         "complaint": {"yes", "high", "true", "service", "food"},
         "sentiment": {"negative", "bad", "dissatisfied", "angry", "frustrated", "sarcastic"},
         "domain": {"restaurant", "food", "dining", "hospitality", "service"},
     }),
    ("2.10 (1 tag) Wildlife danger",
     "A venomous king cobra was spotted slithering beneath the kindergarten playground slide.",
     ["danger"],
     {
         "danger": {"high", "extreme", "critical", "severe", "yes", "deadly"},
     }),
    ("2.11 (5 tags) Peaceful nature observation",
     "Golden autumn leaves drifted gently across the quiet pond at sunrise.",
     ["emotion", "aggression", "profanity", "n word", "topic"],
     {
         "emotion": {"calm", "peaceful", "serene", "tranquil", "joy", "relaxed", "positive"},
         "aggression": {"no", "none", "low", "zero"},
         "profanity": {"no", "none"},
         "n word": {"no", "none"},
         "topic": {"nature", "autumn", "scenery", "outdoors", "weather", "season"},
     }),
    ("2.12 (5 tags) Sports injury report",
     "The striker tore his hamstring during the second half of the championship match in Madrid.",
     ["sport", "medical", "urgency", "profanity", "continent"],
     {
         "sport": {"football", "soccer", "sports", "yes"},
         "medical": {"yes", "injury", "hamstring", "high"},
         "urgency": {"high", "moderate", "medium", "urgent", "yes"},
         "profanity": {"no", "none"},
         "continent": {"europe"},
     }),
]

# 3. MODE 3 HELD-OUT SUITE (12 situations, counts: 3, 1, 2, 4, 5, 3, 6, 2, 4, 5, 4, 6 -> 45 checks)
MODE3_SUITE = [
    ("3.01 (3 q) Legal contract (bound prefix 'indemn')",
     "The software vendor shall hold the hospital harmless and reimburse all legal fees arising from third-party intellectual property claims.",
     [
         ("What type of legal protection clause is this?", {"indemnification", "indemnity"}),
         ("Who provides the software?", {"vendor"}),
         ("What type of property is covered?", {"intellectual", "software"}),
     ]),
    ("3.02 (1 q) Clinical cardiology (bound prefix 'cardi')",
     "A 62-year-old man arrived at the emergency room with crushing chest pain radiating to his left jaw.",
     [
         ("Which medical specialty treats this?", {"cardiology"}),
     ]),
    ("3.03 (2 q) Cyber wire fraud (uppercase proper noun + acronym)",
     "Attackers hijacked the executive's phone number via a fraudulent SIM swap and wired funds to an account in Russia.",
     [
         ("Which country received the stolen funds?", {"russia"}),
         ("What mobile component was hijacked?", {"sim", "phone"}),
     ]),
    ("3.04 (4 q) E-commerce defective monitor",
     "The customer received a cracked monitor screen and demanded an immediate replacement instead of a repair voucher.",
     [
         ("What is the physical condition of the monitor?", {"cracked", "damaged", "broken"}),
         ("How does the customer feel?", {"frustrated", "angry", "upset", "dissatisfied"}),
         ("What did the customer demand instead of a repair voucher?", {"replacement"}),
         ("Which support team handles cracked screens?", {"hardware", "support", "technical", "tech", "warranty"}),
     ]),
    ("3.05 (5 q) Aviation emergency",
     "During the night flight over the Atlantic, the cargo jet lost cabin pressure after a cockpit windshield cracked, forcing an emergency landing in Lisbon.",
     [
         ("What type of aircraft was flying?", {"jet", "plane", "aircraft", "cargo"}),
         ("Which ocean was the aircraft crossing?", {"atlantic"}),
         ("What glass component cracked?", {"windshield", "window"}),
         ("In which city did the plane land?", {"lisbon"}),
         ("What was lost inside the cabin?", {"pressure", "cabin"}),
     ]),
    ("3.06 (3 q) World football match",
     "Brazil beat Argentina two to one in a rainy football final at the Maracana stadium.",
     [
         ("Who won the match?", {"brazil"}),
         ("Which sport was played?", {"football", "soccer"}),
         ("What was the weather like?", {"rainy", "rain", "wet"}),
     ]),
    ("3.07 (6 q) Botany & photosynthesis (bound prefix 'photos')",
     "Green plants absorb sunlight through chlorophyll in their leaves to convert carbon dioxide and water into glucose and oxygen.",
     [
         ("What biological process is described?", {"photosynthesis"}),
         ("What green pigment absorbs light?", {"chlorophyll"}),
         ("Which part of the plant contains the pigment?", {"leaves", "leaf"}),
         ("What gas do plants release?", {"oxygen"}),
         ("What sugar is produced?", {"glucose", "sugar"}),
         ("What energy source drives this process?", {"sunlight", "sun", "light", "solar"}),
     ]),
    ("3.08 (2 q) Archaeological dig (bound prefix 'archae')",
     "Scientists excavated ancient pottery and bronze coins from the buried Roman villa near Pompeii.",
     [
         ("What academic field studies ancient human artifacts?", {"archaeology", "history"}),
         ("Which ancient civilization built the villa?", {"roman", "rome", "romans"}),
     ]),
    ("3.09 (4 q) Tokyo bullet train",
     "The silver bullet train departed Tokyo at dawn and arrived in Osaka two hours later carrying four hundred passengers.",
     [
         ("What mode of transport is described?", {"train", "rail", "railway", "bullet"}),
         ("Which city did the journey start from?", {"tokyo"}),
         ("Which city was the destination?", {"osaka"}),
         ("When did the train depart?", {"dawn", "morning", "sunrise"}),
     ]),
    ("3.10 (5 q) Submarine oceanography",
     "The titanium research submarine descended four thousand meters into the Pacific trench to film bioluminescent jellyfish.",
     [
         ("What underwater vessel was used?", {"submarine", "submersible"}),
         ("What metal was the hull made of?", {"titanium"}),
         ("Which ocean was explored?", {"pacific"}),
         ("What marine creature was filmed?", {"jellyfish"}),
         ("What was the purpose of the mission?", {"research", "filming", "exploration", "science"}),
     ]),
    ("3.11 (4 q) Bakery sourdough fire",
     "Smoke filled the kitchen when the apprentice baker left the sourdough loaves in the brick oven for three hours.",
     [
         ("What type of bread was baking?", {"sourdough", "bread", "loaves"}),
         ("Who forgot the bread in the oven?", {"apprentice", "baker"}),
         ("What material was the oven made from?", {"brick", "bricks"}),
         ("What filled the kitchen?", {"smoke"}),
     ]),
    ("3.12 (6 q) Mount Everest expedition",
     "Sherpa guides led two Norwegian climbers to the summit of Mount Everest in May despite freezing winds and low oxygen.",
     [
         ("Which mountain was climbed?", {"everest"}),
         ("What nationality were the climbers?", {"norwegian", "norway"}),
         ("In which month did they reach the top?", {"may"}),
         ("Who guided the climbers?", {"sherpa", "sherpas", "guides"}),
         ("What gas was scarce at high altitude?", {"oxygen"}),
         ("What part of the mountain did they reach?", {"summit", "top", "peak"}),
     ]),
]


def run_benchmark(model_name="Qwen/Qwen2.5-1.5B", reuse_loaded=True, verbose_failures=True):
    if reuse_loaded and full.active_model is not None:
        print(f"\nUsing currently loaded model: {model_name}")
    else:
        if full.active_model is not None:
            del full.active_model
            del full.active_tokenizer
            full.active_model = None
            full.active_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        full.init_engine(model_name=model_name)

    P = full.predict_unified_tree
    stats = {
        "m1": {"ok": 0, "tot": 0, "passes": 0, "ms": 0.0},
        "m2": {"ok": 0, "tot": 0, "passes": 0, "ms": 0.0},
        "m3": {"ok": 0, "tot": 0, "passes": 0, "ms": 0.0},
    }
    failures = []

    # --- MODE 1 ---
    for idx, (state, exp) in enumerate(MODE1_SUITE, 1):
        res = P(state)
        r = res["results"]["headline"]
        w = r["word"]
        ok = full._check_match(w, accept=exp)
        stats["m1"]["ok"] += int(ok)
        stats["m1"]["tot"] += 1
        stats["m1"]["passes"] += res["metrics"]["forward_passes"]
        stats["m1"]["ms"] += res["metrics"]["latency_ms"]
        if not ok:
            failures.append(("M1", f"1.{idx:02d}", state[:48], w, sorted(exp), r.get("top_k", {})))

    # --- MODE 2 ---
    for title, state, tag_list, exp_map in MODE2_SUITE:
        res = P(state, tags=tag_list)
        stats["m2"]["passes"] += res["metrics"]["forward_passes"]
        stats["m2"]["ms"] += res["metrics"]["latency_ms"]
        for t in tag_list:
            r = res["results"][t]
            w = r["word"]
            exp = exp_map[t]
            ok = full._check_match(w, accept=exp)
            stats["m2"]["ok"] += int(ok)
            stats["m2"]["tot"] += 1
            if not ok:
                failures.append(("M2", title, f"Tag [{t}]", w, sorted(exp), r.get("top_k", {})))

    # --- MODE 3 ---
    for title, state, q_specs in MODE3_SUITE:
        q_list = [q for q, _ in q_specs]
        res = P(state, questions=q_list)
        stats["m3"]["passes"] += res["metrics"]["forward_passes"]
        stats["m3"]["ms"] += res["metrics"]["latency_ms"]
        for q, exp in q_specs:
            r = res["results"][q]
            w = r["word"]
            ok = full._check_match(w, accept=exp)
            stats["m3"]["ok"] += int(ok)
            stats["m3"]["tot"] += 1
            if not ok:
                failures.append(("M3", title, q[:48], w, sorted(exp), r.get("top_k", {})))

    tot_ok = sum(s["ok"] for s in stats.values())
    tot_n = sum(s["tot"] for s in stats.values())
    tot_passes = sum(s["passes"] for s in stats.values())
    tot_ms = sum(s["ms"] for s in stats.values())

    print("\n" + "=" * 100)
    print(f"HELD-OUT v6 BENCHMARK REPORT — {model_name}")
    print("=" * 100)
    print(f"  Mode 1 (State-Only, 15 situations) : {stats['m1']['ok']:2d}/{stats['m1']['tot']:2d} ({stats['m1']['ok']/stats['m1']['tot']*100:5.1f}%) | {stats['m1']['passes']:2d} passes | {stats['m1']['ms']:7.1f} ms")
    print(f"  Mode 2 (Tags, 1..7 per situation)  : {stats['m2']['ok']:2d}/{stats['m2']['tot']:2d} ({stats['m2']['ok']/stats['m2']['tot']*100:5.1f}%) | {stats['m2']['passes']:2d} passes | {stats['m2']['ms']:7.1f} ms")
    print(f"  Mode 3 (Questions, 1..6, no opts)  : {stats['m3']['ok']:2d}/{stats['m3']['tot']:2d} ({stats['m3']['ok']/stats['m3']['tot']*100:5.1f}%) | {stats['m3']['passes']:2d} passes | {stats['m3']['ms']:7.1f} ms")
    print("-" * 100)
    print(f"  OVERALL SCORE                      : {tot_ok:3d}/{tot_n:3d} ({tot_ok/tot_n*100:5.1f}%) | {tot_passes:2d} passes across 39 trees | {tot_ms:7.1f} ms")
    print("=" * 100)

    if verbose_failures and failures:
        print(f"\nDIAGNOSTIC FAILURE LOG ({len(failures)} misses out of {tot_n}):")
        for mode, case_id, item, pred, exp, topk in failures:
            print(f"  [{mode}] {case_id:<32} | {item:<48}")
            print(f"       pred={pred!r:<14} | accepted={exp}")
            print(f"       top_5={topk}")
    return {"model": model_name, "stats": stats, "failures": failures, "tot_ok": tot_ok, "tot_n": tot_n}


if __name__ == "__main__":
    run_benchmark("Qwen/Qwen2.5-1.5B", reuse_loaded=False)
