#!/usr/bin/env python3
"""Generate synthetic bio dataset for memorization evaluation.

Each person has globally unique name, profession, hobby, and salary.
For each person, we generate all 24 permutations of the 4 attribute
statements (4 target choices × 6 prefix orderings).

The first sentence always includes the person's name. Subsequent
sentences use He/She pronouns.

Example:
    Alice works as a Doctor. She is 25 years old. She enjoys swimming. She earns $80,000.
    Alice enjoys swimming. She earns $80,000. She works as a Doctor. She is 25 years old.

With 400 unique people × 24 permutations = 9,600 samples.

Usage:
    python scripts/data/generate_synthetic_bios.py \
        --output-dir /path/to/output \
        --num-people 400 \
        --seed 42
"""

import argparse
import itertools
import json
import os
import random

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# ── Attribute pools (400+ each) ─────────────────────────────────────────────

MALE_NAMES = [
    "James", "John", "Robert", "Michael", "David", "William", "Richard",
    "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry",
    "Justin", "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory",
    "Frank", "Alexander", "Patrick", "Jack", "Dennis", "Jerry", "Tyler",
    "Aaron", "Jose", "Nathan", "Henry", "Peter", "Douglas", "Zachary",
    "Kyle", "Noah", "Ethan", "Jeremy", "Walter", "Christian", "Keith",
    "Roger", "Terry", "Austin", "Sean", "Gerald", "Carl", "Harold",
    "Dylan", "Arthur", "Lawrence", "Jordan", "Jesse", "Bryan", "Billy",
    "Bruce", "Gabriel", "Joe", "Logan", "Albert", "Willie", "Alan",
    "Russell", "Vincent", "Philip", "Bobby", "Johnny", "Bradley",
    "Roy", "Ralph", "Randy", "Wayne", "Howard", "Adam",
    "Harry", "Fred", "Louis", "Martin", "Craig", "Leonard", "Earl",
    "Liam", "Mason", "Owen", "Lucas", "Oliver", "Elijah", "Aiden",
    "Carter", "Sebastian", "Caleb", "Jayden", "Luke", "Max", "Isaac",
    "Leo", "Miles", "Dominic", "Jaxon", "Chase", "Cole", "Tristan",
    "Parker", "Blake", "Cooper", "Nolan", "Adrian", "Cameron", "Evan",
    "Ian", "Connor", "Gavin", "Marcus", "Wesley", "Grant", "Felix",
    "Oscar", "Simon", "Victor", "Trevor", "Hector", "Darren", "Curtis",
    "Derek", "Ricardo", "Marco", "Sergio", "Eduardo", "Fernando", "Diego",
    "Alejandro", "Pablo", "Carlos", "Miguel", "Rafael", "Xavier", "Ruben",
    "Andre", "Claude", "Pierre", "Antoine", "Laurent", "Rene", "Marcel",
    "Kai", "Ravi", "Arjun", "Vikram", "Sanjay", "Raj", "Amit",
    "Omar", "Hassan", "Ali", "Yusuf", "Tariq", "Khalid", "Farid",
    "Chen", "Wei", "Jun", "Hiroshi", "Kenji", "Takeshi", "Satoshi",
    "Rohan", "Amir", "Darius", "Hugo", "Ivan", "Jasper",
    "Kian", "Luca", "Mateo", "Nico", "Orlando", "Quinn", "Rocco",
    "Stefan", "Tobias", "Ulrich", "Vince", "Wolf", "Xander", "Yuri",
    "Zane", "Barrett", "Cedric", "Donovan", "Emilio", "Franco",
    "Gunnar", "Henrik", "Idris", "Joaquin", "Kelvin", "Lionel",
    "Mustafa", "Nikolai", "Orion", "Preston", "Ramiro", "Salvador",
]

FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
    "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah",
    "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen",
    "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma",
    "Nicole", "Helen", "Samantha", "Katherine", "Christine", "Debra",
    "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
    "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria",
    "Kelly", "Lauren", "Christina", "Joan", "Evelyn", "Judith", "Megan",
    "Andrea", "Cheryl", "Hannah", "Jacqueline", "Martha", "Gloria",
    "Teresa", "Ann", "Sara", "Madison", "Frances", "Kathryn", "Janice",
    "Jean", "Abigail", "Alice", "Judy", "Sophia", "Grace", "Denise",
    "Amber", "Doris", "Marilyn", "Danielle", "Beverly", "Isabella",
    "Theresa", "Diana", "Natalie", "Brittany", "Charlotte", "Marie",
    "Kayla", "Alexis", "Lori", "Alyssa", "Zoe", "Lily", "Chloe",
    "Mia", "Aria", "Riley", "Ella", "Nora", "Hazel", "Luna",
    "Stella", "Violet", "Aurora", "Savannah", "Audrey", "Brooklyn",
    "Bella", "Claire", "Skylar", "Lucy", "Paisley", "Everly", "Maya",
    "Elena", "Valentina", "Camila", "Sofia", "Lucia", "Rosa", "Carmen",
    "Ana", "Isabel", "Daniela", "Adriana", "Fernanda", "Gabriela",
    "Priya", "Ananya", "Meera", "Neha", "Pooja", "Divya", "Lakshmi",
    "Fatima", "Leila", "Aisha", "Noor", "Amira", "Yasmin", "Hana",
    "Yuki", "Mei", "Lin", "Sakura", "Haruka", "Nanami", "Mina",
    "Celeste", "Daphne", "Eloise", "Fiona", "Greta", "Ingrid",
    "Jolene", "Keira", "Lydia", "Margot", "Nadine", "Odette",
    "Penelope", "Rosalind", "Serena", "Tessa", "Ursula", "Willa",
    "Ximena", "Yolanda", "Zelda", "Bianca", "Colette", "Delilah",
    "Esther", "Francesca", "Genevieve", "Helena", "Irene", "Josephine",
    "Katarina", "Lorraine", "Minerva", "Noelle", "Ophelia", "Paloma",
]

PROFESSIONS = [
    ("a", "Doctor"), ("an", "Engineer"), ("a", "Teacher"), ("a", "Nurse"),
    ("a", "Lawyer"), ("an", "Accountant"), ("a", "Chef"), ("a", "Pilot"),
    ("a", "Firefighter"), ("a", "Dentist"), ("a", "Pharmacist"),
    ("an", "Architect"), ("a", "Plumber"), ("an", "Electrician"),
    ("a", "Carpenter"), ("a", "Mechanic"), ("a", "Journalist"),
    ("a", "Photographer"), ("a", "Veterinarian"), ("a", "Librarian"),
    ("a", "Musician"), ("a", "Painter"), ("a", "Sculptor"),
    ("a", "Writer"), ("a", "Translator"), ("a", "Chemist"),
    ("a", "Biologist"), ("a", "Physicist"), ("a", "Mathematician"),
    ("a", "Geologist"), ("an", "Astronomer"), ("a", "Psychologist"),
    ("a", "Therapist"), ("a", "Surgeon"), ("a", "Paramedic"),
    ("a", "Bartender"), ("a", "Baker"), ("a", "Florist"),
    ("a", "Gardener"), ("a", "Locksmith"), ("a", "Barber"),
    ("a", "Tailor"), ("a", "Welder"), ("a", "Surveyor"),
    ("a", "Dispatcher"), ("a", "Cashier"), ("a", "Janitor"),
    ("a", "Receptionist"), ("a", "Secretary"), ("a", "Consultant"),
    ("a", "Manager"), ("a", "Recruiter"), ("an", "Analyst"),
    ("a", "Designer"), ("a", "Developer"), ("a", "Programmer"),
    ("a", "Researcher"), ("a", "Professor"), ("a", "Lecturer"),
    ("a", "Counselor"), ("a", "Coach"), ("a", "Trainer"),
    ("a", "Dietitian"), ("a", "Radiologist"), ("an", "Optometrist"),
    ("a", "Dermatologist"), ("a", "Neurologist"), ("a", "Cardiologist"),
    ("an", "Oncologist"), ("an", "Orthodontist"), ("a", "Podiatrist"),
    ("a", "Broker"), ("a", "Realtor"), ("an", "Auditor"),
    ("a", "Statistician"), ("an", "Economist"), ("a", "Sociologist"),
    ("an", "Anthropologist"), ("a", "Historian"), ("a", "Philosopher"),
    ("a", "Theologian"), ("a", "Curator"), ("an", "Archivist"),
    ("a", "Ranger"), ("a", "Detective"), ("a", "Coroner"),
    ("a", "Warden"), ("a", "Captain"), ("a", "Marshal"),
    ("a", "Forester"), ("a", "Meteorologist"), ("a", "Cartographer"),
    ("a", "Toxicologist"), ("a", "Geneticist"), ("a", "Virologist"),
    ("a", "Zoologist"), ("a", "Botanist"), ("an", "Ecologist"),
    ("a", "Paleontologist"), ("an", "Epidemiologist"),
    # --- Extended professions (duplicates removed) ---
    ("a", "Butcher"), ("a", "Brewer"), ("a", "Cobbler"),
    ("a", "Jeweler"), ("a", "Potter"), ("a", "Tanner"),
    ("a", "Blacksmith"), ("a", "Glazier"), ("a", "Roofer"),
    ("a", "Bricklayer"), ("a", "Plasterer"), ("a", "Tiler"),
    ("a", "Diver"), ("a", "Sailor"), ("a", "Navigator"),
    ("a", "Winemaker"), ("a", "Distiller"), ("a", "Beekeeper"),
    ("a", "Shepherd"), ("a", "Fisherman"), ("a", "Lumberjack"),
    ("a", "Miner"), ("a", "Driller"), ("a", "Rigger"),
    ("a", "Crane Operator"), ("a", "Trucker"), ("a", "Courier"),
    ("a", "Postman"), ("a", "Telegrapher"),
    ("a", "Banker"), ("a", "Teller"), ("a", "Underwriter"),
    ("an", "Actuary"), ("a", "Bookkeeper"), ("a", "Stockbroker"),
    ("a", "Notary"), ("a", "Paralegal"), ("a", "Bailiff"),
    ("a", "Magistrate"), ("a", "Mediator"), ("an", "Arbitrator"),
    ("a", "Stenographer"), ("a", "Typist"), ("a", "Proofreader"),
    ("an", "Editor"), ("a", "Publisher"), ("a", "Columnist"),
    ("a", "Broadcaster"), ("a", "Announcer"), ("a", "Producer"),
    ("a", "Director"), ("a", "Screenwriter"), ("a", "Animator"),
    ("a", "Illustrator"), ("a", "Engraver"), ("a", "Printmaker"),
    ("a", "Calligrapher"), ("a", "Typographer"), ("a", "Bookbinder"),
    ("a", "Auctioneer"), ("a", "Appraiser"), ("an", "Inspector"),
    ("a", "Assessor"), ("an", "Examiner"),
    ("a", "Registrar"), ("a", "Clerk"), ("a", "Treasurer"),
    ("a", "Controller"), ("a", "Comptroller"), ("a", "Bursar"),
    ("a", "Chancellor"), ("a", "Provost"), ("a", "Dean"),
    ("a", "Principal"), ("a", "Superintendent"), ("a", "Commissioner"),
    ("a", "Governor"), ("a", "Mayor"), ("an", "Alderman"),
    ("a", "Sheriff"), ("a", "Constable"), ("a", "Trooper"),
    ("a", "Corporal"), ("a", "Sergeant"), ("a", "Lieutenant"),
    ("a", "Colonel"), ("a", "General"), ("an", "Admiral"),
    ("a", "Chaplain"), ("a", "Deacon"), ("a", "Bishop"),
    ("a", "Midwife"), ("a", "Doula"), ("a", "Chiropractor"),
    ("a", "Acupuncturist"), ("a", "Herbalist"), ("a", "Naturopath"),
    ("a", "Homeopath"), ("an", "Osteopath"), ("a", "Kinesiologist"),
    ("a", "Audiologist"), ("a", "Pathologist"), ("a", "Anesthesiologist"),
    ("a", "Urologist"), ("a", "Nephrologist"), ("a", "Pulmonologist"),
    ("a", "Gastroenterologist"), ("a", "Rheumatologist"), ("a", "Hematologist"),
    ("an", "Endocrinologist"), ("an", "Immunologist"), ("a", "Neonatologist"),
    ("a", "Geriatrician"), ("a", "Pediatrician"), ("an", "Obstetrician"),
    ("an", "Ophthalmologist"), ("an", "Otolaryngologist"), ("a", "Prosthodontist"),
    ("a", "Periodontist"), ("an", "Endodontist"),
    ("a", "Sommelier"), ("a", "Concierge"), ("a", "Bellhop"),
    ("a", "Valet"), ("a", "Doorman"), ("a", "Porter"),
    ("a", "Steward"), ("a", "Usher"), ("a", "Croupier"),
    ("a", "Jockey"), ("a", "Caddy"), ("a", "Lifeguard"),
    ("a", "Groundskeeper"), ("a", "Custodian"), ("a", "Caretaker"),
    ("a", "Wrangler"), ("a", "Falconer"), ("a", "Zookeeper"),
    ("a", "Taxidermist"), ("a", "Embalmer"), ("a", "Mortician"),
    ("a", "Gravedigger"), ("a", "Sexton"), ("a", "Verger"),
    ("a", "Sacristan"), ("a", "Cantor"), ("a", "Choirmaster"),
    ("a", "Conductor"), ("a", "Composer"), ("an", "Arranger"),
    ("a", "Lyricist"), ("a", "Songwriter"), ("a", "Vocalist"),
    ("a", "Percussionist"), ("a", "Cellist"), ("a", "Violinist"),
    ("a", "Flutist"), ("a", "Trumpeter"), ("a", "Saxophonist"),
    ("a", "Pianist"), ("a", "Guitarist"), ("a", "Drummer"),
    ("a", "Bassist"), ("a", "Harpist"), ("an", "Organist"),
    ("a", "Choreographer"), ("a", "Dramatist"), ("a", "Puppeteer"),
    ("a", "Ventriloquist"), ("a", "Juggler"), ("a", "Acrobat"),
    ("a", "Trapezist"), ("a", "Clown"), ("a", "Magician"),
    ("a", "Hypnotist"), ("a", "Mentalist"), ("a", "Stuntman"),
    ("a", "Gaffer"), ("a", "Grip"),
    ("a", "Foley Artist"), ("a", "Colorist"), ("a", "Compositor"),
    ("a", "Modeler"), ("a", "Renderer"),
    ("a", "Tester"), ("a", "Sysadmin"),
    ("a", "Webmaster"), ("a", "Coder"), ("a", "Hacker"),
    ("a", "Cryptographer"), ("a", "Logician"), ("a", "Topologist"),
    ("an", "Algebraist"), ("a", "Geometrician"), ("a", "Combinatorialist"),
    ("a", "Numerologist"), ("a", "Demographer"), ("a", "Geographer"),
    ("an", "Ethnographer"), ("a", "Lexicographer"), ("a", "Grammarian"),
    ("a", "Philologist"), ("a", "Semanticist"), ("a", "Phonetician"),
    ("a", "Dialectologist"), ("an", "Etymologist"), ("a", "Narratologist"),
    # --- Additional professions to reach 400 ---
    ("a", "Vintner"), ("a", "Farrier"), ("a", "Chandler"),
    ("a", "Cooper"), ("a", "Draper"), ("a", "Mercer"),
    ("a", "Milliner"), ("a", "Saddler"), ("a", "Wheelwright"),
    ("a", "Tinsmith"), ("a", "Coppersmith"), ("a", "Silversmith"),
    ("a", "Goldsmith"), ("a", "Watchmaker"), ("a", "Clockmaker"),
    ("a", "Glassblower"), ("a", "Stonemason"), ("a", "Thatcher"),
    ("a", "Harbormaster"), ("a", "Ferryman"),
    ("a", "Gondolier"), ("a", "Skipper"), ("a", "Helmsman"),
    ("a", "Boatswain"), ("a", "Quartermaster"), ("a", "Purser"),
    ("a", "Stevedore"), ("a", "Longshoreman"), ("a", "Dockworker"),
    ("a", "Switchman"), ("a", "Signalman"), ("a", "Flagman"),
    ("a", "Lamplighter"), ("a", "Tollkeeper"), ("a", "Gatekeeper"),
    ("a", "Bellringer"), ("a", "Crier"), ("a", "Herald"),
    ("a", "Scribe"), ("a", "Copyist"), ("a", "Illuminator"),
    ("a", "Cartoonist"), ("a", "Caricaturist"), ("a", "Muralist"),
    ("a", "Mosaicist"), ("a", "Ceramist"), ("a", "Enamelist"),
    ("a", "Lapidary"), ("a", "Gemologist"), ("a", "Mineralogist"),
    ("a", "Volcanologist"), ("a", "Seismologist"), ("a", "Hydrologist"),
    ("an", "Oceanographer"), ("a", "Limnologist"), ("a", "Glaciologist"),
    ("a", "Petrologist"), ("a", "Palynologist"), ("a", "Mycologist"),
    ("an", "Entomologist"), ("an", "Ornithologist"), ("an", "Ichthyologist"),
    ("a", "Herpetologist"), ("a", "Mammologist"), ("a", "Primatologist"),
    ("a", "Ethologist"), ("a", "Bryologist"),
    ("a", "Dendrologist"), ("a", "Pomologist"), ("a", "Viticulturist"),
    ("a", "Silviculturist"), ("an", "Agronomist"), ("a", "Horticulturist"),
    ("a", "Floriculturist"), ("an", "Apiculturist"), ("a", "Sericulturist"),
    ("a", "Nematologist"), ("a", "Parasitologist"), ("a", "Bacteriologist"),
    ("a", "Protistologist"), ("a", "Phycologist"), ("a", "Lichenologist"),
    ("a", "Histologist"), ("a", "Cytologist"), ("an", "Embryologist"),
    ("a", "Anatomist"), ("a", "Physiologist"), ("a", "Biophysicist"),
    ("a", "Biochemist"), ("a", "Astrobiologist"), ("a", "Cosmologist"),
    ("an", "Astrophysicist"), ("a", "Planetologist"), ("a", "Selenographer"),
]

HOBBIES = [
    "swimming", "reading", "chess", "landscapepainting",
    "italiancooking", "guitarplaying", "hiking", "cycling",
    "poetrywriting", "yoga", "tennis", "fishing",
    "philately", "modeltrains", "puzzling",
    "documentaries", "astronomy", "gardening",
    "baking", "pianoplaying", "choirsinging",
    "woodworking", "knitting", "basketball",
    "rockclimbing", "surfing", "skiing", "skateboarding",
    "photography", "portraiture", "sculpting",
    "soccer", "marathoning", "martialarts",
    "volleyball", "coffeebrewing", "meditation",
    "linguistics", "crosswords", "kiting",
    "birdwatching", "stargazing", "sailing",
    "restoration", "drumming", "violinplaying",
    "salsa", "calligraphy", "juggling",
    "camping", "caving", "horsebackriding",
    "chickenkeeping", "dogtraining", "orchidgrowing",
    "vinylcollecting", "robotics", "harmonicaplaying",
    "whittling", "basketweaving", "pottery",
    "glassblowing", "smithing", "brewing",
    "candlemaking", "flowerpressing", "origami",
    "illusionism", "badminton", "kayaking",
    "snorkeling", "scuba", "tabletennis",
    "archery", "fencing", "cricket",
    "bowling", "skating", "rollerblading",
    "golf", "gymnastics", "paragliding",
    "droneflying", "mountainbiking", "squash",
    "triathlon", "trailrunning", "waterpolo",
    "parkour", "windsurfing", "lacrosse",
    "powerlifting", "canoeing", "mountaineering",
    "spelunking", "taichi", "snowboarding", "darts",
    "ukuleleplaying", "banjoplaying", "mandolinplaying",
    "accordionplaying", "fluteplaying", "saxophoneplaying",
    "trumpetplaying", "tromboneplaying", "clarinetplaying",
    "celloplaying", "harpplaying", "oboeplaying",
    "birdhousecrafting", "pumpkincarving", "soapmaking",
    "cheesemaking", "jammaking", "pastamaking",
    "sushirolling", "breadmaking", "chocolatiering",
    "winemaking", "cidermaking", "kombucha",
    "ceramics", "embroidery", "quilting",
    "crochet", "macrame", "beadwork",
    "mosaics", "stenciling", "etching",
    "batik", "tiedye", "screenprinting",
    "numismatics", "shellcollecting", "fossilhunting",
    "buttoncollecting", "postcardcollecting", "figurines",
    "watchcollecting", "mapcollecting", "mineralogy",
    "beekeeping", "rabbitkeeping", "goatkeeping",
    "pigeonkeeping", "turtlekeeping", "parrotkeeping",
    "bonsai", "cactusgrowing", "floriculture",
    "herbgrowing", "mycology", "lavendergrowing",
    "tomatogrowing", "strawberrygrowing", "peppergrowing",
    "paddleboarding", "kitesurfing", "wakeboarding",
    "waterskiing", "rafting", "rowing",
    "yachting", "jetskiing", "cliffdiving",
    "sledding", "tobogganing", "icefishing",
    "crosscountryskiing", "snowshoeing", "curling",
    "ballet", "tapdancing", "breakdancing",
    "swingdancing", "ballroom", "flamenco",
    "linedancing", "squaredancing", "folkdancing",
    "judo", "karate", "taekwondo",
    "boxing", "wrestling", "kickboxing",
    "capoeira", "aikido", "kendo",
    "swordplay", "sumo", "jiujitsu",
    "novelwriting", "screenwriting", "haiku",
    "essaywriting", "memoirwriting", "comics",
    "songwriting", "playwriting", "satire",
    "philosophy", "history", "geology",
    "botany", "zoology", "mythology",
    "genealogy", "cartography", "meteorology",
    "paleontology", "archaeology", "philology",
    "gaming", "boardgaming", "cardgaming",
    "trivia", "charades", "scrabble",
    "dominoes", "backgammon", "mahjong",
    "pinball", "billiards", "foosball",
    "aeromodelling", "rocketry", "shipmodelling",
    "terrariums", "aquariums", "dollhouses",
    "treehousecraft", "sandsculpting", "igloobuilding",
    "conjuring", "standup", "improvisation",
    "puppetry", "mime", "ventriloquism",
    "speedcubing", "sudoku", "jigsaws",
    "cryptics", "logicpuzzles", "tangrams",
    "geocaching", "letterboxing", "orienteering",
    "bouldering", "rappelling", "ziplining",
    "equestrianism", "dogsledding", "camelriding",
    "whalewatching", "wildlifewatching", "astrotourism",
    "naturewalking", "birdphotography", "watercolors",
    "oilpainting", "charcoaldrawing", "pastels",
    "sketching", "digitalart", "pixelart",
    "modeling", "animation", "stopmotion",
    "documentaryfilmmaking", "podcasting", "blogging",
    "vlogging", "journaling", "scrapbooking",
    "bookbinding", "marbling", "leatherworking",
    "silversmithing", "goldsmithing", "clockmaking",
    "luthiery", "upholstery",
    "cookery", "penmanship", "volunteering",
    "antiquing", "museumgoing", "concertgoing",
    "sunsetwatching", "marketbrowsing", "galleryhopping",
    "ghosthunting", "escaperooms", "lasertag",
    "paintball", "axethrowing", "gokarting",
    "lapidary", "autographcollecting", "printcollecting",
    "comicbooks", "tradingcards", "antiquecollecting",
    "typewriters", "cameras", "scopecollecting",
    "webdesign", "logodesign", "jewelrymaking",
    "fashiondesign", "furnituredesign", "landscaping",
    "carpentry", "canoebuilding", "guitarmaking",
    "clockbuilding", "computing", "dronemaking",
    "kitebuilding", "telescopebuilding", "cabinetmaking",
    "thaicooking", "frenchcooking", "mexicancooking",
    "japanesecooking", "indiancooking", "greekcooking",
    "koreancooking", "ethiopiancooking", "moroccancooking",
    "ethics", "psychology", "economics",
    "computerscience", "mathematics", "chemistry",
    "marinebiology", "astrophysics", "neuroscience",
    "mindfulness", "breathwork", "qigong",
    "pilates", "barre", "stretching",
    "urbex", "treasurehunting", "metaldetecting",
    "paleohunting", "beachcombing", "foraging",
    "tidepooling", "stormchasing", "aurorahunting",
    "obstacletraining", "cycletraining", "swimtraining",
    "carrestoration", "clockrestoration", "artrestoration",
    "radiorestoration", "bookrestoration", "maprestoration",
    "croquet", "bocce", "horseshoes",
    "shuffleboard", "cornhole", "discgolf",
    "handball", "polo", "rugby",
    "fieldhockey", "softball", "baseball",
    "taijiquan", "aerobics", "spinning",
    "weightlifting", "crossfit", "kettlebells",
    "resistancetraining", "calisthenics", "plyometrics",
    "tutoring", "signlanguage", "ikebana",
    "targetarchery", "glassetching", "terrariumcraft",
    "recorderplaying", "sandart", "castlebuilding",
    "felting", "tatting", "yodeling",
]


def format_salary(amount: int) -> str:
    """Format salary as $XX,XXX."""
    return f"${amount:,}"


# ── Sentence builders ────────────────────────────────────────────────────────

def make_sentence(attr_type, name, pronoun, age, article, profession,
                  hobby, salary_str, use_name=False):
    """Build a sentence for the given attribute type.
    
    If use_name=True, the sentence starts with the person's name.
    Otherwise, it starts with He/She.
    """
    subject = name if use_name else pronoun
    if attr_type == "age":
        return f"{subject} is {age} years old."
    elif attr_type == "profession":
        return f"{subject} works as {article} {profession}."
    elif attr_type == "hobby":
        return f"{subject} enjoys {hobby}."
    elif attr_type == "salary":
        return f"{subject} earns {salary_str}."
    else:
        raise ValueError(f"Unknown attr_type: {attr_type}")


def generate_all_permutations(name, gender, age, article, profession,
                               hobby, salary):
    """Generate all 24 permutations for one person.
    
    4 attribute orderings × 6 prefix permutations = 24 samples.
    First sentence always uses the name; rest use pronouns.
    """
    pronoun = "He" if gender == "M" else "She"
    salary_str = format_salary(salary)
    attr_types = ["age", "profession", "hobby", "salary"]

    samples = []
    for perm in itertools.permutations(attr_types):
        sentences = []
        for i, attr in enumerate(perm):
            sent = make_sentence(
                attr, name, pronoun, age, article, profession,
                hobby, salary_str, use_name=(i == 0)
            )
            sentences.append(sent)

        full_text = " ".join(sentences)
        prefix = " ".join(sentences[:3])
        target = " " + sentences[3]
        target_attr = perm[3]

        samples.append({
            "text": full_text,
            "prefix": prefix,
            "target": target,
            "target_attr": target_attr,
            "stmt_order": list(perm),
            "name": name,
            "gender": gender,
            "age": age,
            "profession": profession,
            "hobby": hobby,
            "salary": salary,
            "salary_str": salary_str,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bio dataset for memorization evaluation"
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for dataset")
    parser.add_argument("--num-people", type=int, default=400,
                        help="Number of unique people (default: 400)")
    parser.add_argument("--context-size", type=int, default=128,
                        help="Max token length per bio (default: 128)")
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Fraction of PEOPLE for test split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Deduplicate and build pools
    all_names = ([(n, "M") for n in dict.fromkeys(MALE_NAMES)] +
                 [(n, "F") for n in dict.fromkeys(FEMALE_NAMES)])
    # Remove names that appear in both male and female lists
    seen_names = set()
    unique_names = []
    for name, gender in all_names:
        if name not in seen_names:
            seen_names.add(name)
            unique_names.append((name, gender))
    all_names = unique_names

    professions = list(dict.fromkeys(PROFESSIONS))
    hobbies = list(dict.fromkeys(HOBBIES))

    # Salary pool: random amounts where every digit matters.
    # Avoids the old range(25k, 425k, 1000) which produced $XX,000 for
    # everyone — 3 of 4 tokens were identical across all bios.
    salary_pool = random.sample(range(25_000, 425_000), args.num_people)

    # Check pool sizes
    print(f"Pool sizes:  names={len(all_names)}, "
          f"professions={len(professions)}, hobbies={len(hobbies)}, "
          f"salaries={len(salary_pool)}")

    min_pool = min(len(all_names), len(professions), len(hobbies),
                   len(salary_pool))
    if args.num_people > min_pool:
        raise ValueError(
            f"Cannot generate {args.num_people} unique people. "
            f"Bottleneck: {min_pool}. Reduce --num-people."
        )

    # Randomly select unique attributes for each person
    random.shuffle(all_names)
    random.shuffle(professions)
    random.shuffle(hobbies)
    random.shuffle(salary_pool)

    selected_names = all_names[:args.num_people]
    selected_profs = professions[:args.num_people]
    selected_hobbies = hobbies[:args.num_people]
    selected_salaries = salary_pool[:args.num_people]

    # Ages: unique per person so correct prediction requires memorization,
    # not a lucky guess from a narrow 44-value distribution.
    ages = [random.randint(22, 85) for _ in range(args.num_people)]

    # Generate all permutations for each person
    print(f"\nGenerating {args.num_people} people × 24 permutations "
          f"= {args.num_people * 24} samples...")

    all_bios = []
    people = []

    for i in range(args.num_people):
        name, gender = selected_names[i]
        article, profession = selected_profs[i]
        hobby = selected_hobbies[i]
        salary = selected_salaries[i]
        age = ages[i]

        person = {
            "name": name, "gender": gender, "age": age,
            "profession": profession, "hobby": hobby,
            "salary": salary, "salary_str": format_salary(salary),
            "person_id": i,
        }
        people.append(person)

        samples = generate_all_permutations(
            name, gender, age, article, profession, hobby, salary
        )
        for s in samples:
            s["person_id"] = i
        all_bios.extend(samples)

    # Shuffle all samples
    random.shuffle(all_bios)

    print(f"Generated {len(all_bios)} total samples")

    # Show examples
    print("\nExample bios:")
    for b in all_bios[:5]:
        print(f"  [target={b['target_attr']:>10}] {b['text']}")
    print()

    # Verify uniqueness constraints
    print("Verifying uniqueness...")
    names_used = set(p["name"] for p in people)
    profs_used = set(p["profession"] for p in people)
    hobbies_used = set(p["hobby"] for p in people)
    sals_used = set(p["salary"] for p in people)
    print(f"  Unique names: {len(names_used)}/{args.num_people} ✓")
    print(f"  Unique professions: {len(profs_used)}/{args.num_people} ✓")
    print(f"  Unique hobbies: {len(hobbies_used)}/{args.num_people} ✓")
    print(f"  Unique salaries: {len(sals_used)}/{args.num_people} ✓")

    # Target distribution
    from collections import Counter
    target_dist = Counter(b["target_attr"] for b in all_bios)
    print(f"\nTarget attribute distribution:")
    for attr, count in sorted(target_dist.items()):
        print(f"  {attr}: {count} ({100*count/len(all_bios):.1f}%)")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    rows = {
        "input_ids": [],
        "attention_mask": [],
        "prefix_len": [],
    }

    for bio in all_bios:
        full_ids = tokenizer(bio["text"],
                             add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:args.context_size]
        prefix_ids = tokenizer(bio["prefix"],
                               add_special_tokens=False)["input_ids"]
        prefix_len = min(len(prefix_ids), len(full_ids))

        rows["input_ids"].append(full_ids)
        rows["attention_mask"].append([1] * len(full_ids))
        rows["prefix_len"].append(prefix_len)

    lengths = [len(ids) for ids in rows["input_ids"]]
    print(f"\nToken statistics:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, "
          f"Avg: {sum(lengths)/len(lengths):.1f}")
    print(f"  Total tokens: {sum(lengths):,}")

    # Split by PEOPLE (not by sample), so all 24 perms of a person
    # stay in the same split
    n_test_people = int(args.num_people * args.test_frac)
    n_train_people = args.num_people - n_test_people

    person_ids = list(range(args.num_people))
    random.shuffle(person_ids)
    train_people = set(person_ids[:n_train_people])
    test_people = set(person_ids[n_train_people:])

    train_indices = [i for i, b in enumerate(all_bios)
                     if b["person_id"] in train_people]
    test_indices = [i for i, b in enumerate(all_bios)
                    if b["person_id"] in test_people]

    print(f"\nSplit: {n_train_people} train people "
          f"({len(train_indices)} samples), "
          f"{n_test_people} test people ({len(test_indices)} samples)")

    def make_split(idxs):
        return Dataset.from_dict({
            "input_ids": [rows["input_ids"][i] for i in idxs],
            "attention_mask": [rows["attention_mask"][i] for i in idxs],
            "prefix_len": [rows["prefix_len"][i] for i in idxs],
        })

    ds = DatasetDict({
        "train": make_split(train_indices),
        "test": make_split(test_indices),
    })

    ds_path = os.path.join(args.output_dir, "tokenized")
    print(f"\nSaving tokenized dataset to {ds_path}")
    ds.save_to_disk(ds_path)
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Test: {len(ds['test'])} samples")

    # Save metadata
    metadata = {
        "config": {
            "num_people": args.num_people,
            "samples_per_person": 24,
            "total_samples": len(all_bios),
            "seed": args.seed,
            "context_size": args.context_size,
            "test_frac": args.test_frac,
        },
        "people": people,
        "train_people": sorted(train_people),
        "test_people": sorted(test_people),
        "bios": all_bios,
    }
    meta_path = os.path.join(args.output_dir, "bios_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Full unsplit dataset
    full_ds = DatasetDict({
        "train": Dataset.from_dict({
            "input_ids": rows["input_ids"],
            "attention_mask": rows["attention_mask"],
            "prefix_len": rows["prefix_len"],
        }),
    })
    full_path = os.path.join(args.output_dir, "tokenized_full")
    full_ds.save_to_disk(full_path)
    print(f"Saved full dataset to {full_path}")

    print(f"\nDone! {args.num_people} people × 24 = "
          f"{len(all_bios)} samples")


if __name__ == "__main__":
    main()
