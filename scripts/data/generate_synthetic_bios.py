#!/usr/bin/env python3
"""Generate synthetic bio dataset for memorization evaluation.

Each person has globally unique name, profession, hobby, and salary.
For each person, we generate all 24 permutations of the 4 attribute
statements (4 target choices × 6 prefix orderings).

The first sentence always includes the person's name. Subsequent
sentences use He/She pronouns.

Example:
    Alice works as a Doctor. She is 25 years old. She loves to swim. She earns $80,000.
    Alice loves to swim. She earns $80,000. She works as a Doctor. She is 25 years old.

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
    "swim", "read books", "play chess", "paint landscapes",
    "cook Italian food", "play guitar", "go hiking", "ride bicycles",
    "write poetry", "practice yoga", "play tennis", "go fishing",
    "collect stamps", "build model trains", "solve puzzles",
    "watch documentaries", "study astronomy", "garden",
    "bake pastries", "play piano", "sing in a choir",
    "do woodworking", "knit scarves", "play basketball",
    "go rock climbing", "surf", "ski", "skateboard",
    "take photographs", "draw portraits", "sculpt clay",
    "play soccer", "run marathons", "practice martial arts",
    "play volleyball", "brew coffee", "meditate",
    "learn languages", "do crossword puzzles", "fly kites",
    "go birdwatching", "stargaze", "sail boats",
    "restore furniture", "play drums", "play violin",
    "dance salsa", "do calligraphy", "juggle",
    "go camping", "explore caves", "ride horses",
    "raise chickens", "train dogs", "grow orchids",
    "collect vinyl records", "build robots", "code games",
    "play harmonica", "whittle wood", "weave baskets",
    "make pottery", "blow glass", "forge metal",
    "brew beer", "make candles", "press flowers",
    "fold origami", "do magic tricks", "play badminton",
    "go kayaking", "go snorkeling", "go scuba diving",
    "play table tennis", "do archery", "fence",
    "play cricket", "go bowling", "ice skate",
    "rollerblade", "play golf", "do gymnastics",
    "go paragliding", "fly drones", "go mountain biking",
    "play squash", "do triathlons", "practice fencing",
    "go trail running", "play water polo", "do parkour",
    "go windsurfing", "play lacrosse", "do powerlifting",
    "go canoeing", "climb mountains", "go spelunking",
    "practice tai chi", "go snowboarding", "play darts",
    # --- Extended hobbies ---
    "play ukulele", "play banjo", "play mandolin",
    "play accordion", "play flute", "play saxophone",
    "play trumpet", "play trombone", "play clarinet",
    "play cello", "play harp", "play oboe",
    "build birdhouses", "carve pumpkins", "make soap",
    "make cheese", "make jam", "make pasta",
    "make sushi", "make bread", "make chocolate",
    "make wine", "make cider", "make kombucha",
    "do pottery", "do embroidery", "do quilting",
    "do crochet", "do macrame", "do beadwork",
    "do mosaics", "do stenciling", "do etching",
    "do batik", "do tie-dye", "do screen printing",
    "collect coins", "collect seashells", "collect fossils",
    "collect buttons", "collect postcards", "collect figurines",
    "collect watches", "collect maps", "collect minerals",
    "raise bees", "raise rabbits", "raise goats",
    "raise pigeons", "raise turtles", "raise parrots",
    "grow bonsai trees", "grow cacti", "grow sunflowers",
    "grow herbs", "grow mushrooms", "grow lavender",
    "grow tomatoes", "grow strawberries", "grow peppers",
    "go paddleboarding", "go kitesurfing", "go wakeboarding",
    "go waterskiing", "go rafting", "go rowing",
    "go sailing", "go jet skiing", "go cliff diving",
    "go sledding", "go tobogganing", "go ice fishing",
    "go cross-country skiing", "go snowshoeing", "go curling",
    "do ballet", "do tap dancing", "do breakdancing",
    "do swing dancing", "do ballroom dancing", "do flamenco",
    "do line dancing", "do square dancing", "do folk dancing",
    "do judo", "do karate", "do taekwondo",
    "do boxing", "do wrestling", "do kickboxing",
    "do capoeira", "do aikido", "do kendo",
    "do fencing", "do sumo", "do jiu-jitsu",
    "write novels", "write screenplays", "write haiku",
    "write essays", "write memoirs", "write comics",
    "write songs", "write plays", "write satire",
    "study philosophy", "study history", "study geology",
    "study botany", "study zoology", "study mythology",
    "study genealogy", "study cartography", "study meteorology",
    "study paleontology", "study archaeology", "study linguistics",
    "play online games", "play board games", "play card games",
    "play trivia", "play charades", "play Scrabble",
    "play dominoes", "play backgammon", "play mahjong",
    "play pinball", "play billiards", "play foosball",
    "fly model airplanes", "build model rockets", "build model ships",
    "build terrariums", "build aquariums", "build dollhouses",
    "build treehouses", "build sandcastles", "build igloos",
    "do magic", "do stand-up comedy", "do improv",
    "do puppetry", "do mime", "do ventriloquism",
    "do speed cubing", "do sudoku", "do jigsaw puzzles",
    "do cryptic crosswords", "do logic puzzles", "do tangrams",
    "do geocaching", "do letterboxing", "do orienteering",
    "do bouldering", "do rappelling", "do zip-lining",
    "do horseback riding", "do dog sledding", "do camel riding",
    "do whale watching", "do safari tours", "do stargazing tours",
    "do nature walks", "do bird photography", "do landscape painting",
    "do watercolor painting", "do oil painting", "do charcoal drawing",
    "do pastel drawing", "do pen sketching", "do digital art",
    "do pixel art", "do 3D modeling", "do animation",
    "do stop-motion filmmaking", "do documentary filmmaking", "do podcasting",
    "do blogging", "do vlogging", "do journaling",
    "do scrapbooking", "do bookbinding", "do paper marbling",
    "do leatherworking", "do silversmithing", "do goldsmithing",
    "do clockmaking", "do luthiery", "do upholstery",
    # --- Additional hobbies to reach 400 ---
    "teach cooking classes", "practice calligraphy", "do volunteer work",
    "go antique shopping", "visit museums", "attend concerts",
    "watch sunsets", "explore flea markets", "visit art galleries",
    "go ghost hunting", "do escape rooms", "play laser tag",
    "go paintballing", "do axe throwing", "go go-karting",
    "do rock polishing", "collect autographs", "collect art prints",
    "collect comic books", "collect trading cards", "collect antiques",
    "collect typewriters", "collect cameras", "collect telescopes",
    "design websites", "design logos", "design jewelry",
    "design clothing", "design furniture", "design gardens",
    "build furniture", "build canoes", "build guitars",
    "build clocks", "build computers", "build drones",
    "build kites", "build telescopes", "build cabinets",
    "cook Thai food", "cook French food", "cook Mexican food",
    "cook Japanese food", "cook Indian food", "cook Greek food",
    "cook Korean food", "cook Ethiopian food", "cook Moroccan food",
    "study philosophy online", "study psychology", "study economics",
    "study computer science", "study mathematics", "study chemistry",
    "study marine biology", "study astrophysics", "study neuroscience",
    "practice mindfulness", "practice breathing exercises", "practice qigong",
    "practice pilates", "practice barre", "practice stretching",
    "go urban exploring", "go treasure hunting", "go metal detecting",
    "go fossil hunting", "go beachcombing", "go mushroom foraging",
    "go tide pooling", "go storm chasing", "go aurora hunting",
    "train for obstacle courses", "train for cycling races", "train for swim meets",
    "restore vintage cars", "restore antique clocks", "restore old paintings",
    "restore vintage radios", "restore antique books", "restore old maps",
    "play croquet", "play bocce", "play horseshoes",
    "play shuffleboard", "play cornhole", "play disc golf",
    "play handball", "play polo", "play rugby",
    "play field hockey", "play softball", "play baseball",
    "do tai chi", "do aerobics", "do spinning",
    "do weight training", "do CrossFit", "do kettlebell workouts",
    "do resistance training", "do calisthenics", "do plyometrics",
    "tutor students", "learn sign language", "do flower arranging",
    "practice archery", "do glass etching", "make terrariums",
    "play recorders", "do sand art", "build model castles",
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
        return f"{subject} loves to {hobby}."
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

    # Salary pool: enough unique values
    salary_pool = list(range(25_000, 25_000 + args.num_people * 1_000,
                             1_000))

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

    # Assign random ages
    ages = [random.randint(22, 65) for _ in range(args.num_people)]

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
