#!/usr/bin/env python3
"""Build controlled continuity-error insertions for renamed stories.

The specs in this file identify exact anchor/violation locations by locator
phrases. The script extracts byte-exact quotes from renamed.txt, applies only
the specified minimal edits to hallucinated.txt, and writes insertions.json.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
WORD_RE = re.compile(r"\b\w+\b")
CAPS = {"5k": 8, "10k": 15, "15k": 20}


@dataclass(frozen=True)
class Spec:
    entity: str
    anchor: str
    violation: str
    old: str
    new: str
    fact: str
    difficulty: str
    difficulty_reason: str
    severity: str
    severity_reason: str
    anchor_mode: str = "sentence"
    violation_mode: str = "sentence"
    other_terms: tuple[str, ...] = ()
    anchor_occurrence: int = 1
    violation_occurrence: int = 1


SPECS: dict[str, list[Spec]] = {
    "cavalleria_rusticana": [
        Spec(
            "square church",
            "At right, in background, a church.",
            "All enter the church, except Rosalia and Grazia.",
            "church",
            "inn",
            "The square's church is established at the right; later the crowd enters the inn as if it were the church.",
            "easy",
            "The same public building is explicitly named in both places.",
            "minor",
            "The staging geography is wrong but the plot can still continue.",
        ),
        Spec(
            "Grazia's inn",
            "At left, the inn and dwelling of Nanna Grazia.",
            "Grazia crosses and enters the inn.",
            "inn",
            "church",
            "Grazia's dwelling is the inn; later she enters the church instead of that dwelling.",
            "easy",
            "The object/place label changes directly.",
            "minor",
            "A reader notices a staging mismatch.",
        ),
        Spec(
            "Nicolino's fate",
            "They embrace, Nicolino bites Beppe's ear, viciously.",
            "Neighbor Nicolino is murdered.",
            "murdered",
            "married",
            "The duel challenge establishes mortal violence; later the reported result becomes a wedding.",
            "hard",
            "The reader must connect the duel custom to the offstage report several scenes later.",
            "major",
            "The tragic ending turns into an incompatible social event.",
        ),
    ],
    "the_inca_of_perusalem": [
        Spec(
            "Duchess's hat and gloves",
            "She takes of her gloves and hat: puts them on the table; and sits down.",
            "noticing that she has left her hat and gloves on the table",
            "hat and gloves",
            "hat and umbrella",
            "The Duchess leaves gloves and a hat on the table; later one item has silently become an umbrella.",
            "easy",
            "The contradiction is a direct object substitution.",
            "minor",
            "Only a prop detail changes.",
        ),
        Spec(
            "Clarimond's waterproof",
            "straight waterproof with a hood over her head gear",
            "Clarimond whips off her waterproof",
            "waterproof",
            "velvet cloak",
            "Clarimond is wearing a waterproof; later she removes a velvet cloak.",
            "easy",
            "The garment is explicitly named in both places.",
            "minor",
            "The disguise prop changes without explanation.",
        ),
        Spec(
            "jewel case",
            "jewel case on the table",
            "replacing the case on the table",
            "table",
            "mantelpiece",
            "The jewel case is placed on the table; later it is replaced on the mantelpiece.",
            "medium",
            "The reader must track where the prop was put across blocking directions.",
            "moderate",
            "The business with the jewel case becomes spatially inconsistent.",
        ),
        Spec(
            "Khan's exile",
            "the Khan will be sent to San Cordelia.",
            "Send the Khan to San Cordelia, madam,",
            "Send the Khan to San Cordelia",
            "Send the Khan to Hohenfeld",
            "The predicted exile destination is San Cordelia; later the Khan asks to be sent to Hohenfeld instead.",
            "medium",
            "The inconsistency depends on remembering an earlier political prediction.",
            "moderate",
            "The joke about exile stops lining up.",
        ),
        Spec(
            "Duchess disclosure",
            "A spinster Duchess, hatted and gloved",
            "Oh, by the way, there is a Duchess,",
            "Duchess",
            "widow",
            "A Duchess is established as present; later the Khan identifies her as a widow instead.",
            "hard",
            "The issue is tracking the title of the hidden visitor across the deception plot.",
            "major",
            "The deception plot depends on who knows the Duchess is present.",
        ),
    ],
    "the_adventure_of_the_devil_s_foot": [
        Spec(
            "Hale's telegram",
            "he has never been known to write where a telegram would serve",
            "received a telegram from Hale",
            "received a telegram",
            "received a letter",
            "Hale's habit makes a telegram expected; the later message becomes a letter.",
            "easy",
            "The communication medium is explicit.",
            "minor",
            "A small procedural detail changes.",
        ),
        Spec(
            "Clara's death interval",
            "Clara lying dead of fright",
            "even in death",
            "death",
            "sleep",
            "Clara is established as dead in her chair; later her state is softened into sleep.",
            "easy",
            "The life/death state is explicit.",
            "major",
            "The central crime scene no longer makes sense.",
        ),
        Spec(
            "family card party",
            "his two brothers, Edwin and Martin, and of his sister Clara",
            "while the two brothers sat on each side of her",
            "two",
            "three",
            "Only two brothers are present; later the account says three brothers were affected.",
            "medium",
            "The detector must combine the family count with the later group description.",
            "moderate",
            "The witness account gains an extra sibling.",
        ),
        Spec(
            "housekeeper departure",
            "Mrs. Keene, the old cook and housekeeper",
            "the elderly Westmoorish housekeeper, Mrs. Keene",
            "housekeeper",
            "gardener",
            "Mrs. Keene is established as the housekeeper; later she is identified as the gardener.",
            "medium",
            "It requires tracking a household role across the investigation.",
            "moderate",
            "The household witness's role becomes confused.",
        ),
        Spec(
            "Penrane lamp",
            "bought a lamp which was the duplicate",
            "The lamp shining in broad daylight",
            "lamp",
            "candle",
            "Hale buys a duplicate lamp for the experiment; later the corresponding light is a candle.",
            "medium",
            "The detector must connect the experimental prop to the final recollection.",
            "moderate",
            "The mechanism of the poison experiment changes.",
        ),
        Spec(
            "Arthur Penrane's death",
            "The killing of Arthur Penrane",
            "after the death of Penrane",
            "death",
            "escape",
            "Arthur Penrane's killing is established; later the same event is recast as an escape.",
            "hard",
            "The detector must connect the formal case summary to the later reconstructed action.",
            "major",
            "The murder sequence no longer has a victim.",
        ),
        Spec(
            "Ashcombe confession leverage",
            "I give you my assurance that the matter will pass out of my hands forever.",
            "Now I have told you all.",
            "told you all",
            "told you nothing",
            "After Hale's ultimatum forces Ashcombe to disclose the truth, the confession says nothing was disclosed.",
            "hard",
            "The contradiction is in Ashcombe's compliance with pressure, not a simple attribute.",
            "major",
            "The confession scene collapses.",
        ),
    ],
    "how_he_lied_to_her_husband": [
        Spec(
            "drawing-room piano",
            "the grand piano along the opposite\nwall to his left",
            "the nearest seat, which happens to be the bench at the piano.",
            "piano",
            "fireplace",
            "The piano is on the left wall; later the piano bench is treated as the fireplace seat.",
            "easy",
            "The room fixture is explicitly identified.",
            "minor",
            "Only blocking in the drawing room is disturbed.",
        ),
        Spec(
            "Cecilia's diamonds",
            "wears many diamonds",
            "we shall leave your diamonds here",
            "your diamonds",
            "your pearls",
            "Cecilia is wearing diamonds; later those same jewels are pearls.",
            "easy",
            "The jewelry type is stated plainly.",
            "minor",
            "A costume detail changes.",
        ),
        Spec(
            "opera tickets",
            "Lyceum Rooms tickets",
            "takes the Seraphina tickets out of his pocket",
            "Seraphina",
            "Tannhauser",
            "The tickets in play are for Seraphina; later the tickets are for Tannhauser.",
            "medium",
            "The reader must track the prop through the argument.",
            "moderate",
            "The quarrel over the outing loses its object.",
        ),
        Spec(
            "glove fastening",
            "If I button my glove",
            "let that glove alone",
            "glove",
            "hat",
            "The object being used for the tableau is a glove; later Percy calls it a hat.",
            "medium",
            "The detector must follow the physical ruse across turns.",
            "moderate",
            "The attempted cover story becomes incoherent.",
        ),
        Spec(
            "Percy's jealousy",
            "looking him up and down with growing resentment",
            "grinning] Faversham: that's really very ready of you.",
            "grinning",
            "weeping",
            "Percy enters resentful and suspicious; moments later his reaction is made emotionally incompatible.",
            "hard",
            "The error turns on character affect across the confrontation.",
            "major",
            "The scene's comic escalation breaks.",
        ),
    ],
    "a_martian_odyssey": [
        Spec(
            "Aster air",
            "learning to breathe the air as tenuous as that of Aster",
            "fire in this atmosphere",
            "atmosphere",
            "poisonous",
            "Aster's air is established as breathable and thin; later the same atmosphere is treated as poison.",
            "easy",
            "The atmospheric property is explicit.",
            "major",
            "The survival premise breaks.",
        ),
        Spec(
            "Twaal's height",
            "It stood an inch or so taller than I",
            "a little roundish body, and a long neck ending in a tiny head",
            "little roundish\nbody",
            "giant roundish\nbody",
            "Twaal is only slightly taller than Mercer; later its body becomes giant.",
            "easy",
            "The creature's size changes directly.",
            "minor",
            "The alien's visual description is inconsistent.",
        ),
        Spec(
            "Twaal's weapon",
            "a queer little glass revolver out of his bag",
            "Twaal had his glass gun out",
            "glass gun",
            "glass knife",
            "Twaal's weapon is established as a glass revolver; later it becomes a glass knife.",
            "easy",
            "The carried weapon changes directly.",
            "moderate",
            "The combat scene loses its established weapon.",
        ),
        Spec(
            "sun-circle target",
            "my sun-circle in the sand--a bull's eye",
            "pointed at a pyramid",
            "pyramid",
            "sun-circle",
            "Mercer's sand symbol is a sun-circle; later a pyramid is substituted as the referent.",
            "medium",
            "The detector must track a drawn symbol through the communication attempts.",
            "moderate",
            "The first-contact exchange becomes confused.",
        ),
        Spec(
            "water cup",
            "squirted it all back into the cup",
            "dumped my water tank on the floor",
            "water tank",
            "empty cup",
            "The liquid is handled in a cup; later the related water supply is an empty cup.",
            "medium",
            "It requires linking repeated water-handling props.",
            "moderate",
            "The survival/action sequence becomes less coherent.",
        ),
        Spec(
            "Twaal rescue",
            "like the one I'd saved Twaal from",
            "watching it die while Twaal trilled and whistled",
            "trilled and whistled",
            "lay dead",
            "Twaal survives the black horror; later the aftermath implies Twaal is dead.",
            "hard",
            "The contradiction is implied by survival and action after a rescue.",
            "major",
            "A central companion is killed while still needed for later events.",
        ),
    ],
    "life_in_the_iron_mills": [
        Spec(
            "rainy night",
            "One rainy night, about eleven o'clock",
            "The rain was falling heavily",
            "The rain",
            "The snow",
            "The night is rainy; later the same weather becomes snow.",
            "easy",
            "The weather value is directly repeated.",
            "minor",
            "Atmosphere changes but the plot survives.",
        ),
        Spec(
            "Martha's route",
            "the woman crouched out of his sight against the wall",
            "Martha, crouching near by on the other side of the wall",
            "wall",
            "river",
            "Martha reaches the mill wall; later her hiding place becomes the river.",
            "easy",
            "The hiding location changes explicitly.",
            "moderate",
            "The blocking of the theft scene is wrong.",
        ),
        Spec(
            "coach location",
            "Where did we leave the coach, Fairfax?",
            "At the other side of the works.",
            "other side",
            "front gate",
            "The coach was left on the other side of the works; later it is at the front gate.",
            "medium",
            "The answer relies on remembering a location from the previous scene.",
            "moderate",
            "The men's movements no longer line up.",
        ),
        Spec(
            "stolen money",
            "Picking Fairfax's pocket at the very time!",
            "A consciousness of power stirred within him.",
            "power",
            "innocence",
            "Evan has been accused of taking Fairfax's money; later his response implies innocence rather than temptation by stolen power.",
            "medium",
            "The reader must connect the theft accusation with Evan's internal reaction.",
            "moderate",
            "The moral crisis is blunted.",
        ),
        Spec(
            "Evan's death",
            "Not for days or years, but never!--that was it.",
            "Did hur know where they'll bury Evan?",
            "bury",
            "release",
            "Evan's permanent death is established; later Martha asks where they will release him.",
            "hard",
            "The contradiction depends on tracking the prison-death consequence into Martha's grief.",
            "major",
            "The ending's death and burial are undone.",
        ),
    ],
    "peter_pan_in_kensington_gardens": [
        Spec(
            "Robin's age",
            "His age\nis one week",
            "Robin wore no night-gown now.",
            "no night-gown",
            "a grey beard",
            "Robin is a one-week-old child; later he is given a grey beard.",
            "medium",
            "The age contradiction is implied by a later descriptive adjective.",
            "minor",
            "It disturbs the fairy-tale premise but not a scene action.",
        ),
        Spec(
            "goat tradition",
            "Therefore there was\nno goat when your grandmother was a little girl.",
            "to begin with the goat",
            "goat",
            "horse",
            "The tradition concerns a goat; later the opening animal becomes a horse.",
            "easy",
            "The animal changes directly.",
            "minor",
            "A recurring anecdotal detail changes.",
        ),
        Spec(
            "Bartholomew's wrong addresses",
            "Bartholomew has sent to the wrong house.",
            "completely puzzled Bartholomew",
            "puzzled",
            "guided",
            "Bartholomew is established as sending babies to wrong houses; later the mistaken boat guides him instead of puzzling him.",
            "medium",
            "The detector must link his role to the later statement.",
            "moderate",
            "The baby-delivery premise becomes inconsistent.",
        ),
        Spec(
            "Elsie's hiding",
            "hid in Freddie’s stead",
            "away she ran to look for them",
            "ran",
            "slept",
            "Elsie actively hides/substitutes herself; later her pursuit action becomes sleep.",
            "medium",
            "It requires tracking her action through the night episode.",
            "moderate",
            "The chase scene loses its cause.",
        ),
        Spec(
            "Robin's pipe",
            "made a pipe of reeds",
            "it is really Robin’s pipe they hear",
            "pipe",
            "fiddle",
            "Robin's instrument is a reed pipe; later the heard instrument becomes a fiddle.",
            "easy",
            "The musical instrument changes directly.",
            "minor",
            "The fairy orchestra detail becomes inconsistent.",
        ),
        Spec(
            "return to mother",
            "he was quite decided to go back.",
            "The iron bars are up for life.",
            "for life",
            "for an hour",
            "Robin's delayed return ends in permanent lock-out; later the lock-out is temporary.",
            "hard",
            "The contradiction is about the story's final consequence and motivation.",
            "major",
            "The ending's irreversible loss is undone.",
        ),
    ],
    "the_frogs": [
        Spec(
            "Xanthias's burden",
            "I'm getting crushed",
            "I've got a bump upon my rump",
            "bump",
            "crown",
            "The servant complains of physical burden; later the injury becomes a royal crown.",
            "easy",
            "The bodily consequence is explicitly changed.",
            "minor",
            "A comic physical detail changes.",
        ),
        Spec(
            "route to Avernus",
            "Which is the quickest way to get to Avernus?",
            "lead the lovely youthful Chorus\nTo the marshy flowery plain",
            "marshy",
            "mountain",
            "The choral destination is a marshy plain; later it becomes mountainous.",
            "easy",
            "The landscape attribute changes plainly.",
            "minor",
            "The chorus setting shifts.",
        ),
        Spec(
            "Orthros threat",
            "Orthros",
            "fetch them hither",
            "fetch",
            "dismiss",
            "A hellish threat is summoned; later the messenger is told to dismiss it.",
            "medium",
            "The detector must connect the threat setup to the instruction.",
            "moderate",
            "The intimidation scene loses its force.",
        ),
        Spec(
            "poetic contest",
            "Euripides does not exist",
            "Timokles has won with his scales",
            "Timokles",
            "Philokles",
            "The contest is between renamed rival poets; later the victory is assigned to the wrong one.",
            "medium",
            "The reader must track the paired rivals.",
            "moderate",
            "The debate's sides collapse.",
        ),
        Spec(
            "journey purpose",
            "You are really game to go?",
            "Battle, if battle they must, far away in their own fatherland.",
            "fatherland",
            "marketplace",
            "The journey is to the underworld and political restoration; the final destination becomes a marketplace.",
            "hard",
            "The contradiction is thematic and distant from the setup.",
            "major",
            "The play's civic resolution no longer follows.",
        ),
    ],
    "the_black_cat": [
        Spec(
            "Mother Sabine's goodness",
            "Mother Sabine bore the imprint of her amiable disposition",
            "GOODNESS OF MOTHER",
            "GOODNESS",
            "CRUELTY",
            "Mother Sabine is established as amiable and good; later the heading names cruelty instead.",
            "easy",
            "The moral attribute changes explicitly.",
            "minor",
            "A background description is inconsistent.",
            violation_mode="line",
        ),
        Spec(
            "Galifard as enemy",
            "he had to do with an enemy",
            "HIS ENEMY.",
            "ENEMY",
            "FRIEND",
            "Mistifou recognizes Galifard as an enemy; later the heading calls him a friend.",
            "medium",
            "The reader must track Mistifou's relation to Galifard across chapters.",
            "moderate",
            "The struggle with Galifard loses its opposition.",
            violation_mode="line",
        ),
        Spec(
            "Galifard's accomplice",
            "Father Galifard searched for an accomplice.",
            "warn his accomplice",
            "accomplice",
            "judge",
            "Galifard seeks an accomplice; later the accomplice he must warn becomes a judge.",
            "medium",
            "It requires connecting the conspiracy setup with the later concealment.",
            "moderate",
            "The attempted cover-up stops making sense.",
        ),
        Spec(
            "Galifard's execution plot",
            "destruction of the poor and innocent cat",
            "cruel execution of Mistifou",
            "execution",
            "coronation",
            "Galifard plans Mistifou's destruction; later the planned execution becomes a coronation.",
            "hard",
            "The detector must connect Galifard's motive to the later staged action.",
            "major",
            "The attempted killing turns into an impossible celebration.",
        ),
        Spec(
            "Mistifou's death",
            "Mistifou is dead, otherwise",
            "He died,--but it was",
            "died",
            "returned",
            "Mistifou is declared dead; later the final account says he returned instead of died.",
            "hard",
            "The contradiction is in the final memorial consequence.",
            "major",
            "The death ending is replaced by an impossible celebration.",
        ),
    ],
    "the_new_paul_and_virginia": [
        Spec(
            "steamship destination",
            "was bound for Albion",
            "outward bound to her husband",
            "outward",
            "homeward",
            "The ship is homeward bound for Albion; later Amelia is homeward bound away from it.",
            "easy",
            "The direction of travel changes directly.",
            "minor",
            "A voyage detail conflicts.",
        ),
        Spec(
            "Julian's identity",
            "One of the objects of this delightful curiosity was a large-boned,\nmiddle-aged man",
            "Julian's eyes",
            "Julian's",
            "Amelia's",
            "The large-boned man is Julian; later his perception is assigned to Amelia.",
            "medium",
            "The detector must preserve which observer is acting.",
            "moderate",
            "The focalization of the island scene breaks.",
        ),
        Spec(
            "parasol cutter",
            "Amelia's beautiful lace parasol",
            "the cutter to the shore",
            "cutter",
            "coach",
            "The parasol helps bring a cutter ashore; later the vehicle becomes a coach.",
            "easy",
            "The vehicle/object is directly substituted.",
            "moderate",
            "The shipwreck logistics become absurd.",
        ),
        Spec(
            "Amelia's boxes",
            "Amelia's boxes.",
            "large box of knick-knacks",
            "box",
            "basket",
            "Amelia's luggage is boxes; later one of the same useful containers is a basket.",
            "medium",
            "The detector must connect shipwreck luggage with later island supplies.",
            "moderate",
            "The salvage inventory becomes inconsistent.",
        ),
        Spec(
            "castaway island",
            "enchanting island, green",
            "the island was just the place",
            "island",
            "desert",
            "The castaways find a green island; later that same place is called a desert.",
            "easy",
            "The location type changes directly.",
            "moderate",
            "The survival setting becomes geographically inconsistent.",
        ),
        Spec(
            "Professor's private box",
            "tin box of the\nProfessor's, marked 'Private,'",
            "The box was full of papers",
            "box",
            "trunk",
            "The Professor's private container is a tin box; later it is a trunk.",
            "easy",
            "The container type changes directly.",
            "minor",
            "A private-property prop changes form.",
        ),
        Spec(
            "Professor's priesthood",
            "Professor had begun life as a clergyman",
            "Once a priest, always a priest",
            "always a priest",
            "always a sailor",
            "The Professor's former clerical identity is treated as indelible; later that identity becomes a sailor's.",
            "medium",
            "The detector must connect clergyman, priest, and orders across the religious argument.",
            "moderate",
            "Amelia's confession logic stops working.",
            violation_mode="line",
        ),
        Spec(
            "bishop's monkey",
            "nice monkey",
            "monkey which the\nbishop led by a chain",
            "monkey",
            "serpent",
            "Amelia asks for a monkey and the bishop brings one; later the brought animal is a serpent.",
            "hard",
            "The contradiction is tied to the long-running missing-link expectation.",
            "major",
            "The ending's rescue/revelation gag breaks.",
        ),
        Spec(
            "Amelia's final belief",
            "once more believe in heaven.",
            "rushing into the arms of her bishop",
            "bishop",
            "chemist",
            "Amelia's return to religious belief culminates with her bishop; the person becomes a chemist.",
            "hard",
            "The contradiction depends on tracking the ideological arc.",
            "major",
            "The conversion ending is structurally broken.",
        ),
    ],
    "second_variety": [
        Spec(
            "Eurasian soldier",
            "The Eurasian soldier made his way nervously up the ragged side of the hill",
            "watching Eurasian ship",
            "Eurasian",
            "Martian",
            "The soldier is Eurasian; later the same soldier becomes Martian.",
            "easy",
            "The identity label changes explicitly.",
            "minor",
            "A surface identity detail changes.",
        ),
        Spec(
            "note capsule",
            "single Eurasian runner with a message",
            "the note capsule",
            "note",
            "food",
            "The object prompting the mission is a note capsule; later it is a food capsule.",
            "easy",
            "The object type is explicit.",
            "moderate",
            "The mission premise becomes confused.",
        ),
        Spec(
            "Nico's bear",
            "clutching his teddy bear",
            "Each with a ragged teddy bear.",
            "teddy bear",
            "tin sword",
            "The child variety is established by its teddy bear; later the copies each carry a tin sword.",
            "medium",
            "The detector must connect Nico's identifying prop with the later replicated type.",
            "moderate",
            "The identification of the machine variety changes.",
        ),
        Spec(
            "Mara's suspicion",
            "I had been watching him. I was suspicious.",
            "Mara said calmly, from behind them.",
            "calmly",
            "ignorantly",
            "Mara had been suspicious and watchful; later her intervention implies ignorance.",
            "hard",
            "The contradiction is about knowledge and suspicion.",
            "major",
            "The detection of the variety no longer works.",
        ),
        Spec(
            "rocket piloting",
            "I'm not accustomed to rocket piloting",
            "running her fingers over the smooth\nmetal",
            "smooth",
            "familiar",
            "Mara says she is not accustomed to piloting; later the controls are familiar to her.",
            "medium",
            "The contradiction requires tying expertise to the ship controls.",
            "moderate",
            "The escape plan becomes suspect.",
        ),
    ],
    "the_hunting_of_the_snark": [
        Spec(
            "Crier's bell",
            "one notion for crossing the ocean",
            "furious bell",
            "furious bell",
            "furious drum",
            "The Crier crosses the ocean by ringing a bell; later that bell becomes a drum.",
            "easy",
            "The carried signal object changes directly.",
            "minor",
            "A nonsense-prop detail changes.",
        ),
        Spec(
            "blank map",
            "large map representing the sea",
            "A perfect and absolute blank",
            "blank",
            "island",
            "The map is celebrated as having no land; later the same map line becomes an island.",
            "medium",
            "The reader must connect the navigational joke across the fit.",
            "moderate",
            "The navigation gag loses coherence.",
            violation_mode="line",
        ),
        Spec(
            "Borlum warning",
            "If your Skrim be a Borlum!",
            "For the Skrim _was_ a Borlum",
            "Borlum",
            "Badger",
            "The danger is that some Skrims are Borlums; the final Skrim becomes a Badger.",
            "hard",
            "The final reveal contradicts a warning from much earlier.",
            "major",
            "The disappearance ending no longer follows.",
        ),
        Spec(
            "Jibjib threat",
            "Should we meet with a Jibjib",
            "The song of the Jibjib recurred",
            "Jibjib",
            "sparrow",
            "The feared bird is the Jibjib; later the remembered song belongs to a sparrow.",
            "easy",
            "The named creature changes plainly.",
            "minor",
            "The comic danger is diluted.",
        ),
        Spec(
            "Factor's fate",
            "We have lost half the day.",
            "we sha’n’t catch a Skrim before night!",
            "night",
            "noon",
            "Half the day is already lost; later the deadline is noon.",
            "medium",
            "The contradiction depends on elapsed time.",
            "moderate",
            "The urgency of the hunt becomes impossible.",
        ),
    ],
    "the_yellow_wallpaper": [
        Spec(
            "nursery location",
            "So we took the nursery, at the\ntop of the house.",
            "I am sitting by the window now, up in this atrocious nursery",
            "up",
            "down",
            "The nursery is at the top of the house; later the narrator is down in it.",
            "easy",
            "The vertical location changes directly.",
            "minor",
            "The house layout is inconsistent.",
        ),
        Spec(
            "wallpaper color",
            "a smouldering, unclean\nyellow",
            "It is the strangest yellow, that wallpaper!",
            "yellow",
            "violet",
            "The wallpaper is yellow; later the wallpaper itself is violet.",
            "easy",
            "The color attribute is explicit.",
            "minor",
            "A cosmetic but recognizable detail changes.",
        ),
        Spec(
            "immovable bed",
            "this great heavy bed, which is\nall we found in the room",
            "great immovable bed",
            "immovable",
            "rolling",
            "The bed is heavy and fixed; later the same bed is rolling.",
            "medium",
            "The detector must connect earlier bed descriptions to a later attempted action.",
            "moderate",
            "The narrator's physical struggle stops making sense.",
        ),
        Spec(
            "barred windows",
            "the windows are barred\nfor little children",
            "it becomes bars!",
            "bars",
            "ribbons",
            "The windows are barred; later the pattern's bars become ribbons.",
            "medium",
            "It requires tracking the window hardware across the room descriptions.",
            "moderate",
            "The confinement logic breaks.",
        ),
        Spec(
            "hidden rope",
            "I’ve got a rope up here",
            "I am securely fastened now by my well-hidden rope",
            "securely fastened",
            "completely unbound",
            "The narrator fastens herself with a hidden rope; later she is unbound.",
            "hard",
            "The contradiction is implied through the narrator's plan and final physical state.",
            "major",
            "The final locked-room action no longer works.",
        ),
    ],
    "the_witch_of_atlas": [
        Spec(
            "vision's wings",
            "winged Vision came",
            "With folded wings and unawakened eyes",
            "wings",
            "roots",
            "The vision is winged; later it beats roots.",
            "easy",
            "The physical attribute changes directly.",
            "minor",
            "A poetic image becomes inconsistent.",
            violation_mode="line",
        ),
        Spec(
            "fountain trance",
            "This lady never slept",
            "All night within the fountain",
            "fountain",
            "furnace",
            "The lady lies in a fountain; later the place becomes a furnace.",
            "easy",
            "The location/object changes explicitly.",
            "moderate",
            "The image's physical setting becomes impossible.",
            anchor_mode="line",
            violation_mode="line",
        ),
        Spec(
            "shooting star",
            "Of shooting stars",
            "Circling the image of a shooting star",
            "shooting star",
            "falling stone",
            "A shooting-star image is established; later the same figure becomes a falling stone.",
            "medium",
            "The detector must connect material imagery across stanzas.",
            "minor",
            "The symbolic object changes material.",
            violation_mode="line",
        ),
        Spec(
            "old age figure",
            "Old age with snow-bright hair",
            "snow-bright hair",
            "snow-bright",
            "raven-black",
            "Old age is pictured with white hair; later the hair is black.",
            "medium",
            "The contradiction requires interpreting the descriptive compound.",
            "minor",
            "A personification's appearance changes.",
            anchor_mode="line",
            violation_mode="line",
        ),
        Spec(
            "final belief",
            "Scarcely believe much more than we can see.",
            "weird winter nights",
            "winter",
            "summer",
            "The poem contrasts winter nights with garish summer days; later winter is made summer.",
            "hard",
            "The contradiction depends on the closing contrast across the poem.",
            "major",
            "The final seasonal opposition collapses.",
            anchor_mode="line",
            violation_mode="line",
        ),
    ],
    "the_adventure_of_the_dying_detective": [
        Spec(
            "landlady",
            "Mrs. Barlow, the landlady of Edmund Hale",
            "Mrs. Barlow was right.",
            "Mrs. Barlow",
            "Mr. Barlow",
            "The landlady is Mrs. Barlow; later the same person is Mr. Barlow.",
            "easy",
            "The form of address changes explicitly.",
            "minor",
            "A character title changes.",
            violation_mode="line",
        ),
        Spec(
            "contagion by touch",
            "Contagious by touch",
            "Keep your distance and all is well.",
            "distance",
            "handshake",
            "The illness is dangerous by touch; later safety depends on a handshake.",
            "medium",
            "The detector must infer why distance matters.",
            "major",
            "The fake medical danger becomes incoherent.",
        ),
        Spec(
            "Roderick's address",
            "of 13 Lower Fenwick Street",
            "Mr. Roderick Vane is in.",
            "Roderick Vane",
            "Roderick Harker",
            "Roderick is sent for at Lower Fenwick Street; later the answering servant names a different Roderick.",
            "easy",
            "The named resident changes directly.",
            "moderate",
            "The visit scene cannot proceed normally.",
            violation_mode="line",
        ),
        Spec(
            "Felix Darrow's death",
            "about Felix Darrow's death",
            "murder of one Felix Darrow",
            "murder",
            "wedding",
            "Felix's death is a murder; later the charge becomes a wedding.",
            "medium",
            "The reader must connect the earlier death allusion with the formal charge.",
            "major",
            "The case's crime disappears.",
        ),
        Spec(
            "Hale's deception",
            "It was very essential that I should impress Mrs. Barlow with the reality of my condition",
            "My dear Bennett, I owe you a thousand apologies.",
            "apologies",
            "accusations",
            "Hale admits staging the illness to manipulate Bennett; later he treats Bennett as culpable.",
            "hard",
            "The contradiction is in intent and interpersonal knowledge.",
            "major",
            "The reveal no longer explains the deception.",
        ),
    ],
    "chitra": [
        Spec(
            "temple guest",
            "I live in this temple.",
            "day when a woman came to you in the temple of Mahadeva",
            "temple",
            "market",
            "The meeting place is a temple; later it becomes a market.",
            "easy",
            "The location changes directly.",
            "minor",
            "The remembered encounter shifts setting.",
        ),
        Spec(
            "Malini's disguise",
            "Enter MALINI, dressed as a woman.",
            "unveiling in her original male attire",
            "male",
            "bridal",
            "Malini's original attire is male; later it becomes bridal.",
            "easy",
            "The clothing state changes plainly.",
            "moderate",
            "The reveal of identity is muddled.",
        ),
        Spec(
            "one-year charm",
            "for one whole year the charm of spring blossoms shall nestle round thy limbs.",
            "the year, almost at its end",
            "year",
            "month",
            "The divine charm lasts one year; later its term is a month.",
            "medium",
            "The detector must track elapsed magical duration.",
            "moderate",
            "The time pressure changes.",
        ),
        Spec(
            "northern robbers",
            "pouring from the northern hills",
            "from the northern hills",
            "northern",
            "southern",
            "The robbers come from northern hills; later that origin becomes southern.",
            "easy",
            "The direction changes explicitly.",
            "minor",
            "A threat location shifts.",
        ),
        Spec(
            "final self-offering",
            "Today I can only offer you Malini, the daughter of a king.",
            "Beloved, my life is full.",
            "full",
            "empty",
            "Malini's final self-revelation fulfills her life; the closing state becomes emptiness.",
            "hard",
            "The contradiction is in the emotional resolution after identity disclosure.",
            "major",
            "The ending's reconciliation reverses.",
        ),
    ],
    "a_pail_of_air": [
        Spec(
            "dark star",
            "The dark star passed",
            "when the dark star captured the Tellus.",
            "dark star",
            "bright moon",
            "The catastrophe is caused by a dark star; later it is a bright moon.",
            "easy",
            "The cosmic object changes explicitly.",
            "major",
            "The physical premise breaks.",
        ),
        Spec(
            "Dad's pail",
            "Dad had sent me out to get an extra pail of air.",
            "take the pail from me",
            "pail",
            "lantern",
            "The needed object is a pail of air; later Dad takes a lantern.",
            "easy",
            "The object changes directly.",
            "moderate",
            "The survival routine becomes confused.",
        ),
        Spec(
            "hammer by hand",
            "his hand went out until it touched and gripped the handle of the hammer beside him.",
            "In through the blanket stepped the beautiful young lady.",
            "stepped",
            "crawled",
            "Dad is prepared with a hammer for an intruder; later the entry action implies a different threat posture.",
            "medium",
            "The detector must relate the defensive setup to the actual entry.",
            "moderate",
            "The confrontation's blocking changes.",
        ),
        Spec(
            "the Burrow",
            "quit crowding the Burrow",
            "let us be alone",
            "alone",
            "outside",
            "The narrator wants privacy inside the Burrow; later the desire becomes being outside.",
            "medium",
            "It requires tracking the family's shelter as safe interior space.",
            "moderate",
            "The shelter premise is weakened.",
        ),
        Spec(
            "waiting ten years",
            "I'll be twenty in only ten years.",
            "beautiful young lady will wait for me",
            "wait",
            "forget",
            "The narrator hopes the young lady will wait ten years; later she will forget him.",
            "hard",
            "The contradiction is about the narrator's final hope and future expectation.",
            "major",
            "The ending's emotional premise is reversed.",
        ),
    ],
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def target_for(bucket: str, total_scenes: int) -> int:
    return min(int(total_scenes * 0.6 + 0.5), CAPS[bucket], total_scenes - 1)


def plan_for(target: int) -> dict[str, int]:
    if target < 5:
        return {"easy": 0, "medium": 0, "hard": 0}
    hard = max(1, round(target * 0.2))
    easy = round(target * 0.4)
    medium = target - easy - hard
    return {"easy": easy, "medium": medium, "hard": hard}


def story_dirs() -> list[Path]:
    return sorted(path.parent for path in ROOT.glob("*/*/*/renamed.txt"))


def load_meta(story_dir: Path) -> dict:
    return json.loads((story_dir / "metadata.json").read_text(encoding="utf-8"))


def load_scenes(story_dir: Path) -> dict:
    return json.loads((story_dir / "scenes.json").read_text(encoding="utf-8"))


def scene_for_offset(scenes: list[dict], offset: int) -> int:
    for scene in scenes:
        if scene["char_start"] <= offset < scene["char_end"]:
            return scene["scene"]
    return scenes[-1]["scene"]


def find_occurrence(text: str, phrase: str, occurrence: int) -> int:
    pattern_text = r"\s+".join(re.escape(part) for part in phrase.split())
    hits = [match.start() for match in re.finditer(pattern_text, text)]
    if not hits:
        raise ValueError(f"Expected at least one hit for {phrase!r}, found 0")
    if occurrence < 0:
        idx = len(hits) + occurrence
    else:
        idx = occurrence - 1
    if idx < 0 or idx >= len(hits):
        raise ValueError(
            f"Expected occurrence {occurrence} for {phrase!r}, found {len(hits)}"
        )
    return hits[idx]


def line_span(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return start, end


def paragraph_span(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, pos)
    start = 0 if start == -1 else start + 2
    end = text.find("\n\n", pos)
    end = len(text) if end == -1 else end
    return start, end


def sentence_span(text: str, pos: int) -> tuple[int, int]:
    para_start, para_end = paragraph_span(text, pos)
    rel = pos - para_start
    para = text[para_start:para_end]
    starts = [0]
    for match in re.finditer(r"(?<=[.!?])(?:[\"'”’\]\)]*)\s+", para):
        starts.append(match.end())
    starts.append(len(para))
    for left, right in zip(starts, starts[1:]):
        if left <= rel < right:
            while left < right and para[left].isspace():
                left += 1
            while right > left and para[right - 1].isspace():
                right -= 1
            return para_start + left, para_start + right
    return para_start, para_end


def quote_span(text: str, phrase: str, mode: str, occurrence: int) -> tuple[int, int]:
    pos = find_occurrence(text, phrase, occurrence)
    if mode == "line":
        return line_span(text, pos)
    if mode == "paragraph":
        return paragraph_span(text, pos)
    if mode == "sentence":
        return sentence_span(text, pos)
    raise ValueError(f"Unknown quote mode: {mode}")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for paragraph in re.split(r"(\n\n)", text):
        if paragraph == "\n\n":
            offset += len(paragraph)
            continue
        if not paragraph.strip():
            offset += len(paragraph)
            continue
        starts = [0]
        for match in re.finditer(r"(?<=[.!?])(?:[\"'”’\]\)]*)\s+", paragraph):
            starts.append(match.end())
        starts.append(len(paragraph))
        for left, right in zip(starts, starts[1:]):
            while left < right and paragraph[left].isspace():
                left += 1
            while right > left and paragraph[right - 1].isspace():
                right -= 1
            if right > left:
                spans.append((offset + left, offset + right))
        offset += len(paragraph)
    return spans


def context_around(text: str, quote_start: int, quote_end: int) -> str:
    spans = sentence_spans(text)
    idx = next(
        (i for i, (start, end) in enumerate(spans) if start <= quote_start and quote_end <= end),
        None,
    )
    if idx is None:
        start, end = paragraph_span(text, quote_start)
        return text[start:end]
    start = spans[max(0, idx - 2)][0]
    end = spans[min(len(spans) - 1, idx + 2)][1]
    return text[start:end]


def words_changed(original: str, edited: str) -> int:
    before = WORD_RE.findall(original)
    after = WORD_RE.findall(edited)
    matcher = difflib.SequenceMatcher(a=before, b=after)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def other_mentions(
    text: str,
    scenes: list[dict],
    anchor_span: tuple[int, int],
    violation_span: tuple[int, int],
    terms: tuple[str, ...],
    old_value: str,
) -> list[dict]:
    if not terms:
        return []
    spans = sentence_spans(text)
    out = []
    for start, end in spans:
        if (start, end) in {anchor_span, violation_span}:
            continue
        quote = text[start:end]
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", quote, re.I) for term in terms):
            out.append(
                {
                    "scene": scene_for_offset(scenes, start),
                    "quote": quote,
                    "still_conflicts": bool(
                        re.search(rf"(?<!\w){re.escape(old_value)}(?!\w)", quote, re.I)
                    ),
                }
            )
    return out


def make_insertion(
    idx: int,
    spec: Spec,
    text: str,
    scenes: list[dict],
) -> dict:
    anchor_span_ = quote_span(text, spec.anchor, spec.anchor_mode, spec.anchor_occurrence)
    violation_span_ = quote_span(
        text, spec.violation, spec.violation_mode, spec.violation_occurrence
    )
    anchor_quote = text[anchor_span_[0] : anchor_span_[1]]
    violation_quote = text[violation_span_[0] : violation_span_[1]]
    if spec.old not in violation_quote:
        raise ValueError(f"Edit source {spec.old!r} not found in violation quote")
    if violation_quote.count(spec.old) != 1:
        raise ValueError(f"Edit source {spec.old!r} appears more than once in violation quote")
    edited = violation_quote.replace(spec.old, spec.new, 1)
    changed = words_changed(violation_quote, edited)
    anchor_scene = scene_for_offset(scenes, anchor_span_[0])
    violation_scene = scene_for_offset(scenes, violation_span_[0])
    return {
        "id": idx,
        "anchor_entity": spec.entity,
        "anchor_scene": anchor_scene,
        "anchor_quote": anchor_quote,
        "anchor_context": context_around(text, anchor_span_[0], anchor_span_[1]),
        "violation_scene": violation_scene,
        "violation_quote_original": violation_quote,
        "violation_quote_edited": edited,
        "violation_context": context_around(text, violation_span_[0], violation_span_[1]),
        "char_offset": violation_span_[0],
        "words_changed": changed,
        "fact": spec.fact,
        "difficulty": spec.difficulty,
        "difficulty_reason": spec.difficulty_reason,
        "severity": spec.severity,
        "severity_reason": spec.severity_reason,
        "scene_distance": violation_scene - anchor_scene,
        "other_mentions": other_mentions(
            text, scenes, anchor_span_, violation_span_, spec.other_terms, spec.old
        ),
    }


def filter_valid_insertions(insertions: list[dict]) -> tuple[list[dict], list[dict]]:
    dropped = []
    kept = []
    entities = set()
    violation_scenes = set()
    for item in insertions:
        failures = []
        if item["anchor_entity"] in entities:
            failures.append(f"duplicate anchor_entity: {item['anchor_entity']}")
        if item["violation_scene"] in violation_scenes:
            failures.append(f"duplicate violation_scene: {item['violation_scene']}")
        if item["violation_scene"] < item["anchor_scene"]:
            failures.append(f"violation not later than anchor for {item['id']}")
        if item["violation_scene"] == 1:
            failures.append(f"violation in scene 1 for {item['id']}")
        if item["anchor_quote"] == item["violation_quote_original"]:
            failures.append(f"anchor and violation quote are identical for {item['id']}")
        if item["words_changed"] > 5:
            failures.append(f"too many words changed for {item['id']}: {item['words_changed']}")
        if any(mention["still_conflicts"] for mention in item["other_mentions"]):
            failures.append(f"cascade conflict remains for {item['id']}")
        if failures:
            dropped.append({"reason": "; ".join(failures)})
            continue
        entities.add(item["anchor_entity"])
        violation_scenes.add(item["violation_scene"])
        kept.append(item)
    for new_id, item in enumerate(kept, start=1):
        item["id"] = new_id
    return kept, dropped


def apply_insertions(text: str, insertions: list[dict]) -> str:
    hallucinated = text
    for item in sorted(insertions, key=lambda row: row["char_offset"], reverse=True):
        start = item["char_offset"]
        original = item["violation_quote_original"]
        edited = item["violation_quote_edited"]
        if hallucinated[start : start + len(original)] != original:
            raise ValueError(f"Offset text mismatch for insertion {item['id']}")
        hallucinated = hallucinated[:start] + edited + hallucinated[start + len(original) :]
    return hallucinated


def verification(text: str, hallucinated: str, insertions: list[dict]) -> tuple[bool, list[str]]:
    failures = []
    for item in insertions:
        if hallucinated.count(item["violation_quote_edited"]) != 1:
            failures.append(f"edited quote count failed for {item['id']}")
        if item["violation_quote_original"] in hallucinated:
            failures.append(f"original quote survives for {item['id']}")
        if item["anchor_quote"] not in hallucinated:
            failures.append(f"anchor quote changed for {item['id']}")
    delta = abs(word_count(hallucinated) - word_count(text))
    if delta > 5 * len(insertions):
        failures.append(f"word-count delta {delta} exceeds allowed {5 * len(insertions)}")
    return not failures, failures


def process_story(story_dir: Path) -> dict:
    meta = load_meta(story_dir)
    scenes_payload = load_scenes(story_dir)
    scenes = scenes_payload["scenes"]
    text = (story_dir / "renamed.txt").read_text(encoding="utf-8")
    target = target_for(meta["length_bucket"], scenes_payload["total_scenes"])
    plan = plan_for(target)
    dropped = []
    specs = SPECS.get(story_dir.name, [])

    if target < 5:
        insertions = []
        dropped.append({"reason": f"too few scenes: target {target} is below 5"})
    else:
        insertions = []
        for idx, spec in enumerate(specs, start=1):
            try:
                insertions.append(make_insertion(idx, spec, text, scenes))
            except Exception as exc:  # keep the story moving and record failures
                dropped.append({"reason": f"spec dropped: {exc}"})
        insertions, validation_drops = filter_valid_insertions(insertions)
        dropped.extend(validation_drops)
        if len(insertions) < target:
            dropped.append(
                {
                    "reason": (
                        f"produced {len(insertions)} of target {target}; remaining slots left empty "
                        "because no additional distinct narration-only cascade-safe facts were curated"
                    )
                }
            )

    hallucinated = apply_insertions(text, insertions) if insertions else text
    (story_dir / "hallucinated.txt").write_text(hallucinated, encoding="utf-8")
    passed, verify_failures = verification(text, hallucinated, insertions)
    for failure in verify_failures:
        dropped.append({"reason": f"verification failed: {failure}"})

    payload = {
        "story_id": meta.get("document_id") or story_dir.name,
        "total_scenes": scenes_payload["total_scenes"],
        "target_insertions": target,
        "plan": plan,
        "insertions": insertions,
        "dropped": dropped,
    }
    (story_dir / "insertions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    difficulties = {key: 0 for key in ("easy", "medium", "hard")}
    severities = {key: 0 for key in ("minor", "moderate", "major")}
    distances = []
    for item in insertions:
        difficulties[item["difficulty"]] += 1
        severities[item["severity"]] += 1
        distances.append(item["scene_distance"])

    return {
        "length_bucket": meta["length_bucket"],
        "genre": meta["genre"],
        "wiki_title": meta["wiki_title"],
        "story_id": payload["story_id"],
        "total_scenes": scenes_payload["total_scenes"],
        "target": target,
        "insertions_produced": len(insertions),
        "difficulty_counts": ",".join(f"{k}:{v}" for k, v in difficulties.items()),
        "severity_counts": ",".join(f"{k}:{v}" for k, v in severities.items()),
        "mean_scene_distance": round(mean(distances), 2) if distances else 0,
        "verification_passed": passed,
        "dropped_count": len(dropped),
        "hallucinated_word_count": word_count(hallucinated),
        "renamed_word_count": word_count(text),
        "word_count_delta": word_count(hallucinated) - word_count(text),
        "path": str(story_dir),
    }


def main() -> None:
    rows = [process_story(story_dir) for story_dir in story_dirs()]
    fields = [
        "length_bucket",
        "genre",
        "wiki_title",
        "story_id",
        "total_scenes",
        "target",
        "insertions_produced",
        "difficulty_counts",
        "severity_counts",
        "mean_scene_distance",
        "verification_passed",
        "dropped_count",
        "renamed_word_count",
        "hallucinated_word_count",
        "word_count_delta",
        "path",
    ]
    with (ROOT / "continuity_insertions_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'story_id':<42} {'scenes':>6} {'target':>6} {'made':>5} "
        f"{'difficulty':<23} {'severity':<26} {'dist':>5} ok"
    )
    print("-" * 128)
    for row in rows:
        print(
            f"{row['story_id']:<42} {row['total_scenes']:>6} {row['target']:>6} "
            f"{row['insertions_produced']:>5} {row['difficulty_counts']:<23} "
            f"{row['severity_counts']:<26} {row['mean_scene_distance']:>5} "
            f"{row['verification_passed']}"
        )


if __name__ == "__main__":
    main()
