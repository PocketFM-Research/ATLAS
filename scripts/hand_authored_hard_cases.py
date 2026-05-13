#!/usr/bin/env python3
"""Hand-authored hard cross-scene repetition cases.

Used when the Gemini key is unavailable. The case TEXTS in this file were
written by a language model (Claude) directly, reasoning about beat structure
per the same tier prompts in `generate_hard_cross_scene_ground_truth.py`.

Running this script produces the same outputs as the Gemini generator:
  scene_text_exports/the_fighter_cross_scene_hard.txt
  scene_text_exports/intolerable_cruelty_cross_scene_hard.txt
  scene_text_exports/inserted_error_ground_truth_cross_scene_hard.json

It also computes Jaccard against the source scene and prints any case that
exceeds its tier target so you can spot-check quality.

Counts: 8 cases per movie, 16 total
  paraphrase_full=3  beat_only=3  partial_overlap=1  motif_distractor=1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SCENE_HEADING_RE = re.compile(r"^\s*(\d+)\s*[\).:：、-]\s*(.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
STOPWORDS = {
    "a","about","again","all","also","an","and","any","are","as","at","be",
    "been","before","but","by","could","did","didn","do","does","don","even",
    "for","from","had","has","have","he","her","him","his","how","i","if",
    "in","into","is","it","its","just","like","maybe","no","not","of","on",
    "or","s","she","so","still","t","that","the","there","they","this",
    "though","to","too","ve","was","wasn","what","when","where","while",
    "why","with","would","you","your","we","our","us","out","up","down",
    "off","over","under","back","him","her","me","my","im","ive","were",
    "are","yeah","im","ya","gonna","gotta","wanna",
}

TIER_INFO = {
    "paraphrase_full": dict(
        max_jaccard=0.30, is_repetition=True, expected_difficulty="medium",
        subtype="paraphrase_whole_scene",
        inserted_change=(
            "Whole-scene paraphrase: same characters, sequence of events, and "
            "outcome as the source; rewritten with no shared multi-word spans; "
            "setting changed."
        ),
    ),
    "beat_only": dict(
        max_jaccard=0.18, is_repetition=True, expected_difficulty="hard",
        subtype="dramatic_beat_replay",
        inserted_change=(
            "Beat-level replay: same dramatic function (setup/conflict/tactic/"
            "resolution) as the source; different characters, location, and "
            "surface content. Tests beat-structure detection independent of "
            "lexical or entity overlap."
        ),
    ),
    "partial_overlap": dict(
        max_jaccard=0.22, is_repetition=True, expected_difficulty="hard",
        subtype="partial_paraphrase_back_half",
        inserted_change=(
            "Partial replay: scene opens on unrelated new material then "
            "paraphrases the source's climactic beat in its back half. Tests "
            "whether the detector localizes a repeated beat inside a longer "
            "scene."
        ),
    ),
    "motif_distractor": dict(
        max_jaccard=0.40, is_repetition=False, expected_difficulty="hard_negative",
        subtype="justified_callback",
        inserted_change=(
            "HARD NEGATIVE: scene reuses one motif (line / object / gesture) "
            "from the source as a narratively justified callback, but the "
            "plot, conflict, and outcome differ. is_repetition=false. Tests "
            "precision against recurring motifs."
        ),
    ),
}


# ---------- CASE LIBRARY -----------------------------------------------------
# Each entry: (movie_key, source_scene_number, tier, heading_after_number, body)
# `heading_after_number` is everything after the leading "<num>、" so the
# assembler can attach a new sequential number at write time.

CASES: List[Tuple[str, int, str, str, str]] = [
    # ================= THE FIGHTER (8) =================

    ("the_fighter", 1, "paraphrase_full",
     "EXT. JIMMY'S BACKYARD - LATE AFTERNOON",
     """Micky stands by the back gate, helmet under one arm, gloves hanging
from his fingers. Inside the yard there's a small birthday cookout going.
JIMMY notices him first, sets down a tray of burgers, walks over slow.
JIMMY
(low)
You shoulda called, bud.
MICKY
I know.
LAURIE comes around the side of the house in flip-flops, sees Micky, stops
cold.
LAURIE
You can't keep showin' up on her good days, Micky.
MICKY
Today's the only day I had.
LAURIE
Then make a bettah schedule. We've been over this.
JIMMY
(to Laurie)
He just wants to see her.
LAURIE
And you stay outta it. Every time you stay outta it I'm the bad one.
KASIE peeks around her mother's hip, sees her dad. She bolts toward him
before Laurie can grab her arm. Hugs him at the gate, gloves and all.
KASIE
You gonna fight him? The boxah from
the news?
MICKY
That's the plan, sweethaht.
LAURIE
Don't promise her anything. You hear me?
MICKY
I'm gonna do better, Kase. I'm gonna get us that place.
LAURIE
(pulls Kasie back)
Kasie, go inside, ya cake's meltin'.
She walks her daughter back across the lawn without looking at Micky again.
Jimmy raises a hand, palm out, sorry. Closes the gate softly. The barbecue
smoke drifts over.
KASIE
(across the yard)
Bye Daddy!
MICKY
(to the gate)
Happy birthday, baby."""),

    ("the_fighter", 5, "paraphrase_full",
     "EXT. ROUTE 38 DINER PARKING LOT - NIGHT",
     """The neon sign buzzes. Micky and Charlene stand by his Tempo, paper
bag of leftovers in her hand. He's pretending to look for change in his
jacket. They haven't spoken since the booth.
CHARLENE
You said two things the whole meal.
MICKY
I'm thinkin'.
CHARLENE
About what.
He won't look up. Fishes for keys he can feel are already in his pocket.
CHARLENE
You drove us forty minutes outta Lowell for meatloaf, Micky. Tell me
what's actually goin' on or just take me home.
MICKY
(quiet)
I don't wanna run into anybody right now.
CHARLENE
Anybody who.
MICKY
Anybody.
She watches him a long beat.
CHARLENE
You told the whole town what you were gonna do. And you didn't do it.
MICKY
- I told Kasie I was gonna get her a room.
CHARLENE
And the fight you took -- everyone in your family pushed you into it.
MICKY
Don't.
CHARLENE
Don't what. I'm sayin' what you told me last week. They put you in
there alone.
MICKY
(finally looks at her)
That's my family you're talkin' about.
CHARLENE
What's left to say, Micky. Look at your face. Look at me.
He stares at her. Something in him breaks open. He pulls her against the
car and kisses her -- the kiss gets longer, more sure, until the bag of
leftovers slips out of her hand onto the pavement and neither of them
picks it up."""),

    ("the_fighter", 17, "paraphrase_full",
     "INT. SAINT MARY'S HIGH SCHOOL GYM - EVENING (RENTED)",
     """Everyone's there. Charlene at the apron. Alice and George in folding
chairs. Little Dicky on a bench. O'Keefe in the ring holding pads. Lights
hum. Floor squeaks. Dicky steps in through the side door, gym bag over a
shoulder, looking lean for once. The room stops.
ALICE
What.
DICKY
Micky said I can't work with him no more.
ALICE
That's not right. Eight months sober and you don't get the corner?
O'KEEFE
Cut the shit, Alice.
DICKY
Watch your mouth around my mother.
ALICE
(to Micky)
Then YOU tell him. You tell O'Keefe you'da won Sanchez without your
brother.
Everyone turns. Micky stands in the middle of the floor. Long beat.
MICKY
- I wouldn'a won Sanchez without Dicky.
CHARLENE
You're not serious.
MICKY
I went in with our plan. It wasn't workin'. I switched to what Dicky
taught me. That's just true.
(to O'Keefe)
I wouldn'a won it without you neither, Mick. You know that.
CHARLENE
You got your focus from him! And from your father, and from me!
ALICE
Lucky shot. Dicky pulled him through.
CHARLENE
WE pulled him through. Dicky was on a pipe.
DICKY
I'm eight months out, Charlene.
CHARLENE
And you'll be lookin' for somethin' for your back by November.
MICKY
ENOUGH. I'm the one in the ring. Not you. Not you. Not YOU.
CHARLENE
So you want him back.
MICKY
I want him. I want you. I want O'Keefe. I want all-a you.
CHARLENE
You can't have that. You sound like them now.
MICKY
Maybe.
CHARLENE
Maybe you ARE them.
She walks. O'Keefe follows her. The door bangs. Alice puts a hand on
Micky's shoulder, eyes bright.
ALICE
Everything happens for a reason, honey.
Micky pulls headgear on, climbs through the ropes. Dicky climbs in across
from him. They circle. Dicky snaps Micky's head with a jab. Micky comes
back with a body shot. Then another. Dicky tries to clinch -- Micky
traps the left elbow, twists, drops a straight right behind Dicky's ear.
Dicky goes down on the canvas. Stays down. Blood from his nose. Alice on
her feet.
ALICE
What're you doin', Micky! What is the MATTER with you!
MICKY
Can it just be MY fight, Ma? One time? Just one. Not for Dicky."""),

    ("the_fighter", 2, "beat_only",
     "EXT. ST. PATRICK'S RECTORY DRIVEWAY - SATURDAY MORNING",
     """Two cars and a pickup are parked along the rectory drive. A wedding
party in tuxedos and gowns moves boxes of programs and a portable speaker
into the trunks. The bride, ELLIE, 26, is helping her grandmother into the
back seat of the Buick. Across the lawn, her father, BERNARD, paces with a
phone to his ear, jaw set.
AUNT MAUREEN
Ellie. Ellie, sweetheart. We have to leave.
ELLIE
Five more minutes.
AUNT MAUREEN
The Monsignor said eleven sharp.
ELLIE
I know what he said.
Bernard hangs up. Shakes his head at his sister.
BERNARD
He's at McGonagle's. Off Lawrence Street.
AUNT MAUREEN
At ELEVEN in the morning. On your daughter's wedding day.
ELLIE
He's my brother.
AUNT MAUREEN
Honey, listen to me. I been goin' to meetings. We don't owe him this.
You walk yourself down. He gave you that toast last Christmas. It's
enough.
ELLIE
He didn't give me a toast last Christmas. I asked him to. I begged him.
He stood up and he cried and he told the whole table he was sorry and
he meant it.
BERNARD
That was last Christmas, kid.
ELLIE
He's been my best man since I was nine years old.
AUNT MAUREEN
You're already doin' it without him, Ellie.
ELLIE
(quiet, looking at the rectory)
I can't.
She slips off her heels, hands them to a bridesmaid, and starts walking
toward Bernard's pickup. Bernard sighs and tosses her the keys."""),

    ("the_fighter", 7, "beat_only",
     "INT. PARISH HALL BINGO ROOM - WEDNESDAY EVENING",
     """Folding tables. Twenty regulars, most of them past sixty. A
chalkboard on a stand at the front. RAY, 50s, in a sport coat that doesn't
quite close, stands by it with a clipboard. His sponsor BERNIE leans on
the wall behind him with his arms crossed.
RAY
- and so each one of you gives me twenty up front, and you get the seat
in the system, and after four nights the system pays back forty. To
EACH of ya. That's a hundred percent.
MRS. PAVLIK
You givin' all twenty of us forty dollars?
RAY
Not all twenty. Just the first ten who get in.
BERNIE
First ten lucky ones.
WALTER
So I pay twenty and I get nothing?
RAY
No no no. You bring in TWO new players. They put in twenty each. You
get forty back. Then they bring two each. EVERYBODY wins.
MRS. PAVLIK
This is the same as that Tupperware mess.
RAY
This is NOT Tupperware. This is a SYSTEM.
WALTER
Yeah and where'd you hear about the system, Ray.
RAY
I -- I read it. I've been readin' a lot.
BERNIE
He's been readin' a lot.
RAY
(arm around Mrs. Pavlik's shoulder)
I LOVE you guys.
The whole table starts complaining at once. Ray laughs, raises his hands,
turns to the chalkboard and underlines the number. The hall door opens at
the back and his daughter, ROSE, eight, peeks in carrying his car keys.
Ray sees her, his face changes for a half-second, then he turns back to
the chalkboard and keeps pitching."""),

    ("the_fighter", 12, "beat_only",
     "INT. HARRIS & WEST CONFERENCE ROOM - DOWNTOWN BOSTON - EVENING",
     """A glass-walled conference room, half-eaten Thai takeout in
clamshells. JANE HARRIS, 40, the founder, at the head of the table. Her
COO PRIYA on her right. Three associates around it. A laptop is open to a
deck called Q4 RAISE.
JANE
- if we close the lead by the end of the week we still have time to get
the term sheet out before the holiday freeze. Priya, you wanted to talk
about a backup.
PRIYA
Two backups. I want to talk about two backups.
JANE
Where's my brother in this. Tom said he'd be here.
PRIYA
(checks her watch)
He said seven.
ASSOCIATE
Have him call in.
At the back of the room a SECOND ASSOCIATE comes in fast, holding her
phone face out so the table can see it. Local news app, push notification.
SECOND ASSOCIATE
(to Jane)
Jane.
JANE
What.
SECOND ASSOCIATE
He drove into a Sweetgreen on Tremont. Witness said he got out and
started yelling at the menu board.
Three associates immediately stand up. The COO doesn't move. Jane doesn't
move. The other three are already at the elevator before anyone has said
anything.
JANE
(to Priya)
Don't.
PRIYA
I wasn't going to.
JANE
Last time you said we have to STOP showin' up.
PRIYA
And I still mean it.
Jane stands. Picks up her bag.
PRIYA
Jane.
JANE
I have to."""),

    ("the_fighter", 14, "partial_overlap",
     "INT. ST. JEROME'S BASEMENT - AA MEETING - NIGHT",
     """Folding chairs in a half circle. Bad fluorescent light. A coffee urn
on a table behind the chairs sweating onto a paper plate of cookies.
MICKEY O'KEEFE sits in the second row, hands folded, leg bouncing. The
share goes around. A WOMAN finishes her turn and the room claps quietly.
DAVE, leader, looks at O'Keefe.
DAVE
Mick. You wanna?
O'KEEFE
- Pass tonight. Just listening.
DAVE
Sure.
O'KEEFE
Actually -- yeah. I'll go.
He stands. Walks to the front. Takes the chip out of his pocket, holds
it up.
O'KEEFE
Two yeahs Tuesday. Most-a you know me. Most-a you know I've been
trainin' Micky Ward. Most-a you know his brother's the reason I started
comin' here. So when I see his brother on the TV doin' what he's doin' I
gotta sit on my hands. Because the worst thing in the world for me right
now is to feel like the one who got it right. I'm not the one who got it
right. I'm the one who's still gettin' it right tonight. So that's --
yeah. That's all. Thanks.
He sits. The room claps quietly.
DAVE
Thanks, Mick.
O'Keefe pulls a beat-up flip phone from his coat. Turns it over in his
hand. Doesn't open it.

INT. MICKY'S APARTMENT ABOVE GARAGE - LATER THAT NIGHT
Micky's on the kitchen phone, the long curly cord twisted around his
wrist. HBO on in the next room. He's watching it through the doorway,
sound low, his brother on the screen sliding down a wall.
MICKY
(into phone)
Ma. I know. I know. But he did it himself.
ALICE (O.S., on the phone)
What was he THINKIN', Micky? They set him up. They wanted this.
MICKY
It's a documentary, Ma. They didn't set anybody up. He did it.
ALICE (O.S.)
Maybe they got him the drugs --
There's a knock at the door. Micky freezes. Knock again. He carries the
phone toward the door, cord stretching, opens it. CHARLENE on the porch
in a coat she's clearly thrown on over a t-shirt. He hasn't seen her in
weeks.
MICKY
(into phone, eyes on her)
Ma. It's bad. I gotta call you back.
He hangs up before she answers. He and Charlene look at each other in
the doorway, the TV light blue behind him. She makes a small sound that
could be a laugh or could be a sob. They step into each other. He cries
into her hair. She walks him backward into the apartment, pushes the
door shut with her foot. They go down on the couch."""),

    ("the_fighter", 10, "motif_distractor",
     "EXT. RIVERSIDE COMMUTER LOT - LATE NIGHT",
     """The lot's almost empty. A beat-up Volvo wagon sits alone at the far
end with its lights off. The wind moves the chain link.
Mickey O'Keefe pulls in, parks two spaces over, kills his engine. He
sits there a long time, key in the ignition, hands on the wheel. Coffee
in the cupholder gone cold.
He looks over at the Volvo. Nobody in it. Bumper sticker peeling, GO
LOWELL HIGH RIVERHAWKS.
He opens his notebook. Reads a page. Closes it.
O'KEEFE
(to himself, quiet)
One day."""),

    # ================= INTOLERABLE CRUELTY (8) =================

    ("intolerable_cruelty", 2, "paraphrase_full",
     "INT. THE JONATHAN CLUB - PRIVATE DINING ALCOVE - EVENING",
     """A waiter clears a plate. Miles, in a different but equally
expensive suit, finishes a glass of Bordeaux. Across from him, three
JUNIOR ATTORNEYS lean in.
MILES
The trouble with the institution of marriage is that everyone arrives
willing to settle. That is what undoes them. One side scores. The other
impeaches. The result is an arithmetic mean defined by whichever
attorney is the less embarrassed about asking for what he wants. The
public calls it compromise. We call it surrender.
ATTORNEY ONE
What do we call winning, sir.
MILES
The complete and humiliating destruction of the opposing party.
The waiter approaches and bends to his ear.
WAITER
Your six o'clock has arrived.
MILES
(to the attorneys)
Read the file on Boyle. We will resume Thursday. In the meantime --
think on this. Sherman. Genghis. Bonaparte. What do they have in
common.
ATTORNEY THREE
They lost a final battle?
MILES
That is not the answer I was looking for.
He stands, buttons his jacket. Across the dining room he sees REX
RAXROTH already standing, looking lost between two ferns. Miles
crosses, offers his hand.
MILES
Mr. Rexroth.
REX
Rex.
MILES
Miles Massey. Sit. Whatever you order tonight is on the firm. Consider
this table your office. Your sanctuary. Your situation room.
He pours Rex the second half of the bottle.
MILES
Now. Tell me what's wrong.
REX
(laughing weakly)
Jesus. Where do I start.
Miles offers an encouraging smile, leans his chin on a steepled hand.
REX
She's got me by the -- well. She's got me.
MILES
That is her function. Don't take it personally.
REX
When we met, we were -- crazy about each other. Not in like a feelings
way. We just couldn't keep our hands off.
MILES
Mm.
REX
Then it. Cooled.
MILES
Ardor cools. The market opens up.
REX
That's it. That's exactly it. There's so much MORE of it out there
than there used to be. You know what I mean by "it."
MILES
Rex. I am an attorney. I do not require diagrams."""),

    ("intolerable_cruelty", 4, "paraphrase_full",
     "INT. PILATES REFORMER STUDIO - WEST HOLLYWOOD - MORNING",
     """Eight women on reformers, an instructor in a headset pacing the
line. Spring tension snaps and resets. Marylin and Sarah are side by
side at the back, working their footbars.
MARYLIN
I don't know what his game is anymore. He turned down every single
proposal Ruth put on the table.
SARAH
Every one?
MARYLIN
Every one. And we were not unreasonable, Sarah. Ruth was generous to
the point of embarrassing.
SARAH
What does he WANT.
MARYLIN
Ruth couldn't tell. She kept her face on but I could see she was
thrown.
SARAH
He's supposed to be a wolf.
A NEW STUDENT a few reformers up loses her grip, the bar snaps back,
springs ping. She bursts into tears. Two instructors converge on her
with bottled water and the studio's empathy face. Marylin and Sarah make
faces, then assemble concern when an older client glares at them.
MARYLIN
(under)
Every Tuesday.
SARAH
I cannot.
The new student is escorted to the lobby.
MARYLIN
Even REX is confused, Sarah. He told me on the phone he doesn't
understand his own lawyer. If I didn't know better I'd say Massey was
running this personally, like I'd offended him in some past life.
SARAH
And so what now.
MARYLIN
If he doesn't move by Friday, court.
SARAH
Oh god.
MARYLIN
Court.
The instructor blows a whistle. They reset the springs."""),

    ("intolerable_cruelty", 18, "paraphrase_full",
     "INT. BEVERLY HILTON BUNGALOW - 3 A.M.",
     """The TV is on, Court TV, sound low. Miles is on top of the covers in
a shirt, no tie, no shoes, an old issue of Architectural Digest open on
his chest.
ON THE TV
A WITNESS being walked through a confession.
PROSECUTOR (TV)
And he then asked you to do what, exactly.
WITNESS (TV)
Tuh fahnd somebody tuh do his wife.
PROSECUTOR (TV)
And by "do" we mean...
WITNESS (TV)
(points)
Yeah. THAT one.
A clap of distant thunder over the hills.
DREAM
A long pale corridor. Files stacked taller than Miles. From inside the
stacks, a rasp:
RASPING VOICE
Two thousand fifty billable hours. Fifteen hundred motions to vacate.
Eight hundred summary judgments. A hundred and thirty thousand --
A clammy hand, IV taped to its wrist, comes out of the dark and points.
Miles is holding something heavy, a sawn-off shotgun maybe. He fires.
BONNIE falls. He fires again. MRS. GUTTMAN. He turns -- MARYLIN. He
hesitates.
RASPING VOICE
Counseluh? Counseluh?
A phone is ringing somewhere. He turns it on HERB and -- the ring is
louder now -- the dream comes apart and he is on top of a hotel
bedspread in the dark holding nothing, sweating through his shirt. The
nightstand phone is going.
MILES
(picking up)
Hello.
MARYLIN
Miles?
MILES
Marylin?
MARYLIN
You were right about me. I am hollow. I am hollow and I always was.
MILES
Marylin. When did I say that.
MARYLIN
I don't blame Rex. I don't blame Howard. I don't blame my mother. I
don't blame any of them."""),

    ("intolerable_cruelty", 3, "beat_only",
     "EXT. RIVIERA COUNTRY CLUB - FAIRWAY OF THE 4TH - MORNING",
     """A foursome moves up the fairway in two carts. STAN, 60, drives the
lead cart with VICTOR. ART and TED follow in the back cart. They've all
made money in different and approximately equivalent ways and are dressed
to make sure each other knows it.
ART
(over the wind)
So who'd you finally go with.
STAN
Dieter Lansky.
VICTOR
Lansky. Out of Century City?
STAN
Out of nowhere now. He works from a house in Brentwood. Did the
Sinclair deal.
VICTOR
The Sinclair deal was him?
TED
He did a Sinclair?
STAN
He did a Hartwell, too. Did all of them.
ART
Who's Sinclair's guy.
STAN
That'd be Don Massey. They go way back.
VICTOR
By reputation only. He's the guy who got Brian Holcombe that compound
on St. John.
ART
Brian Holcombe was so impressed with him he kept Don for the second
divorce.
TED
Who'd he end up married to.
ART
The first one or the second one?
TED
Either one.
ART
The second one runs a Best Western near Bakersfield.
The cart slows at Victor's ball. Victor gets out, hefts a five iron,
considers the green.
VICTOR
You shoulda gotten remarried, Stan. Lock yourself in.
STAN
I'm not gonna get married for tax purposes.
VICTOR
That's why everybody does it.
STAN
I'm not everybody.
ART
Famous last words."""),

    ("intolerable_cruelty", 12, "beat_only",
     "EXT. CONVENTION CENTER ROOFTOP - DUSK (TECH CONFERENCE)",
     """A few hundred folding chairs facing a small stage. A banner reads
LAUNCH LA -- COMMENCEMENT 2002. The graduating cohort sit in matching
t-shirts under the lights. KAI MEREDITH, 28, founder of the accelerator,
stands at the lectern with a wireless mic. In the back row, REGINA, his
former cofounder, sits next to her assistant DANI. Regina has a glass of
champagne and a face like she's swallowed a battery.
KAI
Founders. A startup -- if you think about it -- is a marriage. The
Series A is the wedding. The seed round is when you decide to move in
together. The friends-and-family round is when you tell your mom about
him. And the bridge round, when you do it, is when you're sleeping in
separate rooms but still trying.
Restlessness in the third row. Someone coughs.
KAI
The product is a child. The team is the family that raises the child.
And the exit -- the exit is the moment your child goes off to college,
and you and your cofounder look at each other across the kitchen and
you realize you have nothing to talk about anymore.
DANI
(to Regina)
Is he --
REGINA
Yes.
DANI
Did he --
REGINA
He always does.
KAI
So when I ask you tonight to choose a cofounder, I am asking you to
choose a SPOUSE. Choose someone whose Series B you can survive. Choose
someone whose pivot you can forgive. Choose someone whose --
DANI
You want a mint.
REGINA
I want him to fall off the roof.
DANI
(opening her clutch)
That seems extreme.
KAI
And so. With the authority vested in me by the Launch LA charter -- I
declare you. Launched.
Applause. Champagne. The graduating cohort gets to their feet. Regina
does not stand."""),

    ("intolerable_cruelty", 26, "beat_only",
     "INT. RUSTIC RIDGE WINERY - TASTING ROOM - SATURDAY AFTERNOON",
     """A long marble bar. Wine glasses lined up in fours. NINA, 34,
recently promoted to VP, is with her two best friends LEX and CARMEN. The
SOMMELIER pours. There is a cheese board. There is a bill, building.
CARMEN
You said yes, didn't you.
NINA
I said yes.
She holds the glass to the light. Sniffs.
NINA
Is that the Pinot or the Gamay.
LEX
That's the Gamay.
CARMEN
Is Gamay heavier than Pinot.
NINA
I think Pinot is heavier.
SOMMELIER
The Gamay's lighter. Brighter. We just got the magnums.
NINA
Okay. I'll take the case.
He sets it aside next to four other bottles, an artisan honey, two
chocolate bars priced like watches, and a wooden cutting board.
NINA
(to her friends)
I cannot do this anymore. Let's get an actual meal.
LEX
Aren't we supposed to stop at the goat dairy?
NINA
Right.
SOMMELIER
And on the card, ma'am?
She slides out her new corporate Amex.
SOMMELIER
(scanning it)
Very good, Ms. Albright.
He vanishes. Nina absently runs a finger along a cork from a flight she
didn't actually finish.
NINA
He said to "make the team mine."
CARMEN
If he only knew.
NINA
He's -- he's not what I expected. He's so -- earnest.
LEX
You're still doing it though.
NINA
I'm doing it.
CARMEN
He has no idea you've already taken meetings with Eos.
NINA
None.
LEX
You're going to break him.
NINA
I'm not going to break him. I'm going to outgrow him."""),

    ("intolerable_cruelty", 42, "partial_overlap",
     "INT. RAMADA INN COFFEE SHOP - VENTURA HIGHWAY - LATE NIGHT",
     """A coffee shop attached to a freeway motel. One waitress wiping the
counter. JOE, 50s, the same hitman, sits in the corner booth with a piece
of pie he hasn't started. A cell phone next to his plate, off.
WAITRESS
You good, sweetie?
JOE
I'm good.
She moves on. He takes a small black notebook out of his coat and turns
to a page near the back.
JOE
(reading, quiet)
Stockton. Encino. Hayward. Encino. Encino.
He closes the notebook. Sets a fork on his slice. Doesn't lift the fork.
A FAMILY of three comes in, the kid maybe eight, fighting sleep in his
mother's arms. Joe watches them order. The mother pays cash. They leave
with three styrofoam cups. Joe looks at his cooling pie.
He turns the cell phone on. Waits for it to find a signal. A single bar.
He dials.

INT. MASSEY MANSION - BEDROOM - SAME NIGHT (LATER)
Marylin is in the doorway of the bedroom when the landline rings on the
nightstand. She walks over, lifts the receiver.
MARYLIN
Hello.
INTERCUT - MILES IN A SPEEDING CAR
MILES
Marylin. Marylin, listen.
MARYLIN
Miles. Where have you BEEN. I've been trying you for --
MILES
You have to get out of the house. Right now. Pack nothing.
MARYLIN
I will. Miles, I will, but I have to tell you --
MILES
Out. Now.
MARYLIN
Miles. I'm sorry. About the agreement. About -- the original plan
was --
MILES
I know what it was. Get OUT.
MARYLIN
I fell in love. I didn't plan to.
MILES
I did too. Now pack a bag and --
MARYLIN
You did?
END INTERCUT
She hangs up the phone. She walks slowly to the dresser, lifts a small
framed photo of Miles, looks at it. Pans with her back into frame --
JOE, against the wall, near the door. She doesn't quite jump.
MARYLIN
Whoever paid you, I'll pay twice.
JOE
A Mr. Dumbarton.
She shows him the photo.
MARYLIN
Is this Mr. Dumbarton?
JOE
That's his lawyer.
MARYLIN
Three times.
JOE
Who's the pigeon.
Distant tires shriek in the driveway."""),

    ("intolerable_cruelty", 42, "motif_distractor",
     "INT. MASSEY MEYERSON - WRIGLEY'S OFFICE - MORNING",
     """A wall of framed plaques. Three FIRST-YEARS sit on the leather
sofa across from WRIGLEY'S desk, holding their orientation folders. On
the corner of the desk, prominent, is the SAME framed picture of Miles
that lived on Marylin's mantelpiece -- studio portrait, charcoal suit,
slightly aspirational lighting.
WRIGLEY
You are going to spend the next three weeks pretending to listen to me.
That's fine. Most of what I'm gonna say isn't load-bearing. Here is the
load-bearing part.
He turns the framed picture so it faces the sofa.
WRIGLEY
This is Miles Massey. Mr. Massey is on sabbatical. Mr. Massey will be
back when Mr. Massey is back. Until that day, all references to "Miles
says" or "Miles thinks" or "Miles wants" should be considered
historical, not operative.
FIRST-YEAR ONE
Is he -- is he okay.
WRIGLEY
Mr. Massey is fine. Mr. Massey is in Nevis. The state of Mr. Massey's
wellness is not currently the subject of this orientation.
FIRST-YEAR TWO
We were told in the lobby --
WRIGLEY
You were told a number of things in the lobby. None of them are firm
policy.
He puts the picture back where it was. Turns to the whiteboard.
WRIGLEY
Right. Billable hour conventions. Page one of your folder. Stay with
me."""),
]


# ---------- helpers ----------------------------------------------------------


def content_tokens(text: str) -> set:
    return {
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_scenes(text: str) -> Dict[int, Tuple[str, str]]:
    scenes: Dict[int, Tuple[str, str]] = {}
    current_num: Optional[int] = None
    current_heading: str = ""
    current_lines: List[str] = []

    def flush():
        if current_num is not None and current_lines:
            body = "\n".join(current_lines).strip()
            scenes[current_num] = (current_heading, body)

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = SCENE_HEADING_RE.match(line)
        if m:
            flush()
            current_num = int(m.group(1))
            current_heading = line.strip()
            current_lines = []
            continue
        if current_num is not None:
            current_lines.append(raw_line)

    flush()
    return scenes


def assemble(movie_key: str, clean_text: str,
             cases_for_movie: List[Tuple[int, str, str, str]]
             ) -> Tuple[str, List[Dict]]:
    scenes = parse_scenes(clean_text)
    if not scenes:
        raise SystemExit(f"No scenes parsed for {movie_key}.")
    next_num = max(scenes) + 1
    appended_scenes: List[str] = []
    gold: List[Dict] = []

    for idx, (source_num, tier, heading_tail, body) in enumerate(
            cases_for_movie, start=1):
        if source_num not in scenes:
            raise SystemExit(
                f"{movie_key}: source scene {source_num} not in clean text."
            )
        src_heading, src_body = scenes[source_num]
        heading = f"{next_num}、{heading_tail}"
        appended_scenes.append(f"{heading}\n{body.strip()}")

        info = TIER_INFO[tier]
        actual_j = jaccard(src_body, body)
        gold.append({
            "id": f"HXR{idx:03d}",
            "category": "cross_scene_repetition",
            "subtype": info["subtype"],
            "tier": tier,
            "is_repetition": info["is_repetition"],
            "source_scene": source_num,
            "repeated_scene": next_num,
            "source_heading": src_heading,
            "repeated_heading": heading,
            "target_max_jaccard": info["max_jaccard"],
            "actual_jaccard": round(actual_j, 4),
            "expected_difficulty": info["expected_difficulty"],
            "inserted_change": info["inserted_change"],
            "repeated_body_preview": body.strip()[:300],
        })
        next_num += 1

    out_text = clean_text.rstrip() + "\n\n\n" + "\n\n\n".join(appended_scenes) + "\n"
    return out_text, gold


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene_text_dir", type=Path,
                   default=Path("scene_text_exports"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("scene_text_exports"))
    p.add_argument("--gold_out_name", type=str,
                   default="inserted_error_ground_truth_cross_scene_hard.json")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # group cases by movie
    by_movie: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for movie_key, source_num, tier, heading_tail, body in CASES:
        by_movie.setdefault(movie_key, []).append(
            (source_num, tier, heading_tail, body)
        )

    gold_all: Dict[str, List[Dict]] = {}
    for movie_key, cases in by_movie.items():
        clean_path = args.scene_text_dir / f"{movie_key}_clean_scene_text.txt"
        if not clean_path.exists():
            print(f"missing: {clean_path}", file=sys.stderr)
            continue
        clean_text = clean_path.read_text(encoding="utf-8")
        out_text, gold = assemble(movie_key, clean_text, cases)
        out_path = args.out_dir / f"{movie_key}_cross_scene_hard.txt"
        out_path.write_text(out_text, encoding="utf-8")
        gold_all[movie_key] = gold

        # quality report
        by_tier: Dict[str, List[float]] = {}
        violations = []
        for g in gold:
            by_tier.setdefault(g["tier"], []).append(g["actual_jaccard"])
            if g["is_repetition"] and g["actual_jaccard"] > g["target_max_jaccard"]:
                violations.append(g)
        print(f"\n{movie_key}: wrote {out_path}")
        for tier, vals in by_tier.items():
            print(
                f"  {tier:18s}  n={len(vals)}  "
                f"jaccard mean={sum(vals)/len(vals):.3f}  "
                f"max={max(vals):.3f}"
            )
        if violations:
            print(f"  WARNING: {len(violations)} case(s) above target jaccard:")
            for v in violations:
                print(
                    f"    {v['id']} tier={v['tier']} "
                    f"actual={v['actual_jaccard']:.3f} "
                    f"target<={v['target_max_jaccard']:.3f}"
                )

    gold_path = args.out_dir / args.gold_out_name
    gold_path.write_text(
        json.dumps(gold_all, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote ground truth: {gold_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
