"""
Typing Prompts for Free Type Mode
=================================
Fixed 208-sentence prompt set:
  - first 52 prompts are kept unchanged (compatible with existing part 1-4 data)
  - plus 156 additional prompts for continued collection (part 5-16)
"""

import csv
import os
import re


PROMPTS_PER_PART = 13
TOTAL_PARTS = 16
EXPECTED_PROMPTS = PROMPTS_PER_PART * TOTAL_PARTS


PROMPTS = [
    # Legacy 52 (kept unchanged for compatibility)
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "sphinx of black quartz judge my vow",
    "i need to buy some groceries from the store today",
    "please send me the report before friday afternoon",
    "she walked quickly through the park and sat on a bench",
    "we should meet at the coffee shop around noon",
    "the weather forecast says it will rain tomorrow morning",
    "can you help me fix this problem with my computer",
    "he finished reading the entire book in just two days",
    "my favorite subject in school was always mathematics",
    "they decided to travel across europe during the summer",
    "the project deadline has been extended until next week",
    "my phone number is 8675309 please call me back",
    "the meeting is scheduled for 3 pm on march 14 2026",
    "room 207 is on the 2nd floor of building 5",
    "the password requires at least 8 characters and 2 digits",
    "flight 492 departs at 6 am from gate 17",
    "order number 30518 was shipped on january 9 2026",
    "the zip code for downtown is 10001",
    "add 250 grams of flour and 3 eggs to the mixture",
    "chapter 7 starts on page 143 of the textbook",
    "the final score was 96 to 84 in overtime",
    "apartment 4b is located at 123 elm street",
    "version 2 point 0 was released on october 15",
    "there are exactly 365 days in a regular year",
    "the library opens at 9 am and closes at 8 pm",
    "she scored 97 out of 100 on the final exam",
    "working from home has become much more common since 2020 and many people prefer it",
    "the restaurant on 5th avenue serves excellent pasta and their prices are very reasonable",
    "i just finished watching a 3 hour documentary about deep ocean exploration",
    "please update your profile with your new address before december 31",
    "the concert tickets cost 75 dollars each and we need 4 of them",
    "the quiz was extremely difficult but quite fair overall",
    "jazz musicians often explore complex rhythms and exotic scales",
    "the zookeeper examined the injured fox very carefully",
    "six juicy steaks sizzled in a pan while the chef relaxed",
    "our backup window starts at 2 am and ends at 4 am",
    "ticket 7812 was reopened after the nightly test failed",
    "please verify that user 42 can access report 9",
    "server 3 reached 92 percent cpu for 15 minutes",
    "we booked table 12 for 5 people at 7 pm",
    "batch 608 processed 144 files in 27 seconds",
    "meeting room 11 is reserved from 10 to 11",
    "the train leaves platform 6 at 8 35 tomorrow",
    "device 19 reported error code 503 twice today",
    "set reminder 30 minutes before the 6 pm call",
    "order 4409 includes 2 cables and 1 adapter",
    "my package arrived on april 3 at 9 20 am",
    "please archive log folder 7 before june 1",

    # New 156 (part 5-16, each part 13 prompts)
    "please confirm task 101 before 9 am meeting",
    "team alpha will review sprint 5 at 2 pm",
    "send draft 3 to manager 7 by noon",
    "project room 12 has 4 open seats today",
    "daily standup starts at 10 and ends at 10 15",
    "we logged 28 bugs and closed 19 this week",
    "client 24 requested invoice 8801 on monday",
    "backup job 6 failed at 1 30 and retried",
    "printer 2 needs toner after 640 pages",
    "office wifi 5g was stable for 8 hours",
    "the planner shows 11 tasks for day 3",
    "mark checkpoint 2 after reading chapter 14",
    "return keycard 9 to desk 4 after lunch",

    "open ticket 3201 and assign it to user 18",
    "api server 4 handled 920 requests in 5 minutes",
    "node 7 used 63 percent memory at 11 05",
    "cache hit rate reached 97 on run 2",
    "deploy build 41 to stage at 6 45 pm",
    "database shard 3 stored 240000 rows today",
    "rotate logs every 12 hours on host 9",
    "alert rule 15 triggers when cpu stays above 90",
    "session 88 expired after 30 idle minutes",
    "queue depth dropped from 44 to 6 overnight",
    "latency stayed near 23 ms during test 4",
    "service 10 restarted 2 times after patch 1",
    "monitor panel 5 shows green status for all pods",

    "train line 2 reaches station 16 at 7 20",
    "bus route 45 departs stop 9 every 30 minutes",
    "hotel room 308 includes 2 beds and desk 1",
    "gate 22 changed to gate 18 before boarding",
    "seat 14a was moved to row 17 seat 14",
    "taxi fare was 26 for a 12 km ride",
    "flight 730 arrived at 5 55 in terminal 3",
    "car rental desk closes at 11 on sunday",
    "map version 6 lists 9 nearby cafes",
    "platform 4 announcement repeated 3 times",
    "train ticket 5520 covers zones 1 2 and 3",
    "luggage tag 77 matched claim number 77",
    "shuttle 8 runs from lot 5 every 20 minutes",

    "student 12 solved 8 math problems before dinner",
    "chapter 15 includes 4 examples on linear models",
    "exam room 6 opens at 8 30 for section 2",
    "the class average was 84 on quiz 9",
    "read pages 50 to 63 and write summary 1",
    "team 3 built 2 prototypes in week 7",
    "lab group 5 measured 0 25 volts at node b",
    "history unit 4 covers years 1914 to 1918",
    "practice set 11 has 30 short questions",
    "submit homework 10 before 23 59 tonight",
    "score target is 95 after 14 study days",
    "library desk 2 issued book 34017",
    "course code cs101 meets on monday at 9",

    "runner 4 finished 5 km in 24 minutes",
    "team blue won game 3 by score 7 to 6",
    "coach planned 12 drills for day 5",
    "cycle session lasted 45 minutes at zone 2",
    "heart rate stayed near 132 during lap 8",
    "pool lane 1 opens again at 6 am",
    "the gym logged 210 checkins this month",
    "player 9 hit 3 goals in period 2",
    "set timer for 40 seconds and rest for 20",
    "workout plan 6 uses 4 rounds of squats",
    "track meet starts at 13 00 on saturday",
    "referee assigned court 7 to team 11",
    "medal count reached 18 after event 9",

    "grocery list has 2 apples 3 bananas and 1 melon",
    "receipt 9007 shows total 48 for week 2",
    "order 7781 shipped 6 items to box 14",
    "store aisle 5 has cereal pack size 12",
    "coupon 30 saved 7 on bill 105",
    "kitchen timer rang at 18 45 during step 4",
    "cook rice for 20 minutes then rest 5",
    "recipe 9 needs 250 ml milk and 2 eggs",
    "fridge shelf 3 holds 4 lunch boxes",
    "delivery window is 14 00 to 16 00 today",
    "market booth 8 sold 35 cups by noon",
    "payment id 4412 was approved in 3 seconds",
    "budget tracker shows 62 spent out of 90",

    "meeting note 1 says action item 7 is urgent",
    "draft version 5 includes 12 edited lines",
    "minute 18 of call 4 covers pricing plan",
    "speaker 2 shared 3 key points today",
    "report table 6 lists 45 entries for april",
    "calendar week 11 has 2 holidays",
    "follow up email 14 was sent at 21 10",
    "board room 3 booked from 9 to 10 30",
    "agenda section 4 needs 6 more comments",
    "survey form 27 received 108 replies",
    "contract page 19 has clause 8 revised",
    "status update 2 reached all 5 teams",
    "training video 13 runs for 17 minutes",

    "sensor kit 4 recorded value 73 at time 12",
    "sample batch 16 passed 9 quality checks",
    "control group 2 had 14 participants",
    "trial day 7 produced 3 stable signals",
    "model run 5 achieved error 0 08 on fold 1",
    "signal peak appears near index 128 in test 6",
    "dataset 3 contains 520 labeled windows",
    "window length is 39 with stride 13",
    "feature set 2 keeps 116 columns",
    "validation split uses ratio 8 to 2",
    "experiment 21 repeats each task 4 times",
    "notes file 10 was synced at 04 40",
    "result chart 9 highlights trend over 12 weeks",

    "system check 1 verifies camera mic and disk",
    "user 31 reset password 2 times this month",
    "policy version 7 became active on day 15",
    "audit log 44 flagged 5 unusual events",
    "security level 3 requires code 6 and badge 2",
    "vault door opens after step 1 step 2 step 3",
    "token 9021 expires in 14 days",
    "admin panel 8 shows 0 critical alerts",
    "access list has 26 names and 4 guests",
    "session key 77 rotated at 00 05",
    "firmware build 19 patched issue 302",
    "device group 4 completed scan in 11 minutes",
    "backup vault 2 stores archive 2025 11",

    "morning routine starts at 6 with task 1",
    "water plants in room 2 for 5 minutes",
    "clean desk drawer 3 before 8 pm",
    "laundry load 2 finishes in 47 minutes",
    "set alarm for 7 10 on day 4",
    "replace filter every 30 days in unit 1",
    "garage light 5 stayed on for 2 hours",
    "mailbox had 4 letters and 1 package today",
    "thermostat setpoint is 24 at night 22 by day",
    "window 6 was open during rain at 3 pm",
    "check battery level 9 on remote 2",
    "plant shelf 1 has 7 small pots",
    "home budget sheet 3 tracks bills 12 and 13",

    "code review 5 found 2 bugs in module 9",
    "branch 14 merged into main after test 7",
    "commit 88 added 36 lines and removed 4",
    "unit test 11 passed in 0 42 seconds",
    "ci job 3 ran 12 steps without failure",
    "python env 2 uses torch version 2 2",
    "script file 6 writes output to folder 3",
    "benchmark 9 reports 130 hz average rate",
    "parser rule 4 handles token 17 correctly",
    "feature flag 1 toggled on for group 5",
    "model epoch 20 reached accuracy 0 91",
    "log file 21 includes warning code 404",
    "release 2026 03 build 2 is now ready",

    "quick test sentence 1 mixes words and number 8",
    "zebra runs across yard 5 while fox waits",
    "every keyboard row appears in sample 12 today",
    "type all digits 0 1 2 3 4 5 6 7 8 9 once",
    "brown dog jumps 2 times then sleeps 7 minutes",
    "we expect coverage of a to z in run 16",
    "alpha beta gamma 3 delta 4 epsilon 5",
    "north zone 2 south zone 3 east 4 west 5",
    "final checklist has 10 items and 1 review",
    "participant 1 completes part 16 before 18 00",
    "round 3 dataset now includes group 5 to 16",
    "this prompt set adds 156 new sentences total",
    "training consistency stays unchanged after expansion",
]


def _validate_prompts(prompts) -> None:
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError(f"PROMPTS must be {EXPECTED_PROMPTS}, got {len(prompts)}")

    if len(set(prompts)) != len(prompts):
        raise ValueError("PROMPTS contains duplicates")

    pattern = re.compile(r"^[a-z0-9 ]+$")
    for p in prompts:
        if not pattern.fullmatch(p):
            raise ValueError(f"invalid characters in prompt: {p!r}")

    merged = " ".join(prompts)
    letters = {c for c in merged if c.isalpha()}
    digits = {c for c in merged if c.isdigit()}
    if letters != set("abcdefghijklmnopqrstuvwxyz"):
        missing = "".join(sorted(set("abcdefghijklmnopqrstuvwxyz") - letters))
        raise ValueError(f"missing letters: {missing}")
    if digits != set("0123456789"):
        missing = "".join(sorted(set("0123456789") - digits))
        raise ValueError(f"missing digits: {missing}")


_validate_prompts(PROMPTS)


def export_prompts_csv(path: str = "data/raw/free_type/free_type_prompt_dataset_208.csv") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_index", "prompt_text"])
        for i, prompt in enumerate(PROMPTS):
            w.writerow([i, prompt])
    return path


if __name__ == "__main__":
    out = export_prompts_csv()
    print(f"saved: {out}")
    print(f"count: {len(PROMPTS)}")
    print(f"parts: {TOTAL_PARTS} x {PROMPTS_PER_PART}")
