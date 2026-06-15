# Annotation Guidelines: REV4RE Classification Scheme

## Overview

This document provides the labeling rules for annotating app reviews into three classes: **Explicit Requirements (label 1)**, **Implicit Requirements (label 2)**, and **Irrelevant Reviews (label 0)**. These rules were derived through a grounded theory analysis and are operationalized through 14 requirement characteristics.

---

## Explicit Requirements (Label: 1)

Reviews that directly express software-related needs, such as desired features or functionality issues.

| ID | Characteristic | Contextual Cues |
|----|---------------|-----------------|
| ER1 | Specific app features not working as intended | Feature names + "not working", "broken", "fails to", "keeps crashing" |
| ER2 | Complaints about bugs, glitches, or functional issues | "buggy", "glitchy", "unusable", repeated failure descriptions |
| ER3 | Requests for help with app features | "how do I…", "please help", "I can't figure out…" |
| ER4 | Missing features users wish to be added | "I wish", "It would be great if", "Please add…" |

**Examples:**
- *"The app's search bar is not working properly. It keeps suggesting irrelevant products."* → ER1
- *"I can't believe how buggy this app is. The checkout process keeps crashing."* → ER2
- *"I'm having trouble understanding how to use the app's filters."* → ER3
- *"I wish this app had a feature that allowed me to save my favorite products."* → ER4

---

## Implicit Requirements (Label: 2)

Reviews where users do not directly express software-related needs but imply them through experiences, narratives, or complaints.

| ID | Characteristic | Contextual Cues |
|----|---------------|-----------------|
| MR1 | Opinions on overall app performance | "takes too long", "slow to load", general performance observations |
| MR2 | Non-functional complaints (battery, heating) | "battery drain", "phone heats up", hardware-related impacts |
| MR3 | Storytelling-style feature suggestions | "I always forget…", "It would be nice if…", narrative accounts |
| MR4 | Competitor comparisons | "unlike other apps", "some of the others let me…", naming rivals |
| MR5 | UI/UX navigation difficulties | "hard to find", "confusing layout", "keep getting lost" |
| MR6 | Device/platform incompatibility | "not supported", "doesn't work on [device]", "not compatible" |

**Examples:**
- *"I've noticed that it's been taking longer and longer to load recent transactions."* → MR1
- *"It's been draining my battery like crazy lately."* → MR2
- *"It would be great to set reminders for my bills. I always forget to pay them on time."* → MR3
- *"I'm frustrated with the lack of features compared to some of the other apps out there."* → MR4
- *"I've been trying to book a flight, but it's been really difficult to navigate."* → MR5
- *"I love this app, but it's not compatible with my new tablet."* → MR6

---

## Irrelevant Reviews (Label: 0)

Reviews focused on service-related topics unrelated to app functionality.

| ID | Characteristic | Contextual Cues |
|----|---------------|-----------------|
| IR1 | Customer support feedback | "support team", "customer service", "helpful staff" |
| IR2 | Pricing or value-for-money | "too expensive", "worth it", "cheaper alternatives" |
| IR3 | Community or social features | "community", "group", "chat with others" |
| IR4 | Content or media quality | "great movie selection", "outdated videos", "missing albums" |

**Examples:**
- *"The customer support team was really helpful when I had a question about my order."* → IR1
- *"I think the app is overpriced for what it offers."* → IR2
- *"I love the community aspect of this app."* → IR3
- *"The app has a great selection of movies and TV shows."* → IR4

---

## Important Notes

- These are **not keyword-based rules**. Identifying requirements often depends on nuanced language, user intent, and narrative context.
- When in doubt between explicit and implicit, consider: does the user **directly state** a need (explicit) or does the need have to be **inferred** from their narrative (implicit)?
- When in doubt between implicit and irrelevant, consider: does the review relate to **app functionality** (implicit) or purely to **services delivered through the app** (irrelevant)?
- Reviews with poor grammar, spelling errors, or colloquial language should still be classified based on their semantic content.
