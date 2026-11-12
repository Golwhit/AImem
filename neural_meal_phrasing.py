"""Hand-authored sentence frames for the authored meal teaching and held-out sets.

Frames 0-2 are optimizer input. Frame 3 is reserved for held-out changed wording and never trains.
"""

FRAMES = {
    "explicit_owner": (
        "{holder} has the {obj}. Who has the {obj}?",
        "The {obj} belongs to {holder}. Which person owns the {obj}?",
        "{holder} is holding the {obj} right now. Who is holding the {obj}?",
        "Look: the {obj} is in {holder}'s hands. Name whoever is holding the {obj}.",
    ),
    "attribute": (
        "{owner} has a {color} {obj}. What color is the {obj}?",
        "{owner} owns a {color} {obj}. Tell me the color of the {obj}.",
        "{owner}'s {obj} is {color}. Which color is the {obj}?",
        "Quick check: {owner} has a {color} {obj}, so what color is that {obj}?",
    ),
    "transfer": (
        ("{owner} has a {obj}. {owner} keeps the {obj}. Who has the {obj} now?",
         "{owner} has a {obj}. {owner} gives the {obj} to {other}. Who has the {obj} now?"),
        ("{owner} got a {obj} and still holds it. Who holds the {obj} at the end?",
         "{owner} got a {obj}, then hands it to {other}. Who holds the {obj} at the end?"),
        ("At first {owner} had the {obj}. Later {owner} kept it. Who has the {obj} today?",
         "At first {owner} had the {obj}. Later {owner} passed it to {other}. Who has the {obj} today?"),
        ("{owner} owns the {obj}, and still keeps it. Name the current holder.",
         "{owner} owns the {obj}, but gives it away to {other}. Name the current holder."),
    ),
    "negation": (
        "{owner} has a {obj}. The {obj} is not {opp}. It is {color}. Is the {obj} red?",
        "The {obj} that {owner} has is {color}, not {opp}. Is the {obj} red?",
        "{owner} has a {obj}. It is {color} and not {opp}. Would you call the {obj} red?",
        "Is red right for the {obj} that {owner} has? It is {color}, not {opp}.",
    ),
    "correction": (
        ("{owner}'s {obj} is {opp}.", "Correction: the {obj} is {color}. What color is it?"),
        ("I think {owner}'s {obj} is {opp}.", "Correction: it is {color}, not {opp}. What color is it?"),
        ("{owner} has a {opp} {obj}.", "No, the {obj} is {color}. Tell me its color."),
        ("{owner} says the {obj} is {opp}.", "Actually that is wrong: the {obj} is {color}. Which color is it?"),
    ),
    "missing_information": (
        "{owner} has a {attr}{obj}. What color is it?",
        "{owner} owns a {attr}{obj}. Tell me its color.",
        "{owner} has a {attr}{obj}. Which color is it?",
        "{owner} has a {attr}{obj}. Describe its color.",
    ),
    "reference": (
        "It is {color}. What color is the {obj}?",
        "It is {color}. Which color is the {obj}?",
        "It is {color}. Tell me its color.",
        "That one is {color}. What color is the {obj} I mentioned?",
    ),
    "unknown_location": (
        ("{owner} has a {obj}. The {obj} is {loc}. Where is the {obj}?",
         "{owner} has the {obj}. Where is the {obj}?"),
        ("{owner} owns a {obj} that is {loc}. Where is the {obj}?",
         "{owner} owns the {obj}. Where is the {obj}?"),
        ("The {obj} belongs to {owner} and sits {loc}. Where is it?",
         "The {obj} belongs to {owner}. Where is it?"),
        ("Tell me where the {obj} is for {owner}. The {obj} is {loc}.",
         "Tell me where the {obj} is for {owner}. {owner} owns the {obj}."),
    ),
}


def exchanges(owner: str, other: str, obj: str, case: int, split: str, *, wording: bool = False) -> list[dict]:
    """Build one paired exchange per skill and variant; ``wording`` selects the reserved held-out frame."""
    known = ("red", "blue")[case % 2]
    location = ("here", "there")[case % 2]
    rows = []
    for index, (skill, frames) in enumerate(FRAMES.items()):
        frame = 3 if wording else (case + index) % 3
        entry = frames[frame]
        pair = f"{split}-{case}-{skill}"
        for variant in (0, 1):
            color = ("red", "blue")[variant]
            opp = ("blue", "red")[variant]
            holder = (owner, other)[variant]
            turns = []
            if skill == "explicit_owner":
                content = entry.format(holder=holder, obj=obj)
                answer = holder + "."
            elif skill == "attribute":
                content = entry.format(owner=owner, color=color, obj=obj)
                answer = color.capitalize() + "."
            elif skill == "transfer":
                content = entry[variant].format(owner=owner, other=other, obj=obj)
                answer = (owner if variant == 0 else other) + "."
            elif skill == "negation":
                content = entry.format(owner=owner, obj=obj, color=color, opp=opp)
                answer = "Yes." if color == "red" else "No."
            elif skill == "correction":
                first = entry[0].format(owner=owner, obj=obj, opp=opp)
                turns = [{"role": "user", "content": first}, {"role": "assistant", "content": first}]
                content = entry[1].format(obj=obj, color=color, opp=opp)
                answer = color.capitalize() + "."
            elif skill == "missing_information":
                attr = known + " " if variant == 0 else ""
                content = entry.format(owner=owner, attr=attr, obj=obj)
                answer = known.capitalize() + "." if variant == 0 else "I do not know."
            elif skill == "reference":
                first = f"{owner} has a {obj}."
                turns = [{"role": "user", "content": first}, {"role": "assistant", "content": first}]
                content = entry.format(color=color, obj=obj)
                answer = color.capitalize() + "."
            else:
                content = entry[variant].format(owner=owner, obj=obj, loc=location)
                answer = location.capitalize() + "." if variant == 0 else "I do not know."
            rows.append({"id": f"{pair}-{variant}", "pair": pair, "skill": skill, "case": case,
                         "evaluation_group": split if split != "train" else None,
                         "turns": turns + [{"role": "user", "content": content}], "answer": answer})
    return rows
