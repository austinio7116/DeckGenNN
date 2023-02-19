import os
import re
import requests
import pickle
import tqdm

metadata_cache = {}

def get_card_metadata(name):
    if name in metadata_cache:
        return metadata_cache[name]

    response = requests.get(f"https://api.scryfall.com/cards/named?exact={name}")
    if response.status_code == 200:
        data = response.json()
        topdata = data
    else:
        response = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={name}")
        if response.status_code == 200:
            data = response.json()
            topdata = data
        else:
            print(f"Failed to retrieve card metadata for {name}")
            return {}
    if "card_faces" in data:
        for face in data["card_faces"]:
            if face["name"] == name:
                data = face
                break
    try:
        colors = data.get("colors", [])
        metadata = {
            "id": topdata["id"],
            "name": data["name"],
            "type_line": data["type_line"],
            "mana_cost": data.get("mana_cost", ""),
            "cmc": data.get("cmc", 0),
            "power": data.get("power", None),
            "toughness": data.get("toughness", None),
            "colors": colors,
        }
        metadata_cache[name] = metadata
        return metadata
    except KeyError:
        print(f"Failed to retrieve card metadata for {name}")
        return {}


def tokenize_decklist(decklist):
    main_deck = []
    deck = None

    for line in decklist.split("\n"):
        if line.startswith("[Main]"):
            deck = main_deck
        elif line.startswith("[Sideboard]"):
            break
        elif deck is not None:
            count_and_name = re.findall(r'^\d+\s+(.+)$', line)
            if count_and_name:
                count = int(line.split()[0])
                name = count_and_name[0]
                card = {
                    "count": count,
                    "name": name,
                    **get_card_metadata(name)
                }
                main_deck.extend([card] * count)
    return main_deck

def tokenize_decklists_in_folder(folder_path):
    decklists = {}
    files = os.listdir(folder_path)
    for file_name in tqdm.tqdm(files, desc="Progress"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                decklist = f.read()
                decklist = tokenize_decklist(decklist)
                if len(decklist) == 60:
                    decklists[file_name] = decklist
        except UnicodeDecodeError:
            print(f"Skipping file {file_path} due to encoding error.")
    #sort_deck_list(decklists)
    return decklists

def sort_deck_list(deck):
    types_order = {"Creature": 0, "Instant": 1, "Sorcery": 1, "Artifact": 2, "Enchantment": 3, "Planeswalker": 4, "Land": 5}

    def sort_key(card):
        type_line = card["type_line"].split("-")[0].strip()
        cost = card["cmc"]
        name = card["name"]
        return types_order[type_line], cost, name

    sorted_deck = sorted(deck, key=sort_key)
    deck.clear()
    creatures = []
    instants_sorceries = []
    enchantments = []
    planeswalkers = []
    lands = []
    for card in sorted_deck:
        type_line = card["type_line"].split("-")[0].strip()
        if type_line == "Creature":
            creatures.append(card)
        elif type_line in ("Instant", "Sorcery"):
            instants_sorceries.append(card)
        elif type_line == "Artifact":
            enchantments.append(card)
        elif type_line == "Enchantment":
            enchantments.append(card)
        elif type_line == "Planeswalker":
            planeswalkers.append(card)
        elif type_line == "Land":
            lands.append(card)
    deck.extend(creatures)
    deck.extend(instants_sorceries)
    deck.extend(enchantments)
    deck.extend(planeswalkers)
    deck.extend(lands)


if __name__ == '__main__':
    folder_path = "C:\\Users\\Mark\\PycharmProjects\\ChatGPTGenerator\\venv\\decklists"
    decklists = tokenize_decklists_in_folder(folder_path)
    data = []
    for file_name, (main_deck) in decklists.items():
        data.append(sort_deck_list(main_deck))

    # save the data to a file
    with open("decklists_data.pickle", "wb") as f:
        pickle.dump(data, f)
