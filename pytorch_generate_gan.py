import pickle

import pandas as pd
from collections import Counter
import torch
import requests

from MagicGAN import MagicGAN

metadata_cache = {}

def print_deck_list(generated_data, card_index_to_name):
    num_cards = 60  # Assuming each deck has 60 cards
    num_name_cols = len(card_index_to_name)  # Number of columns for card names
    # Convert the tensor to a numpy array
    generated_data = generated_data.cpu().detach().numpy()
    # Split the tensor into the separate blocks
    name_tensor = generated_data[:, :num_cards, :num_name_cols]
    # Decode the card names
    for deck in range(name_tensor.shape[0]):
        deck_cards = []
        for i in range(num_cards):
            card_index = name_tensor[deck, i, :].argmax().item()
            card_name = card_index_to_name[card_index]
            deck_cards.append(card_name)
        #print(format_decklist(deck_cards))
        print(preprocess_decklist(deck_cards))
    return



def get_card_type(card_name):
    # Call Scryfall API to get card information
    url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'type_line' in data:
            return data['type_line']
    return None

def format_decklist(cards):
    # Count the number of occurrences of each card
    card_counts = Counter(cards)

    # Build the decklist string
    decklist = ""
    for card, count in card_counts.items():
        decklist += f"{count} {card}\n"

    return decklist

def preprocess_decklist(decklist):
    # Group the decklist by card name and count the number of occurrences of each card
    card_counts = Counter(decklist)

    # Remove excess cards and count the number of cards that were removed
    total_excess_cards = 0
    total_excess_lands = 0
    for card_name, count in card_counts.items():
        card_type = get_card_type(card_name)
        if not card_type or 'Basic Land' in card_type:
            # Skip basic lands
            continue
        elif count > 4:
            if 'Land' in card_type:
                # Remove excess non-basic land cards
                total_excess_lands += count - 4
            else:
                # Remove excess non-basic land cards
                total_excess_cards += count - 4
            card_counts[card_name] = 4

    if total_excess_cards > 0:
        # Find cards to add to the deck
        for card_name, count in card_counts.items():
            card_type = get_card_type(card_name)
            if not card_type or 'Land' in card_type:
                # Skip lands
                continue
            elif count < 4 and total_excess_cards > 0:
                # Add cards of the same type until the maximum of 4 is reached
                cards_to_add = min(4-count, total_excess_cards)
                card_counts[card_name] += cards_to_add
                total_excess_cards -= cards_to_add
            if total_excess_cards == 0:
                break

    if total_excess_lands > 0:
        # Find cards to add to the deck
        for card_name, count in card_counts.items():
            card_type = get_card_type(card_name)
            if not card_type or 'Land' not in card_type:
                # Skip non lands
                continue
            elif count < 4 and total_excess_lands > 0:
                # Add cards of the same type until the maximum of 4 is reached
                cards_to_add = min(4 - count, total_excess_lands)
                card_counts[card_name] += cards_to_add
                total_excess_lands -= cards_to_add
            if total_excess_lands == 0:
                break

    num_cards_needed = total_excess_lands + total_excess_cards
    # If more cards are needed, add basic lands
    if num_cards_needed > 0:
        basic_land_types = ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']
        # Count the number of basic lands already in the deck
        num_basic_lands = sum(card_counts.get(land_type, 0) for land_type in basic_land_types)
        if num_basic_lands == 0:
            # If there are no basic lands in the deck, add the remaining lands evenly across all land types
            for card_type in card_counts.keys():
                if card_type in basic_land_types:
                    card_counts[card_type] += num_cards_needed // len(card_counts)
                    num_land_to_add = num_cards_needed // num_basic_lands
                    num_cards_needed -= num_land_to_add
        else:
            # If there are basic lands in the deck, add the remaining lands evenly across existing basic lands
            for land_type in basic_land_types:
                if num_cards_needed > 0:
                    num_land = card_counts.get(land_type, 0)
                    if num_land > 0:
                        # Distribute the remaining lands evenly across existing basic lands of this type
                        num_land_to_add = num_cards_needed // num_basic_lands
                        num_cards_needed -= num_land_to_add
                        card_counts[land_type] += num_land_to_add
                    if num_cards_needed == 0:
                        break

    # Build the decklist string
    decklist_str = format_decklist(card_counts)

    return decklist_str



def format_decklist(cards):
    # Count the number of occurrences of each card
    card_counts = Counter(cards)

    # Build the Arena decklist string
    decklist = ""
    for card, count in card_counts.items():
        decklist += f"{count} {card}\n"

    return decklist


def get_deck_tensor(deck_list_df, card_encodings, mana_encodings, type_encodings):

    # One-hot encode the mana costs using the mana_encodings dictionary
    card_one_hot = torch.zeros((deck_list_df.shape[0], len(card_encodings)))
    for i, id in enumerate(deck_list_df['name']):
        card_one_hot[i][card_encodings[id]] = 1

    # One-hot encode the mana costs using the mana_encodings dictionary
    mana_costs_one_hot = torch.zeros((deck_list_df.shape[0], len(mana_encodings)))
    for i, mana_cost in enumerate(deck_list_df['mana_cost']):
        mana_costs_one_hot[i][mana_encodings[mana_cost]] = 1

    # One-hot encode the card types using the type_encodings dictionary
    types_one_hot = torch.zeros((deck_list_df.shape[0], len(type_encodings)))
    for i, card_type in enumerate(deck_list_df['type_line']):
        types_one_hot[i][type_encodings[card_type]] = 1

    # Concatenate the one-hot encoded vectors into a single tensor
    deck_tensor = torch.cat((card_one_hot, mana_costs_one_hot, types_one_hot), dim=1)

    # Pad or truncate the tensor to the correct size (60 x (n+m+c))
    pad_size = (60, len(card_encodings) + len(mana_encodings) + len(type_encodings))
    if deck_tensor.shape[0] < 60:
        deck_tensor = torch.cat((deck_tensor, torch.zeros(pad_size[0] - deck_tensor.shape[0], pad_size[1])), dim=0)
    elif deck_tensor.shape[0] > 60:
        deck_tensor = deck_tensor[:60, :]

    return deck_tensor

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


if __name__ == '__main__':
    with open('decklists_data.pickle', 'rb') as f:
        decklists = pickle.load(f)

    # Tokenize the decklists and create a Pandas DataFrame
    data = []
    for i, decklist in enumerate(decklists):
        for card in decklist:
            data.append({
                'deck_id': i,
                'name': card['name'],
                'count': card['count'],
                'mana_cost': card['mana_cost'],
                'type_line': card['type_line'],
            })
    deck_list_df = pd.DataFrame(data)
    unique_names = deck_list_df['name'].unique()
    unique_mana_costs = deck_list_df['mana_cost'].unique()
    unique_card_types = deck_list_df['type_line'].unique()

    # Create encoding dictionaries
    card_encodings = {id: i for i, id in enumerate(unique_names)}
    mana_encodings = {mana_cost: i for i, mana_cost in enumerate(unique_mana_costs)}
    type_encodings = {card_type: i for i, card_type in enumerate(unique_card_types)}

    card_index_to_name = {v: k for k, v in card_encodings.items()}
    mana_index_to_mana = {v: k for k, v in mana_encodings.items()}
    type_index_to_type = {v: k for k, v in type_encodings.items()}
    card_index_to_type = {}

    # Group the decklists data by deck_id and get the deck tensors for each deck
    deck_groups = deck_list_df.groupby('deck_id')
    deck_tensors = [get_deck_tensor(deck, card_encodings, mana_encodings, type_encodings) for _, deck in deck_groups]

    deck_tensors = torch.stack(deck_tensors)
    gan = MagicGAN(card_index_to_name,mana_index_to_mana,type_index_to_type, card_index_to_type,cardcount=deck_tensors[0].shape[0],metadata_size=deck_tensors[0].shape[1],noise_size=100
                   ,device=torch.device('cuda'))
    gan.load_generator('generator_62_53.pt')

    print_deck_list(gan.generate(32),card_index_to_name)
