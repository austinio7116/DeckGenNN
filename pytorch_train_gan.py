import pickle
import torch
import pandas as pd

from MagicGAN import MagicGAN


def create_encodings(deck_list_df):
    # Get the unique Scryfall IDs, mana costs, and card types in the deck list
    unique_names = deck_list_df['name'].unique()
    unique_mana_costs = deck_list_df['mana_cost'].unique()
    unique_card_types = deck_list_df['type_line'].unique()

    # Create encoding dictionaries
    card_encodings = {name: i for i, name in enumerate(unique_names)}
    mana_encodings = {mana_cost: i for i, mana_cost in enumerate(unique_mana_costs)}
    type_encodings = {card_type: i for i, card_type in enumerate(unique_card_types)}

    return card_encodings, mana_encodings, type_encodings

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


def train_gan(deck_list_df):
    # Get the unique Scryfall IDs, mana costs, and card types across all decks
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
    card_index_to_type = {card_encodings[k]: deck_list_df.loc[deck_list_df['name'] == k, 'type_line'].values[0] for k in
                          card_encodings.keys()}

    # Group the decklists data by deck_id and get the deck tensors for each deck
    deck_groups = deck_list_df.groupby('deck_id')
    deck_tensors = [get_deck_tensor(deck, card_encodings, mana_encodings, type_encodings) for _, deck in deck_groups]

    deck_tensors = torch.stack(deck_tensors)
    # Initialize the MagicGAN
    gan = MagicGAN(card_index_to_name,mana_index_to_mana,type_index_to_type, card_index_to_type,cardcount=deck_tensors[0].shape[0],metadata_size=deck_tensors[0].shape[1],noise_size=100
                   ,device=torch.device('cuda'))

    # Train the MagicGAN on the deck tensors
    gan.train(deck_tensors, num_epochs=100, batch_size=32, save_interval=10000)


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

    train_gan(deck_list_df)
