import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from collections import Counter

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

class MagicGAN:
    def __init__(self, card_index_to_name,mana_index_to_mana,type_index_to_type, card_index_to_type, cardcount, metadata_size, noise_size=100, device=torch.device('cuda')):
        self.cardcount = cardcount
        self.metadata_size = metadata_size
        self.deck_size = cardcount*metadata_size
        self.card_index_to_name = card_index_to_name
        self.mana_index_to_mana = mana_index_to_mana
        self.type_index_to_type = type_index_to_type
        self.card_index_to_type = card_index_to_type
        self.device = device
        self.noise_size = noise_size
        print(self.deck_size)
        # Initialize the generator and discriminator networks
        self.generator = Generator(self.noise_size, self.cardcount, self.metadata_size).to(device)
        print(len(self.card_index_to_name))
        self.discriminator = Discriminator(self.cardcount, len(self.card_index_to_name)).to(device)

        # Initialize the optimizer for the generator and discriminator networks
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))

        # Initialize the loss function for the generator and discriminator
        self.loss = nn.BCELoss()


    def binarize(self, input):
        output = input
        output = output.reshape(-1, self.cardcount, self.metadata_size)
        name_indices = output[:, :, :len(self.card_index_to_name)].argmax(dim=-1)
        mana_indices = output[:, :, len(self.card_index_to_name):len(self.card_index_to_name) + len(self.mana_index_to_mana)].argmax(dim=-1)
        type_indices = output[:, :, len(self.card_index_to_name) + len(self.mana_index_to_mana):].argmax(dim=-1)
        output = torch.zeros_like(output)
        output.scatter_(2, name_indices.unsqueeze(-1), 1)
        output.scatter_(2, mana_indices.unsqueeze(-1) + len(self.card_index_to_name), 1)
        output.scatter_(2, type_indices.unsqueeze(-1) + len(self.card_index_to_name) + len(self.mana_index_to_mana), 1)
        return output

    def train(self, real_data, num_epochs, batch_size, save_interval=100):
        # Initialize lists to store loss values
        generator_losses = []
        discriminator_losses = []
        real_discriminator_losses = []
        fake_discriminator_losses = []
        fig = plt.figure(figsize=(10, 10))  # create a new figure with size 1024x1024


        # Convert the real data to a tensor and move it to the device
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Define the labels for the real and fake data
        j = 0
        for epoch in range(num_epochs):
            for i, real_batch in enumerate(dataloader):
                real_batch = real_batch[0].to(self.device)
                # Generate a batch of fake data
                noise = Variable(torch.FloatTensor(batch_size, self.noise_size).normal_(0, 1)).to(self.device)
                fake_data = self.generator(noise)
                #fake_data = self.binarize(fake_data)

                # Train the discriminator on the real and fake data
                self.discriminator_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                real_discriminator_output = self.discriminator(real_batch[:, :, :len(self.card_index_to_name)])
                fake_discriminator_output = self.discriminator(fake_data[:, :, :len(self.card_index_to_name)])
                real_discriminator_loss = self.loss(real_discriminator_output, real_labels)
                fake_discriminator_loss = self.loss(fake_discriminator_output, fake_labels)
                discriminator_loss = real_discriminator_loss + fake_discriminator_loss
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                # Train the generator by fooling the discriminator
                self.generator_optimizer.zero_grad()
                # Generate a batch of fake data
                noise2 = Variable(torch.FloatTensor(batch_size, self.noise_size).normal_(0, 1)).to(self.device)
                fake_data2 = self.generator(noise2)
                #fake_data2 = self.binarize(fake_data2)
                generator_labels = torch.ones(batch_size, 1, device=self.device)
                fake_discriminator_output2 = self.discriminator(fake_data2[:, :, :len(self.card_index_to_name)])
                generator_loss = self.loss(fake_discriminator_output2, generator_labels)
                # Compute excess non-basic land card penalty
                excess_nonbasic_count = self.compute_excess_nonbasic_cards(fake_data2)
                penalty_factor = 0.1*(epoch/num_epochs)/batch_size  # adjust this as needed
                generator_pen = penalty_factor * excess_nonbasic_count
                generator_loss += generator_pen
                generator_loss.backward()
                self.generator_optimizer.step()

                # Append loss values to the corresponding lists
                generator_losses.append(generator_loss.item())
                discriminator_losses.append(discriminator_loss.item())
                real_discriminator_losses.append(real_discriminator_loss.item())
                fake_discriminator_losses.append(fake_discriminator_loss.item())

                j = j+1
                # Print the loss and save the model weights if requested
                if j % 100 == 0:
                    #self.decode_deck_list(real_batch)
                    self.decode_deck_list(fake_data2)
                    print('[%d/%d][%d/%d] Discriminator Loss: %.4f (%.4f,%.4f) Generator Loss: %.4f (%4f)' % (
                        epoch + 1, num_epochs, i + 1, len(dataloader), discriminator_loss.item(),real_discriminator_loss.item()
                        ,fake_discriminator_loss.item(), generator_loss.item(), generator_pen))
                    # Plot the loss graph
                    plt.clf()
                    plt.plot(generator_losses, label='Generator Loss')
                    plt.plot(discriminator_losses, label='Discriminator Loss')
                    plt.plot(real_discriminator_losses, label='Real Discriminator Loss')
                    plt.plot(fake_discriminator_losses, label='Fake Discriminator Loss')
                    plt.legend()
                    #plt.ylim(0, 2)  # set y-axis limits
                    plt.draw()
                    plt.pause(0.001)
                if save_interval and j % save_interval == 0:
                    torch.save(self.generator.state_dict(), 'generator_%d_%d.pt' % (epoch + 1, i + 1))
                    torch.save(self.discriminator.state_dict(), 'discriminator_%d_%d.pt' % (epoch + 1, i + 1))
        torch.save(self.generator.state_dict(), 'generator_%d_%d.pt' % (epoch + 1, i + 1))
        torch.save(self.discriminator.state_dict(), 'discriminator_%d_%d.pt' % (epoch + 1, i + 1))

    def generate(self, num_decks):
        # Generate a batch of fake data
        noise = Variable(torch.FloatTensor(num_decks, self.noise_size).normal_(0, 1)).to(self.device)
        metadata = noise.to(self.device)
        fake_data = self.generator(metadata)
        return fake_data

    def compute_excess_nonbasic_cards(self, fake_data):
        excess_nonbasic_count = 0
        num_cards = 60  # Assuming each deck has 60 cards
        num_name_cols = len(self.card_index_to_name)  # Number of columns for card names

        name_tensor = fake_data[:, :num_cards, :num_name_cols]
        for deck in range(name_tensor.shape[0]):
            deck_tensor = name_tensor[deck]
            # Count the number of non-basic lands in each deck
            nonbasic_card_counts = torch.zeros((len(self.card_index_to_type),), dtype=torch.float32)
            deck_excess_nonbasic_count = 0
            for i in range(num_cards):
                card_index = name_tensor[deck, i, :].argmax(dim=0).item()
                nonbasic_card_counts[card_index] += 1

            # Compute the excess non-basic cards (cards with more than 4 copies)
            nonbasic_card_counts = nonbasic_card_counts.cpu().numpy()
            for i, type in self.card_index_to_type.items():
                if 'Basic Land' not in type and nonbasic_card_counts[i] > 4:
                    deck_excess_nonbasic_count += nonbasic_card_counts[i] - 4
            #self.decode_deck_list(deck_tensor.reshape(1, deck_tensor.shape[0], deck_tensor.shape[1]),
                                  #card_index_to_name, mana_index_to_mana, type_index_to_type)
            excess_nonbasic_count += deck_excess_nonbasic_count * deck_excess_nonbasic_count

        return excess_nonbasic_count


    def decode_deck_list(self, generated_data):
        deck_list = []
        num_cards = 60  # Assuming each deck has 60 cards
        num_name_cols = len(self.card_index_to_name)  # Number of columns for card names
        #num_mana_cols = len(self.mana_index_to_mana)
        #num_type_cols = len(self.type_index_to_type)

        # Split the tensor into the separate blocks
        name_tensor = generated_data[:, :num_cards, :num_name_cols]
        #mana_tensor = generated_data[:, :num_cards, num_name_cols:num_name_cols + num_mana_cols]
        #type_tensor = generated_data[:, :num_cards,
         #             num_name_cols + num_mana_cols:num_name_cols + num_mana_cols + num_type_cols]

        # Decode the card names
        for deck in range(name_tensor.shape[0]):
            deck_cards = []
            for i in range(num_cards):
                card_index = name_tensor[deck, i, :].argmax().item()
                card_name = self.card_index_to_name[card_index]
                deck_cards.append(card_name)
            deck_list.append(deck_cards)
            if deck >= name_tensor.shape[0]-2:
                print(self.format_decklist(deck_cards))
        return deck_list

    def print_deck_list(self, generated_data):
        deck_list = []
        num_cards = 60  # Assuming each deck has 60 cards
        num_name_cols = len(self.card_index_to_name)  # Number of columns for card names
        # Split the tensor into the separate blocks
        name_tensor = generated_data[:, :num_cards, :num_name_cols]
        # Decode the card names
        for deck in range(name_tensor.shape[0]):
            deck_cards = []
            for i in range(num_cards):
                card_index = name_tensor[deck, i, :].argmax().item()
                card_name = self.card_index_to_name[card_index]
                deck_cards.append(card_name)
            print(self.format_decklist(deck_cards))
        return



    def format_decklist(self, cards):
        # Count the number of occurrences of each card
        card_counts = Counter(cards)

        # Build the Arena decklist string
        decklist = ""
        for card, count in card_counts.items():
            decklist += f"{count} {card}\n"

        return decklist

    def load_generator(self, filename):
        self.generator.load_state_dict(torch.load(filename))
        self.generator.eval()

class Discriminator(nn.Module):
    def __init__(self, cardcount, metadata_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(cardcount*metadata_size, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.reshape(-1, self.num_flat_features(input))
        return self.main(input)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Generator(nn.Module):
    def __init__(self, input_size, cardcount, metadata_size):
        super(Generator, self).__init__()
        self.cardcount = cardcount
        self.metadata_size = metadata_size
        self.deck_size = cardcount * metadata_size
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5), # add dropout layer
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5), # add dropout layer
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5), # add dropout layer
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5), # add dropout layer
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5), # add dropout layer
            nn.Linear(1024, self.deck_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        output = output.reshape(-1, self.cardcount, self.metadata_size)
        return output



