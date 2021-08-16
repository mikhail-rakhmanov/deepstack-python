# Samples random card combinations.
# @module random_card_generator
import torch

from Source.Settings.arguments import arguments
from Source.Settings.game_settings import game_settings

# Samples a random set of cards.
# Each subset of the deck of the correct size is sampled with
# uniform probability.
# @param count the number of cards to sample
# @return a vector of cards, represented numerically


def card_generator(count):
    # marking all used cards
    used_cards = torch.empty(game_settings['card_count'], dtype=torch.uint8, device=arguments['device']).zero_()

    out = torch.empty(count, dtype=arguments['dtype'], device=arguments['device'])
    # counter for generated cards
    generated_cards_count = []
    while len(generated_cards_count) != count:
        card = torch.randint(game_settings['card_count'], (1,), dtype=torch.int64, device=arguments['device'])
        if used_cards[card] == 0:
            out[len(generated_cards_count)] = card
            used_cards[card] = 1
            generated_cards_count.append(1)
    return out
