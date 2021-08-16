# Samples random probability vectors for use as player ranges.
# @classmod range_generator
import torch

from Source.Settings.arguments import arguments
from Source.Settings.game_settings import game_settings
from Source.Game.Evaluation.evaluator import Evaluator
from Source.Game.card_tools import CardTools


# Recursively samples a section of the range vector.
# @param cards an NxJ section of the range tensor, where N is the batch size
# and J is the length of the range sub-vector
# @param mass a vector of remaining probability mass for each batch member
# @see generate_range
# @local
class RangeGenerator:
    def _generate_recursion(self, cards, mass):
        batch_size = cards.size(0)
        assert mass.size(0) == batch_size
        # we terminate recursion at size of 1
        card_count = cards.size(1)
        if card_count == 1:
            mass.resize_as_(cards)
            cards.copy_(mass)
        else:
            rand = torch.rand(batch_size, dtype=arguments['dtype'], device=arguments['device'])

            mass1 = mass.clone().mul_(rand)
            mass1[torch.lt(mass1, 0.00001)] = 0
            mass1[torch.gt(mass1, 0.99999)] = 1

            mass2 = mass - mass1
            halfsize = card_count / 2
            # if the tensor contains an odd number of cards, randomize which way the middle card goes
            if halfsize % 1 != 0:
                halfsize = halfsize - 0.5
                halfsize = halfsize + torch.randint(2, (1,), dtype=torch.int64, device=arguments['device'])

            self._generate_recursion(cards[:, :int(halfsize)], mass1)
            self._generate_recursion(cards[:, int(halfsize):], mass2)

    # Samples a batch of ranges with hands sorted by strength on the board.
    # @param range a NxK tensor in which to store the sampled ranges, where N is
    # the number of ranges to sample and K is the range size
    # @see generate_range
    # @local

    # noinspection PyShadowingBuiltins
    def _generate_sorted_range(self, range):
        batch_size = range.size(0)
        self._generate_recursion(range,
                                 torch.empty(batch_size, dtype=arguments['dtype'], device=arguments['device']).fill_(1))

    # Sets the (possibly empty) board cards to sample ranges with.
    # The sampled ranges will assign 0 probability to any private hands that
    # share any cards with the board.
    # @param board a possibly empty vector of board cards
    def set_board(self, te, board):
        hand_strengths = torch.empty(game_settings['hand_count'], dtype=arguments['dtype'], device=arguments['device'])
        for i in range(0, game_settings['hand_count']):
            hand_strengths[i] = i

        if board.dim() == 0:
            hand_strengths = te.get_hand_strengths().squeeze_()
        elif board.size(0) == 5:
            hand_strengths = Evaluator().batch_eval(board)
        else:
            hand_strengths = te.get_hand_strengths().squeeze_()

        possible_hand_indexes = CardTools().get_possible_hand_indexes(board)
        self.possible_hands_count = possible_hand_indexes.sum(-1).item()
        self.possible_hands_mask = torch.empty((possible_hand_indexes.view(1, -1).size()), dtype=torch.bool,
                                               device=arguments['device']).copy_(possible_hand_indexes.view(1, -1))
        non_colliding_strengths = torch.empty(int(self.possible_hands_count), dtype=arguments['dtype'],
                                              device=arguments['device'])
        torch.masked_select(hand_strengths, self.possible_hands_mask, out=non_colliding_strengths)

        s1, order = non_colliding_strengths.sort()
        s2, self.reverse_order = order.sort()
        self.reverse_order = self.reverse_order.view(1, -1).long()
        self.reordered_range = torch.empty((), dtype=arguments['dtype'], device=arguments['device'])
        self.sorted_range = torch.empty((), dtype=arguments['dtype'], device=arguments['device'])

    # Samples a batch of random range vectors.
    # Each vector is sampled indepently by randomly splitting the probability
    # mass between the bottom half and the top half of the range, and then
    # recursing on the two halfs.
    # @{set_board} must be called first.
    # @param range a NxK tensor in which to store the sampled ranges, where N is
    # the number of ranges to sample and K is the range size

    # noinspection PyShadowingBuiltins
    def generate_range(self, range):
        batch_size = range.size(0)
        self.sorted_range.resize_(batch_size, int(self.possible_hands_count))
        self._generate_sorted_range(self.sorted_range)
        # we have to reorder the range back to undo the sort by strength
        index = torch.empty((self.reverse_order.expand_as(self.sorted_range).size()), dtype=torch.int64,
                            device=arguments['device']).copy_(self.reverse_order.expand_as(self.sorted_range))
        self.reordered_range = self.sorted_range.gather(1, index)

        range.zero_()
        range.masked_scatter_(self.possible_hands_mask.expand_as(range), self.reordered_range)
