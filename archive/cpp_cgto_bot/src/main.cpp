#include <skeleton/actions.h>
#include <skeleton/constants.h>
#include <skeleton/runner.h>
#include <skeleton/states.h>
#include <bitmask_tables.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <map>
#include <time.h>

using namespace pokerbots::skeleton;

struct HandRank {
  std::vector<int> rankTuple;  // The lexicographically comparable tuple
  int category;  // 0-8 representing hand category

  HandRank(std::vector<int> tuple, int cat) : rankTuple(std::move(tuple)), category(cat) {}

  // Comparison operators for lexicographic comparison
  bool operator>(const HandRank& other) const {
    // Python compares tuples lexicographically: compare elements first
    size_t minSize = std::min(rankTuple.size(), other.rankTuple.size());
    for (size_t i = 0; i < minSize; ++i) {
      if (rankTuple[i] != other.rankTuple[i]) {
        return rankTuple[i] > other.rankTuple[i];
      }
    }
    // If all compared elements are equal, longer tuple wins (in Python, shorter tuple is less)
    return rankTuple.size() > other.rankTuple.size();
  }

  bool operator==(const HandRank& other) const {
    if (rankTuple.size() != other.rankTuple.size()) {
      return false;
    }
    for (size_t i = 0; i < rankTuple.size(); ++i) {
      if (rankTuple[i] != other.rankTuple[i]) {
        return false;
      }
    }
    return true;
  }
};

struct Bot {
  static constexpr int MC_ITERATIONS = 100;
  static constexpr const char* SUITS = "hdcs";
  
  std::vector<std::string> REMAINING_DECK;
  std::random_device rd;
  std::mt19937 gen;

  Bot() : gen(rd()) {
  }

  // Helper to count bits (C++20 has std::popcount, but for compatibility we'll use this)
  int bitCount(unsigned int x) const {
    int count = 0;
    while (x) {
      count += x & 1;
      x >>= 1;
    }
    return count;
  }

  // Get suit index: h=0, d=1, c=2, s=3
  int getSuitIndex(char suit) const {
    switch (suit) {
      case 'h': return 0;
      case 'd': return 1;
      case 'c': return 2;
      case 's': return 3;
      default: return 0;
    }
  }

  HandRank bestHandRank8(const std::vector<std::string>& cards) const {
    std::array<int, 13> rankCounts = {0};
    std::array<int, 4> suitMasks = {0};
    int rankMask = 0;

    // Build masks
    for (const auto& card : cards) {
      char rank = card[0];
      char suit = card[1];
      int rankIdx = RANK_TO_INDEX.at(rank);
      int suitIdx = getSuitIndex(suit);

      rankCounts[rankIdx]++;
      suitMasks[suitIdx] |= (1 << rankIdx);
      rankMask |= (1 << rankIdx);
    }

    // ---------- 1. STRAIGHT FLUSH ----------
    for (int suit = 0; suit < 4; ++suit) {
      int sm = suitMasks[suit];
      if (bitCount(sm) >= 5) {
        for (const auto& [mask, high] : STRAIGHT_MASKS) {
          if ((sm & mask) == mask) {
            return HandRank({8, high}, 8);
          }
        }
      }
    }

    // ---------- 2. FOUR OF A KIND ----------
    int quad = -1;
    int kicker = -1;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] == 4) {
        quad = r;
      } else if (rankCounts[r] > 0 && kicker == -1) {
        kicker = r;
      }
    }
    if (quad != -1) {
      return HandRank({7, quad, kicker}, 7);
    }

    // ---------- 3. FULL HOUSE ----------
    std::vector<int> trips;
    std::vector<int> pairs;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] >= 3) {
        trips.push_back(r);
      } else if (rankCounts[r] >= 2) {
        pairs.push_back(r);
      }
    }
    if (!trips.empty()) {
      int trip = trips[0];
      int pair = (trips.size() > 1) ? trips[1] : (pairs.empty() ? -1 : pairs[0]);
      if (pair != -1) {
        return HandRank({6, trip, pair}, 6);
      }
    }

    // ---------- 4. FLUSH ----------
    for (int suit = 0; suit < 4; ++suit) {
      int sm = suitMasks[suit];
      if (bitCount(sm) >= 5) {
        std::vector<int> kickers;
        for (int r = 12; r >= 0; --r) {
          if (sm & (1 << r)) {
            kickers.push_back(r);
          }
        }
        std::vector<int> rankTuple = {5};
        for (size_t i = 0; i < std::min(kickers.size(), size_t(5)); ++i) {
          rankTuple.push_back(kickers[i]);
        }
        return HandRank(rankTuple, 5);
      }
    }

    // ---------- 5. STRAIGHT ----------
    for (const auto& [mask, high] : STRAIGHT_MASKS) {
      if ((rankMask & mask) == mask) {
        return HandRank({4, high}, 4);
      }
    }

    // ---------- 6. THREE OF A KIND ----------
    if (!trips.empty()) {
      int trip = trips[0];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      std::vector<int> rankTuple = {3, trip};
      for (size_t i = 0; i < std::min(kickers.size(), size_t(2)); ++i) {
        rankTuple.push_back(kickers[i]);
      }
      return HandRank(rankTuple, 3);
    }

    // ---------- 7. TWO PAIR ----------
    if (pairs.size() >= 2) {
      int highPair = pairs[0];
      int lowPair = pairs[1];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      int kicker = kickers.empty() ? -1 : *std::max_element(kickers.begin(), kickers.end());
      return HandRank({2, highPair, lowPair, kicker}, 2);
    }

    // ---------- 8. ONE PAIR ----------
    if (!pairs.empty()) {
      int pair = pairs[0];
      std::vector<int> kickers;
      for (int r = 12; r >= 0; --r) {
        if (rankCounts[r] == 1) {
          kickers.push_back(r);
        }
      }
      std::vector<int> rankTuple = {1, pair};
      for (size_t i = 0; i < std::min(kickers.size(), size_t(3)); ++i) {
        rankTuple.push_back(kickers[i]);
      }
      return HandRank(rankTuple, 1);
    }

    // ---------- 9. HIGH CARD ----------
    std::vector<int> kickers;
    for (int r = 12; r >= 0; --r) {
      if (rankCounts[r] > 0) {
        kickers.push_back(r);
      }
    }
    std::vector<int> rankTuple = {0};
    for (size_t i = 0; i < std::min(kickers.size(), size_t(5)); ++i) {
      rankTuple.push_back(kickers[i]);
    }
    return HandRank(rankTuple, 0);
  }

  // Compare hands: returns {win_value (1, 0.5, or 0), opponent_rank_category}
  std::pair<double, int> compareHands(const std::vector<std::string>& p1Cards,
                                      const std::vector<std::string>& p2Cards) const {
    HandRank p1Rank = bestHandRank8(p1Cards);
    HandRank p2Rank = bestHandRank8(p2Cards);

    double value;
    if (p1Rank > p2Rank) {
      value = 1.0;
    } else if (p1Rank == p2Rank) {
      value = 0.5;
    } else {
      value = 0.0;
    }
    return {value, p2Rank.category};
  }

  // Monte Carlo simulation for discard selection
  std::pair<double, int> mcOnce(const std::vector<std::string>& myCards,
                                 const std::vector<std::string>& boardCards,
                                 int discardIdx) {
    std::vector<std::string> newBoard = boardCards;
    std::vector<std::string> keptCards = myCards;

    if (discardIdx != -1 && discardIdx < (int)myCards.size()) {
      keptCards.clear();
      for (size_t i = 0; i < myCards.size(); ++i) {
        if (i != (size_t)discardIdx) {
          keptCards.push_back(myCards[i]);
        }
      }
      newBoard = boardCards;
      newBoard.push_back(myCards[discardIdx]);
    }

    // Use remaining deck
    std::vector<std::string> deck = REMAINING_DECK;
    std::shuffle(deck.begin(), deck.end(), gen);

    std::vector<std::string> oppCards;
    if (deck.size() >= 2) {
      oppCards = {deck[0], deck[1]};
    }

    int needed = 6 - (int)newBoard.size();
    std::vector<std::string> futureBoard;
    if (deck.size() >= 2 + needed) {
      futureBoard = std::vector<std::string>(deck.begin() + 2, deck.begin() + 2 + needed);
    }

    std::vector<std::string> myHand = keptCards;
    myHand.insert(myHand.end(), newBoard.begin(), newBoard.end());
    myHand.insert(myHand.end(), futureBoard.begin(), futureBoard.end());

    std::vector<std::string> oppHand = oppCards;
    oppHand.insert(oppHand.end(), newBoard.begin(), newBoard.end());
    oppHand.insert(oppHand.end(), futureBoard.begin(), futureBoard.end());

    return compareHands(myHand, oppHand);
  }

  int chooseDiscardMc(const std::vector<std::string>& myCards,
                      const std::vector<std::string>& boardCards) {
    std::array<double, 3> wins = {0, 0, 0};

    for (int i = 0; i < 3; ++i) {
      for (int iter = 0; iter < MC_ITERATIONS; ++iter) {
        auto [value, rank] = mcOnce(myCards, boardCards, i);
        wins[i] += value;
      }
    }

    int bestIdx = 0;
    for (int i = 1; i < 3; ++i) {
      if (wins[i] > wins[bestIdx]) {
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  double calcWinningProb(const std::vector<std::string>& myCards,
                        const std::vector<std::string>& boardCards,
                        int street) {
    double wins = 0;
    int total = 0;

    for (int iter = 0; iter < MC_ITERATIONS; ++iter) {
      auto [value, rank] = mcOnce(myCards, boardCards, -1);
      if (rank != 0 || street <= 3) {
        wins += value;
        total += 1;
      }
    }

    total = (street <= 3) ? MC_ITERATIONS : total;
    return wins / MC_ITERATIONS;
  }

  /*
    Called when a new round starts. Called NUM_ROUNDS times.
  */
  void handleNewRound(GameInfoPtr gameState, RoundStatePtr roundState, int active) {
    // Reset deck for new round
    REMAINING_DECK.clear();
    for (const char* r = RANKS; *r; ++r) {
      for (const char* s = SUITS; *s; ++s) {
        std::string card;
        card += *r;
        card += *s;
        REMAINING_DECK.push_back(card);
      }
    }
  }

  /*
    Called when a round ends. Called NUM_ROUNDS times.
  */
  void handleRoundOver(GameInfoPtr gameState, TerminalStatePtr terminalState, int active) {
    // Nothing needed here
  }

  /*
    Where the magic happens - your code should implement this function.
  */
  Action getAction(GameInfoPtr gameState, RoundStatePtr roundState, int active) {
    auto legalActions = roundState->legalActions();
    int street = roundState->street;
    auto myCards = roundState->hands[active];
    auto boardCards = roundState->board;
    int myPip = roundState->pips[active];
    int oppPip = roundState->pips[1 - active];
    int myStack = roundState->stacks[active];
    int oppStack = roundState->stacks[1 - active];
    int continueCost = oppPip - myPip;
    int myContribution = STARTING_STACK - myStack;
    int oppContribution = STARTING_STACK - oppStack;

    // Deck management - remove known cards
    if (street == 0) {
      // Pre-flop: remove my cards
      REMAINING_DECK.clear();
      for (const char* r = RANKS; *r; ++r) {
        for (const char* s = SUITS; *s; ++s) {
          std::string card;
          card += *r;
          card += *s;
          bool isMyCard = false;
          for (const auto& c : myCards) {
            if (c == card) {
              isMyCard = true;
              break;
            }
          }
          if (!isMyCard) {
            REMAINING_DECK.push_back(card);
          }
        }
      }
    }
    if (street == 2) {
      // Flop is revealed - remove board cards
      for (const auto& c : boardCards) {
        REMAINING_DECK.erase(
          std::remove(REMAINING_DECK.begin(), REMAINING_DECK.end(), c),
          REMAINING_DECK.end()
        );
      }
    }
    if (street == 3) {
      // After first discard - remove the newly discarded card
      if (!boardCards.empty()) {
        REMAINING_DECK.erase(
          std::remove(REMAINING_DECK.begin(), REMAINING_DECK.end(), boardCards.back()),
          REMAINING_DECK.end()
        );
      }
    }
    if (street == 4) {
      // After second discard - remove the newly discarded card
      if (!boardCards.empty()) {
        REMAINING_DECK.erase(
          std::remove(REMAINING_DECK.begin(), REMAINING_DECK.end(), boardCards.back()),
          REMAINING_DECK.end()
        );
      }
    }
    if (street == 5) {
      // Turn is revealed
      if (!boardCards.empty()) {
        REMAINING_DECK.erase(
          std::remove(REMAINING_DECK.begin(), REMAINING_DECK.end(), boardCards.back()),
          REMAINING_DECK.end()
        );
      }
    }
    if (street == 6) {
      // River is revealed
      if (!boardCards.empty()) {
        REMAINING_DECK.erase(
          std::remove(REMAINING_DECK.begin(), REMAINING_DECK.end(), boardCards.back()),
          REMAINING_DECK.end()
        );
      }
    }

    // Handle DISCARD action
    if (legalActions.find(Action::Type::DISCARD) != legalActions.end()) {
      int discardIdx = chooseDiscardMc(myCards, boardCards);
      return {Action::Type::DISCARD, 0, discardIdx};
    }

    // Calculate win probability and EV
    double winProbability = calcWinningProb(myCards, boardCards, street);
    double ev = winProbability * (myContribution + oppContribution) -
                (1 - winProbability) * continueCost;

    // EV negative - fold or check
    if (ev < 0) {
      if (legalActions.find(Action::Type::CHECK) != legalActions.end()) {
        return {Action::Type::CHECK};
      }
      return {Action::Type::FOLD};
    }

    // EV positive - consider raising
    if (legalActions.find(Action::Type::RAISE) != legalActions.end()) {
      // play conservatively by calling
      if (legalActions.find(Action::Type::CALL) != legalActions.end()) {
        return {Action::Type::CALL};
      }
      if (legalActions.find(Action::Type::CHECK) != legalActions.end()) {
        return {Action::Type::CHECK};
      }

      // fallback raise
      // auto raiseBounds = roundState->raiseBounds();
      // int minRaise = raiseBounds[0];
      // int maxRaise = raiseBounds[1];
      // int raiseVal = (int)std::min(
      //   (double)maxRaise,
      //   3.0 * winProbability * (myContribution + oppContribution)
      // );
      
      // return {Action::Type::RAISE, std::max(minRaise, raiseVal)};
    }

    // Fallback
    if (legalActions.find(Action::Type::CHECK) != legalActions.end()) {
      return {Action::Type::CHECK};
    }
    if (legalActions.find(Action::Type::CALL) != legalActions.end()) {
      return {Action::Type::CALL};
    }
    return {Action::Type::FOLD};
  }
};

/*
  Main program for running a C++ pokerbot.
*/
int main(int argc, char* argv[]) {
  auto [host, port] = parseArgs(argc, argv);
  runBot<Bot>(host, port);
  return 0;
}
